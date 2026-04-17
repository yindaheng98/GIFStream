"""
GIFStream Decode Performance Benchmark
=======================================
Profiles each stage of the GIFStream decoding pipeline using synthetic data
that matches real-world scale. Since no trained model is available, we create
mock compressed artifacts and realistic-scale tensors to benchmark:

  1. Entropy decoding (bitstream → quantized parameters)
  2. Post-processing (re-voxelization, inverse sigmoid, etc.)
  3. Neural Gaussian decoding (MLP forward passes)
  4. Rasterization (CUDA splatting)
  5. Full frame decode (end-to-end per-frame latency)
"""

import os
import sys

msvc_bin = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
if os.path.isdir(msvc_bin) and msvc_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = msvc_bin + os.pathsep + os.environ.get("PATH", "")
import json
import time
import math
import tempfile
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))

from gsplat.compression_simulation.entropy_model import ConditionEntropy
from gsplat.compression.gifstream_end2end_compression import (
    GIFStreamEnd2endCompression,
    _compress_png_16bit,
    _decompress_png_16bit,
    _compress_end2end,
    _decompress_end2end,
    _compress_end2end_ar,
    _decompress_end2end_ar,
    _compress_npz,
    _decompress_npz,
)
try:
    from gsplat.rendering import rasterization, view_to_visible_anchors
    GSPLAT_CUDA_AVAILABLE = True
except Exception:
    GSPLAT_CUDA_AVAILABLE = False
    rasterization = None
    view_to_visible_anchors = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Realistic scene parameters (from coffee_martini @ data_factor=2) ──────────
N_ANCHORS = 200_000
N_OFFSETS = 5
ANCHOR_FEAT_DIM = 24
C_PERFRAME = 8
GOP_SIZE = 60
TIME_DIM = 16
IMAGE_W, IMAGE_H = 1014, 756
N_WARMUP = 3
N_REPEAT = 20


# ─────────────────────────── Helpers ──────────────────────────────────────────

def create_mock_splats(n_anchors: int, device: str) -> Dict[str, torch.Tensor]:
    return {
        "anchors": torch.randn(n_anchors, 3, device=device) * 2,
        "scales": torch.randn(n_anchors, 6, device=device) * 0.5 - 4,
        "quats": F.normalize(torch.randn(n_anchors, 4, device=device), dim=-1),
        "opacities": torch.randn(n_anchors, 1, device=device),
        "anchor_features": torch.randn(n_anchors, ANCHOR_FEAT_DIM, device=device),
        "offsets": torch.randn(n_anchors, N_OFFSETS, 3, device=device) * 0.1,
        "factors": torch.sigmoid(torch.randn(n_anchors, 4, device=device)),
        "time_features": torch.randn(n_anchors, GOP_SIZE, C_PERFRAME, device=device),
    }


def create_mock_entropy_models(device: str) -> Dict[str, ConditionEntropy]:
    models = {
        "anchors": None,
        "quats": None,
        "opacities": None,
        "scales": ConditionEntropy(ANCHOR_FEAT_DIM, 18, 8).to(device),
        "offsets": ConditionEntropy(ANCHOR_FEAT_DIM, 9 * N_OFFSETS, 16).to(device),
        "anchor_features": ConditionEntropy(3 * C_PERFRAME, 3 * C_PERFRAME, 12).to(device),
        "factors": ConditionEntropy(ANCHOR_FEAT_DIM, 12, 8).to(device),
        "time_features": ConditionEntropy(3 * C_PERFRAME, 3 * C_PERFRAME, 12).to(device),
    }
    for m in models.values():
        if m is not None:
            m.eval()
    return models


def create_mock_decoders(device: str) -> nn.ModuleDict:
    mlp_opacity = nn.Sequential(
        nn.Linear(ANCHOR_FEAT_DIM + C_PERFRAME, ANCHOR_FEAT_DIM),
        nn.ReLU(True),
        nn.Linear(ANCHOR_FEAT_DIM, N_OFFSETS),
        nn.Tanh(),
    ).to(device)

    mlp_cov = nn.Sequential(
        nn.Linear(ANCHOR_FEAT_DIM + C_PERFRAME, ANCHOR_FEAT_DIM),
        nn.ReLU(True),
        nn.Linear(ANCHOR_FEAT_DIM, 7 * N_OFFSETS),
    ).to(device)

    mlp_color = nn.Sequential(
        nn.Linear(ANCHOR_FEAT_DIM + C_PERFRAME, ANCHOR_FEAT_DIM),
        nn.ReLU(True),
        nn.Linear(ANCHOR_FEAT_DIM, 3 * N_OFFSETS),
        nn.Sigmoid(),
    ).to(device)

    mlp_motion = nn.Sequential(
        nn.Linear(ANCHOR_FEAT_DIM + TIME_DIM + C_PERFRAME, ANCHOR_FEAT_DIM),
        nn.ReLU(True),
        nn.Linear(ANCHOR_FEAT_DIM, 7),
    ).to(device)

    decoders = nn.ModuleDict({
        "mlp_opacity": mlp_opacity,
        "mlp_cov": mlp_cov,
        "mlp_color": mlp_color,
        "mlp_motion": mlp_motion,
    })
    decoders.eval()
    return decoders


def create_mock_camera(device: str):
    c2w = torch.eye(4, device=device).unsqueeze(0)
    c2w[0, 2, 3] = 4.0
    K = torch.tensor([[500.0, 0, IMAGE_W / 2],
                       [0, 500.0, IMAGE_H / 2],
                       [0, 0, 1]], device=device).unsqueeze(0)
    return c2w, K


def quaternion_to_rotation_matrix(q):
    if q.dim() == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


class Timer:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.records = defaultdict(list)

    def time_fn(self, name, fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
        for _ in range(n_warmup):
            fn()

        if self.use_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(n_repeat):
            if self.use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn()
            if self.use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        self.records[name] = times
        return result

    def report(self):
        print("\n" + "=" * 80)
        print(f"{'Stage':<50} {'Mean(ms)':>10} {'Std(ms)':>10} {'Min(ms)':>10}")
        print("-" * 80)
        total = 0
        for name, times in self.records.items():
            mean = np.mean(times)
            std = np.std(times)
            mn = np.min(times)
            total += mean
            print(f"{name:<50} {mean:>10.3f} {std:>10.3f} {mn:>10.3f}")
        print("-" * 80)
        print(f"{'TOTAL (summed means)':<50} {total:>10.3f}")
        print("=" * 80)
        return dict(self.records)


# ──────────────── Benchmark 1: Entropy Decoding (Bitstream I/O) ──────────────

def benchmark_entropy_decoding(timer: Timer):
    print("\n[Benchmark 1] Entropy Decoding (bitstream → parameters)")
    print("-" * 60)

    splats = create_mock_splats(N_ANCHORS, DEVICE)
    entropy_models = create_mock_entropy_models(DEVICE)
    compress_dir = tempfile.mkdtemp(prefix="gifstream_bench_")

    scaling = {
        "anchors": None, "scales": 0.01, "quats": None, "opacities": None,
        "anchor_features": 1, "offsets": 0.01, "factors": 1 / 16, "time_features": 1,
    }

    compressor = GIFStreamEnd2endCompression(use_sort=False, verbose=False)

    try:
        # --- Compress first to create valid bitstreams ---
        pruning_mask = splats["factors"][:, -1] > 0
        for k, v in splats.items():
            splats[k] = v[pruning_mask]

        n_gs = len(splats["anchors"])
        n_sidelen = math.ceil(n_gs ** 0.5)
        n_target = n_sidelen ** 2
        if n_gs > n_target:
            for k, v in splats.items():
                splats[k] = v[:n_target]
        elif n_gs < n_target:
            for k, v in splats.items():
                pad = torch.zeros([n_target - n_gs] + list(v.shape[1:]), device=v.device)
                splats[k] = torch.cat([v, pad], dim=0)
        n_gs = n_target

        splats["quats"] = F.normalize(splats["quats"], dim=-1)
        choose_idx = splats["factors"][:, 0] > 0
        time_feats_selected = splats["time_features"][choose_idx]

        print(f"  Compressing {n_gs} anchors for benchmark...")

        # Compress anchors (PNG 16-bit)
        meta_anchors = _compress_png_16bit(
            compress_dir, "anchors", splats["anchors"].cpu(),
            n_sidelen=n_sidelen, voxel_size=0.01
        )

        # Compress anchor_features (autoregressive)
        af_q = torch.round(splats["anchor_features"] / scaling["anchor_features"]) * scaling["anchor_features"]
        meta_af = _compress_end2end_ar(
            compress_dir, "anchor_features", splats["anchor_features"],
            n_sidelen=n_sidelen, scaling=scaling["anchor_features"],
            entropy_model=entropy_models["anchor_features"],
            anchor_features=af_q, c_channel=C_PERFRAME, p_channel=C_PERFRAME,
            verbose=False
        )

        # Compress scales
        meta_scales = _compress_end2end(
            compress_dir, "scales", splats["scales"],
            n_sidelen=n_sidelen, scaling=scaling["scales"],
            entropy_model=entropy_models["scales"],
            anchor_features=af_q, c_channel=C_PERFRAME, p_channel=C_PERFRAME,
            verbose=False
        )

        # Compress offsets
        meta_offsets = _compress_end2end(
            compress_dir, "offsets", splats["offsets"].flatten(1),
            n_sidelen=n_sidelen, scaling=scaling["offsets"],
            entropy_model=entropy_models["offsets"],
            anchor_features=af_q, c_channel=C_PERFRAME, p_channel=C_PERFRAME,
            verbose=False
        )

        # Compress factors
        meta_factors = _compress_end2end(
            compress_dir, "factors", splats["factors"],
            n_sidelen=n_sidelen, scaling=scaling["factors"],
            entropy_model=entropy_models["factors"],
            anchor_features=af_q, c_channel=C_PERFRAME, p_channel=C_PERFRAME,
            verbose=False
        )

        # Compress time_features (autoregressive)
        meta_tf = _compress_end2end_ar(
            compress_dir, "time_features", time_feats_selected,
            n_sidelen=n_sidelen, scaling=scaling["time_features"],
            entropy_model=entropy_models["time_features"],
            anchor_features=af_q[:time_feats_selected.shape[0]],
            c_channel=C_PERFRAME, p_channel=C_PERFRAME, verbose=False
        )

        # Compress quats/opacities (npz)
        meta_quats = _compress_npz(compress_dir, "quats", splats["quats"].cpu())
        meta_opacities = _compress_npz(compress_dir, "opacities", splats["opacities"].cpu())

        print("  Compression done. Starting decode benchmarks...\n")

        # --- Benchmark individual decode steps ---

        # 1a. PNG 16-bit decode (anchors)
        timer.time_fn("1a. Decode anchors (PNG 16-bit)", lambda: _decompress_png_16bit(
            compress_dir, "anchors", meta_anchors, device=DEVICE,
            entropy_model=None, anchor_features=None
        ))
        print(f"  1a. anchors decode: {np.mean(timer.records['1a. Decode anchors (PNG 16-bit)']):.3f} ms")

        # 1b. Autoregressive decode (anchor_features) -- KEY BOTTLENECK
        def decode_af():
            return _decompress_end2end_ar(
                compress_dir, "anchor_features", meta_af,
                entropy_model=entropy_models["anchor_features"], device=DEVICE,
                anchor_features=None
            )
        af_result = timer.time_fn("1b. Decode anchor_features (AR entropy)", decode_af)
        print(f"  1b. anchor_features AR decode: {np.mean(timer.records['1b. Decode anchor_features (AR entropy)']):.3f} ms")

        # 1c. Conditional entropy decode (scales)
        def decode_scales():
            return _decompress_end2end(
                compress_dir, "scales", meta_scales,
                entropy_model=entropy_models["scales"], device=DEVICE,
                anchor_features=af_result
            )
        timer.time_fn("1c. Decode scales (conditional entropy)", decode_scales)
        print(f"  1c. scales decode: {np.mean(timer.records['1c. Decode scales (conditional entropy)']):.3f} ms")

        # 1d. Conditional entropy decode (offsets)
        def decode_offsets():
            return _decompress_end2end(
                compress_dir, "offsets", meta_offsets,
                entropy_model=entropy_models["offsets"], device=DEVICE,
                anchor_features=af_result
            )
        timer.time_fn("1d. Decode offsets (conditional entropy)", decode_offsets)
        print(f"  1d. offsets decode: {np.mean(timer.records['1d. Decode offsets (conditional entropy)']):.3f} ms")

        # 1e. Conditional entropy decode (factors)
        def decode_factors():
            return _decompress_end2end(
                compress_dir, "factors", meta_factors,
                entropy_model=entropy_models["factors"], device=DEVICE,
                anchor_features=af_result
            )
        timer.time_fn("1e. Decode factors (conditional entropy)", decode_factors)
        print(f"  1e. factors decode: {np.mean(timer.records['1e. Decode factors (conditional entropy)']):.3f} ms")

        # 1f. Autoregressive decode (time_features) -- KEY BOTTLENECK
        def decode_tf():
            return _decompress_end2end_ar(
                compress_dir, "time_features", meta_tf,
                entropy_model=entropy_models["time_features"], device=DEVICE,
                anchor_features=None
            )
        timer.time_fn("1f. Decode time_features (AR entropy)", decode_tf)
        print(f"  1f. time_features AR decode: {np.mean(timer.records['1f. Decode time_features (AR entropy)']):.3f} ms")

        # 1g. NPZ decode (quats + opacities)
        def decode_npz_all():
            q = _decompress_npz(compress_dir, "quats", meta_quats)
            o = _decompress_npz(compress_dir, "opacities", meta_opacities)
            return q, o
        timer.time_fn("1g. Decode quats+opacities (NPZ)", decode_npz_all)
        print(f"  1g. quats+opacities NPZ decode: {np.mean(timer.records['1g. Decode quats+opacities (NPZ)']):.3f} ms")

    finally:
        shutil.rmtree(compress_dir, ignore_errors=True)


# ────────────── Benchmark 2: Post-processing after decode ─────────────────

def benchmark_postprocessing(timer: Timer):
    print("\n[Benchmark 2] Post-processing (re-voxelize, recover features, inverse sigmoid)")
    print("-" * 60)

    splats = create_mock_splats(N_ANCHORS, DEVICE)
    voxel_size = 0.01

    def postprocess():
        mask = splats["quats"].any(dim=1) != 0
        result = {}
        for k, v in splats.items():
            if k != "time_features":
                result[k] = v[mask]
            else:
                result[k] = v

        result["anchors"] = torch.round(result["anchors"] / voxel_size) * voxel_size

        choose_idx = result["factors"][:, 0] > 0
        tf = torch.zeros_like(result["time_features"])
        n_chosen = choose_idx.sum().item()
        tf[choose_idx] = result["time_features"][:n_chosen]
        result["time_features"] = tf
        result["factors"] = -torch.log(1 / (result["factors"].clamp(1e-7, 1 - 1e-7)) - 1)
        return result

    timer.time_fn("2. Post-processing", postprocess)
    print(f"  Post-processing: {np.mean(timer.records['2. Post-processing']):.3f} ms")


# ────────────── Benchmark 3: Visibility check ─────────────────────────────

def benchmark_visibility(timer: Timer):
    print("\n[Benchmark 3] Anchor Visibility Check (view_to_visible_anchors)")
    print("-" * 60)

    splats = create_mock_splats(N_ANCHORS, DEVICE)
    c2w, K = create_mock_camera(DEVICE)

    if not GSPLAT_CUDA_AVAILABLE or view_to_visible_anchors is None:
        print("  SKIPPED: gsplat CUDA extension not available")
        mask = torch.rand(N_ANCHORS, device=DEVICE) > 0.5
        timer.records["3. Visibility check (SKIPPED)"] = [0.0]
        return mask

    try:
        test_mask = view_to_visible_anchors(
            means=splats["anchors"][:10],
            quats=splats["quats"][:10],
            scales=torch.exp(splats["scales"][:10, :3]),
            viewmats=torch.linalg.inv(c2w),
            Ks=K, width=IMAGE_W, height=IMAGE_H,
            packed=False, rasterize_mode="classic",
        )
    except Exception as e:
        print(f"  SKIPPED: gsplat CUDA extension failed: {e}")
        mask = torch.rand(N_ANCHORS, device=DEVICE) > 0.5
        timer.records["3. Visibility check (SKIPPED)"] = [0.0]
        return mask

    def check_visibility():
        return view_to_visible_anchors(
            means=splats["anchors"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"][:, :3]),
            viewmats=torch.linalg.inv(c2w),
            Ks=K, width=IMAGE_W, height=IMAGE_H,
            packed=False, rasterize_mode="classic",
        )

    mask = timer.time_fn("3. Visibility check", check_visibility)
    n_visible = mask.sum().item()
    print(f"  Visible anchors: {n_visible}/{N_ANCHORS} ({100*n_visible/N_ANCHORS:.1f}%)")
    print(f"  Visibility check: {np.mean(timer.records['3. Visibility check']):.3f} ms")
    return mask


# ────────────── Benchmark 4: MLP forward passes ──────────────────────────

def benchmark_mlp_forward(timer: Timer, visible_mask: torch.Tensor):
    print("\n[Benchmark 4] MLP Forward Passes (Neural Gaussian Decoding)")
    print("-" * 60)

    n_visible = visible_mask.sum().item()
    decoders = create_mock_decoders(DEVICE)

    feat_dim = ANCHOR_FEAT_DIM + C_PERFRAME
    feat_input = torch.randn(n_visible, feat_dim, device=DEVICE)
    motion_input = torch.randn(n_visible, ANCHOR_FEAT_DIM + TIME_DIM + C_PERFRAME, device=DEVICE)

    with torch.no_grad():
        timer.time_fn("4a. MLP opacity forward", lambda: decoders["mlp_opacity"](feat_input))
        print(f"  4a. mlp_opacity ({n_visible} anchors): {np.mean(timer.records['4a. MLP opacity forward']):.3f} ms")

        timer.time_fn("4b. MLP color forward", lambda: decoders["mlp_color"](feat_input))
        print(f"  4b. mlp_color: {np.mean(timer.records['4b. MLP color forward']):.3f} ms")

        timer.time_fn("4c. MLP cov forward", lambda: decoders["mlp_cov"](feat_input))
        print(f"  4c. mlp_cov: {np.mean(timer.records['4c. MLP cov forward']):.3f} ms")

        timer.time_fn("4d. MLP motion forward", lambda: decoders["mlp_motion"](motion_input))
        print(f"  4d. mlp_motion: {np.mean(timer.records['4d. MLP motion forward']):.3f} ms")

        def all_mlps():
            o = decoders["mlp_opacity"](feat_input)
            c = decoders["mlp_color"](feat_input)
            s = decoders["mlp_cov"](feat_input)
            m = decoders["mlp_motion"](motion_input)
            return o, c, s, m
        timer.time_fn("4e. All MLPs combined", all_mlps)
        print(f"  4e. All MLPs combined: {np.mean(timer.records['4e. All MLPs combined']):.3f} ms")


# ────────────── Benchmark 5: Feature Assembly + Gaussian Construction ─────

def benchmark_feature_assembly(timer: Timer, visible_mask: torch.Tensor):
    print("\n[Benchmark 5] Feature Assembly (select, aggregate, transform)")
    print("-" * 60)

    splats = create_mock_splats(N_ANCHORS, DEVICE)
    n_visible = visible_mask.sum().item()

    selected_anchors = splats["anchors"][visible_mask]
    selected_offsets = splats["offsets"][visible_mask]
    selected_scales = torch.exp(splats["scales"][visible_mask])
    selected_features = splats["anchor_features"][visible_mask]
    selected_time_features = splats["time_features"][visible_mask][:, 0]
    selected_factors = splats["factors"][visible_mask]

    motion = torch.randn(n_visible, 7, device=DEVICE) * 0.01
    neural_opacity = torch.rand(n_visible * N_OFFSETS, 1, device=DEVICE)
    neural_colors = torch.rand(n_visible * N_OFFSETS, 3, device=DEVICE)
    neural_scale_rot = torch.randn(n_visible * N_OFFSETS, 7, device=DEVICE)

    def assemble_gaussians():
        anchor_offset = motion[:, :3]
        anchors = selected_anchors + anchor_offset
        anchor_rot = F.normalize(0.1 * motion[:, 3:] + torch.tensor([[1, 0, 0, 0]], device=DEVICE))
        anchor_rotation = quaternion_to_rotation_matrix(anchor_rot)
        offsets = torch.bmm(
            selected_offsets.view(-1, N_OFFSETS, 3) * selected_scales.unsqueeze(1)[:, :, :3],
            anchor_rotation.transpose(1, 2)
        ).reshape(-1, 3)

        neural_mask = (neural_opacity > 0.0).view(-1)
        sel_opacity = neural_opacity[neural_mask].squeeze(-1)
        sel_colors = neural_colors[neural_mask]
        sel_scale_rot = neural_scale_rot[neural_mask]
        sel_offsets = offsets[neural_mask]

        scales_rep = selected_scales.unsqueeze(1).repeat(1, N_OFFSETS, 1).view(-1, 6)[neural_mask]
        anchors_rep = anchors.unsqueeze(1).repeat(1, N_OFFSETS, 1).view(-1, 3)[neural_mask]

        final_scales = scales_rep[:, 3:] * torch.sigmoid(sel_scale_rot[:, :3])
        final_rotation = F.normalize(sel_scale_rot[:, 3:7])
        final_means = anchors_rep + sel_offsets
        return final_means, final_scales, final_rotation, sel_colors, sel_opacity

    timer.time_fn("5. Feature assembly + Gaussian construction", assemble_gaussians)
    print(f"  Feature assembly: {np.mean(timer.records['5. Feature assembly + Gaussian construction']):.3f} ms")
    return assemble_gaussians()


# ────────────── Benchmark 6: Rasterization ───────────────────────────────

def benchmark_rasterization(timer: Timer, means, scales, quats, colors, opacities):
    print("\n[Benchmark 6] CUDA Rasterization")
    print("-" * 60)

    c2w, K = create_mock_camera(DEVICE)
    viewmats = torch.linalg.inv(c2w)
    n_gs = means.shape[0]
    print(f"  Rasterizing {n_gs} Gaussians at {IMAGE_W}x{IMAGE_H}")

    def rasterize():
        return rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=K,
            width=IMAGE_W,
            height=IMAGE_H,
            packed=False,
            rasterize_mode="classic",
        )

    try:
        with torch.no_grad():
            timer.time_fn("6. CUDA rasterization", rasterize)
        print(f"  Rasterization: {np.mean(timer.records['6. CUDA rasterization']):.3f} ms")
    except Exception as e:
        print(f"  SKIPPED: gsplat CUDA rasterization failed: {type(e).__name__}")
        timer.records["6. CUDA rasterization (SKIPPED)"] = [0.0]


# ─────────── Benchmark 7: Full Per-Frame Decode (end-to-end) ─────────────

def benchmark_full_frame_decode(timer: Timer):
    print("\n[Benchmark 7] Full Per-Frame Decode Pipeline (visibility → MLP → assemble → rasterize)")
    print("-" * 60)

    splats = create_mock_splats(N_ANCHORS, DEVICE)
    decoders = create_mock_decoders(DEVICE)
    c2w, K = create_mock_camera(DEVICE)
    viewmats = torch.linalg.inv(c2w)

    time_val = 0.5
    feat_start = int(time_val * (GOP_SIZE - 1))
    i = torch.ones(1, dtype=torch.float32)
    time_embedding = torch.cat(
        [torch.sin(2 ** n * torch.pi * i * time_val) for n in range(TIME_DIM // 2)] +
        [torch.cos(2 ** n * torch.pi * i * time_val) for n in range(TIME_DIM // 2)]
    ).to(DEVICE)

    def full_frame():
        # 1) Visibility
        vis_mask = view_to_visible_anchors(
            means=splats["anchors"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"][:, :3]),
            viewmats=viewmats,
            Ks=K,
            width=IMAGE_W,
            height=IMAGE_H,
            packed=False,
            rasterize_mode="classic",
        )

        # 2) Feature selection
        sel_anchors = splats["anchors"][vis_mask]
        sel_offsets = splats["offsets"][vis_mask]
        sel_features = splats["anchor_features"][vis_mask]
        sel_scales = torch.exp(splats["scales"][vis_mask])
        sel_time_feats = splats["time_features"][vis_mask][:, feat_start]
        sel_factors = splats["factors"][vis_mask]
        time_factor = sel_factors[:, 0:1]
        motion_factor = sel_factors[:, 1:2]
        pruning_factor = sel_factors[:, 3:4]

        sel_scales = torch.cat([sel_scales[:, :3], sel_scales[:, 3:] * pruning_factor], dim=-1)

        time_feat_input = torch.cat([sel_features, sel_time_feats * time_factor], dim=-1)
        motion_input = torch.cat([
            sel_features, sel_time_feats * time_factor,
            time_embedding.unsqueeze(0).expand(sel_features.shape[0], -1)
        ], dim=1)

        # 3) MLP decode
        opa = decoders["mlp_opacity"](time_feat_input).view(-1, 1)
        opa = opa * pruning_factor.view(-1, 1).expand(-1, N_OFFSETS).reshape(-1, 1)
        col = decoders["mlp_color"](time_feat_input).view(-1, 3)
        sr = decoders["mlp_cov"](time_feat_input).view(-1, 7)
        mot = decoders["mlp_motion"](motion_input) * motion_factor

        # 4) Assemble Gaussians
        anchor_off = mot[:, :3]
        final_anchors = sel_anchors + anchor_off
        anchor_rot = F.normalize(0.1 * mot[:, 3:] + torch.tensor([[1, 0, 0, 0]], device=DEVICE))
        R = quaternion_to_rotation_matrix(anchor_rot)
        offsets_transformed = torch.bmm(
            sel_offsets.view(-1, N_OFFSETS, 3) * sel_scales.unsqueeze(1)[:, :, :3],
            R.transpose(1, 2)
        ).reshape(-1, 3)

        mask = (opa > 0).view(-1)
        final_opa = opa[mask].squeeze(-1)
        final_col = col[mask]
        final_sr = sr[mask]
        final_off = offsets_transformed[mask]
        final_s_rep = sel_scales.unsqueeze(1).repeat(1, N_OFFSETS, 1).view(-1, 6)[mask]
        final_a_rep = final_anchors.unsqueeze(1).repeat(1, N_OFFSETS, 1).view(-1, 3)[mask]

        final_scales = final_s_rep[:, 3:] * torch.sigmoid(final_sr[:, :3])
        final_quats = F.normalize(final_sr[:, 3:7])
        final_means = final_a_rep + final_off

        # 5) Rasterize
        render_colors, render_alphas, info = rasterization(
            means=final_means,
            quats=final_quats,
            scales=final_scales,
            opacities=final_opa,
            colors=final_col,
            viewmats=viewmats,
            Ks=K,
            width=IMAGE_W,
            height=IMAGE_H,
            packed=False,
            rasterize_mode="classic",
        )
        return render_colors

    try:
        with torch.no_grad():
            timer.time_fn("7. Full per-frame decode", full_frame)
        mean_ms = np.mean(timer.records["7. Full per-frame decode"])
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0
        print(f"  Full frame decode: {mean_ms:.3f} ms ({fps:.1f} FPS)")
    except Exception as e:
        print(f"  SKIPPED: Full frame decode failed (CUDA extension): {type(e).__name__}")
        timer.records["7. Full per-frame decode (SKIPPED)"] = [0.0]


# ─────────────────────── Benchmark 8: Entropy Model Inference ─────────────

def benchmark_entropy_model_inference(timer: Timer):
    print("\n[Benchmark 8] Entropy Model NN Inference (condition → distribution params)")
    print("-" * 60)

    models = create_mock_entropy_models(DEVICE)
    n = N_ANCHORS

    with torch.no_grad():
        # scales: condition = anchor_features (24-dim), output = 18-dim
        x_s = torch.randn(n, 6, device=DEVICE)
        cond_s = torch.randn(n, ANCHOR_FEAT_DIM, device=DEVICE)
        timer.time_fn("8a. Entropy model inference (scales)", lambda: models["scales"](x_s, cond_s, adaptive=True))
        print(f"  8a. scales entropy NN: {np.mean(timer.records['8a. Entropy model inference (scales)']):.3f} ms")

        # offsets: condition = anchor_features (24-dim), output = 45-dim
        x_o = torch.randn(n, N_OFFSETS * 3, device=DEVICE)
        timer.time_fn("8b. Entropy model inference (offsets)", lambda: models["offsets"](x_o, cond_s, adaptive=True))
        print(f"  8b. offsets entropy NN: {np.mean(timer.records['8b. Entropy model inference (offsets)']):.3f} ms")

        # anchor_features AR: each step decodes c_channel dims
        cond_af = torch.randn(n, 3 * C_PERFRAME, device=DEVICE)
        x_af = torch.randn(n, C_PERFRAME, device=DEVICE)
        timer.time_fn("8c. Entropy model inference (AF single AR step)", lambda: models["anchor_features"](x_af, cond_af))
        n_ar_steps = ANCHOR_FEAT_DIM // C_PERFRAME
        print(f"  8c. anchor_features single AR step: {np.mean(timer.records['8c. Entropy model inference (AF single AR step)']):.3f} ms")
        print(f"       (x{n_ar_steps} AR steps = ~{n_ar_steps * np.mean(timer.records['8c. Entropy model inference (AF single AR step)']):.1f} ms total NN inference)")


# ─────────────────────────────── Main ────────────────────────────────────

def main():
    print("=" * 80)
    print("GIFStream Decode Performance Benchmark")
    print("=" * 80)
    print(f"Device:          {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    print(f"PyTorch:         {torch.__version__}")
    print(f"N_ANCHORS:       {N_ANCHORS}")
    print(f"N_OFFSETS:       {N_OFFSETS}")
    print(f"ANCHOR_FEAT_DIM: {ANCHOR_FEAT_DIM}")
    print(f"C_PERFRAME:      {C_PERFRAME}")
    print(f"GOP_SIZE:        {GOP_SIZE}")
    print(f"IMAGE_SIZE:      {IMAGE_W}x{IMAGE_H}")
    print(f"WARMUP/REPEAT:   {N_WARMUP}/{N_REPEAT}")

    timer = Timer(use_cuda=True)

    # Run individual benchmarks
    benchmark_entropy_decoding(timer)
    benchmark_postprocessing(timer)
    vis_mask = benchmark_visibility(timer)
    benchmark_mlp_forward(timer, vis_mask)
    means, scales, quats, colors, opacities = benchmark_feature_assembly(timer, vis_mask)
    benchmark_rasterization(timer, means, scales, quats, colors, opacities)
    benchmark_entropy_model_inference(timer)
    benchmark_full_frame_decode(timer)

    # Final summary
    records = timer.report()

    # Bottleneck analysis
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    sorted_stages = sorted(records.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print("\nTop 5 slowest stages:")
    for i, (name, times) in enumerate(sorted_stages[:5]):
        mean = np.mean(times)
        print(f"  {i+1}. {name}: {mean:.3f} ms")

    full_frame_ms = np.mean(records.get("7. Full per-frame decode", [0]))
    entropy_stages = [k for k in records if k.startswith("1")]
    entropy_total = sum(np.mean(records[k]) for k in entropy_stages)
    mlp_total = np.mean(records.get("4e. All MLPs combined", [0]))
    raster_total = np.mean(records.get("6. CUDA rasterization", [0]))
    vis_total = np.mean(records.get("3. Visibility check", [0]))
    assembly_total = np.mean(records.get("5. Feature assembly + Gaussian construction", [0]))

    print(f"\nPer-frame decode breakdown:")
    print(f"  Visibility check:    {vis_total:.3f} ms")
    print(f"  MLP forward passes:  {mlp_total:.3f} ms")
    print(f"  Feature assembly:    {assembly_total:.3f} ms")
    print(f"  Rasterization:       {raster_total:.3f} ms")
    print(f"  Full frame (measured): {full_frame_ms:.3f} ms ({1000/full_frame_ms:.1f} FPS)")

    print(f"\nOne-time bitstream decode cost (per-GOP):")
    print(f"  Total entropy decode: {entropy_total:.3f} ms")

    ar_af = np.mean(records.get("1b. Decode anchor_features (AR entropy)", [0]))
    ar_tf = np.mean(records.get("1f. Decode time_features (AR entropy)", [0]))
    print(f"    of which AR decode (anchor_features): {ar_af:.3f} ms ({100*ar_af/max(entropy_total,1e-6):.1f}%)")
    print(f"    of which AR decode (time_features):   {ar_tf:.3f} ms ({100*ar_tf/max(entropy_total,1e-6):.1f}%)")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "benchmark_decode_results.json")
    json_results = {k: {"mean_ms": float(np.mean(v)), "std_ms": float(np.std(v)), "min_ms": float(np.min(v))} for k, v in records.items()}
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

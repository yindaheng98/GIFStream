from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .base import Strategy
from .ops import _update_param_with_optimizer
from functools import reduce
from typing_extensions import Literal
try:
    from torch_scatter import scatter_max
except ImportError:
    scatter_max = None

@dataclass
class GIFStreamStrategy(Strategy):
    """A default strategy that follows the original Scaffold-GS paper:
    
    """

    update_depth = 3
    update_init_factor = 16
    update_hierachy_factor = 4
    percent_dense = 0.01
    success_threshold = 0.8
    check_interval = 100
    densify_grad_threshold: float = 0.00009

    prune_opa: float = 0.005
    cap_max = None
    deformation_gate: float = 0.04
    # grow_grad2d: float = 0.0002
    # grow_scale3d: float = 0.01
    # grow_scale2d: float = 0.05
    # prune_scale3d: float = 0.1
    # prune_scale2d: float = 0.15
    # refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    # reset_every: int = 3000
    refine_every: int = 100
    # pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = True
    key_for_gradient: Literal["means2d"] = "means2d"

    def initialize_state(self, scene_scale: float = 1.0, n_offsets: int = 5, voxel_size: float = 0.001, anchor_feature_dim: int = 32 ) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"offset_grad2d": None, "scene_scale": scene_scale, "offset_demon": None, "opacity_accum": None, "anchor_demon": None, "n_offsets": n_offsets, "voxel_size": voxel_size, "anchor_feature_dim": anchor_feature_dim}
        # if self.refine_scale2d_stop_iter > 0:
        #     state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["anchors", "scales", "quats", "opacities","offsets", "anchor_features", "factors", "time_features"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        mask = None,
        max_steps: int = 30000,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            if step % (5 * self.refine_every) == 0 and step < max_steps * 2 / 3 and mask is not None:
                self.pruning_gs_according_mask(params, optimizers, state, step, mask)
        else:
            if step > self.refine_start_iter:
                self._update_state(params, state, info, packed=packed)

            if (
                step > self.refine_start_iter
                and step % self.refine_every == 0
            ):
                # grow GSs
                n_anchors = self._grow_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Now having {(params['anchors'].shape[0])} anchors."
                    )
                torch.cuda.empty_cache()

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        batch_info: Dict[str, Any],
        packed: bool = False,
    ):
        for info in batch_info:
            if info is None: continue
            for key in [
                "width",
                "height",
                "n_cameras",
                "radii",
                "gaussian_ids",
                self.key_for_gradient,
            ]:
                assert key in info, f"{key} is required but missing."

            # normalize grads to [-1, 1] screen space
            if self.absgrad:
                grads = info[self.key_for_gradient].absgrad.clone()
            else:
                grads = info[self.key_for_gradient].grad.clone()
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

            # initialize state on the first run
            n_gaussian = params["anchors"].shape[0]

            if state["offset_grad2d"] is None:
                state["offset_grad2d"] = torch.zeros((n_gaussian*state["n_offsets"],info["gop"]), device=grads.device)
            if state["offset_demon"] is None:
                state["offset_demon"] = torch.zeros((n_gaussian*state["n_offsets"],info["gop"]), device=grads.device)
            if state["opacity_accum"] is None:
                state["opacity_accum"] = torch.zeros((n_gaussian,info["gop"]), device=grads.device)
            if state["anchor_demon"] is None:
                state["anchor_demon"] = torch.zeros((n_gaussian,info["gop"]), device=grads.device)

            anchor_visible_mask = info["anchor_visible_mask"]
            anchor_ids = anchor_visible_mask.nonzero(as_tuple=False).squeeze(-1)

            temp_opacity = info["neural_opacity"].clone().view(-1).detach()
            temp_opacity[temp_opacity<0] = 0
            temp_opacity = temp_opacity.view([-1, state["n_offsets"]])
            state["opacity_accum"][:,info["time"]].index_add_(
                0, anchor_ids, temp_opacity.sum(dim=1) / len(batch_info)
            )

            state["anchor_demon"][:,info["time"]].index_add_(
                0, anchor_ids, torch.ones_like(anchor_ids, dtype=torch.float32) / len(batch_info)
            )
            anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, state["n_offsets"]]).view(-1)
            combined_mask = torch.zeros_like(state["offset_grad2d"][:,0].squeeze(-1), dtype=torch.bool)
            combined_mask[anchor_visible_mask] = info["neural_selection_mask"]
            temp_mask = combined_mask.clone()
            combined_mask[temp_mask] = info["update_filter"]
            offset_ids = combined_mask.nonzero(as_tuple=False).squeeze(1)

            state["offset_demon"][:,info["time"]].index_add_(
                0, offset_ids, torch.ones_like(offset_ids, dtype=torch.float32) / len(batch_info)
            )
            state["offset_grad2d"][:,info["time"]].index_add_(
                0, offset_ids, grads.squeeze(0)[info["update_filter"].squeeze(0),:2].norm(dim=-1) / len(batch_info)
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        peak_ratio = 0.1
        grads = peak_ratio * state["offset_grad2d"] / state["offset_demon"] + (1-peak_ratio) * state["offset_grad2d"].sum(dim=-1,keepdim=True) / state["offset_demon"].sum(dim=-1,keepdim=True)
        device = grads.device
        grads[grads.isnan()] = 0.0
        grads = grads.max(dim=-1, keepdim=True)[0]
        grads_norm = torch.norm(grads, dim=-1)
        offset_demmon = peak_ratio * state["offset_demon"].max(dim=-1, keepdim=True)[0].repeat(1,state["offset_demon"].shape[-1]) + (1-peak_ratio) * state["offset_demon"].mean(dim=-1,keepdim=True)
        offset_mask = (offset_demmon.sum(dim=-1,keepdim=True) > self.check_interval*self.success_threshold*0.5).squeeze(dim=1)

        # anchors growing
        new_anchors_count = 0
        init_length = state["anchor_demon"].shape[0] * state["n_offsets"]
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = self.densify_grad_threshold * ((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads_norm >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.to(device)
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = state["anchor_demon"].shape[0]*state["n_offsets"] - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device=device)], dim=0)

            all_xyz = params["anchors"].unsqueeze(dim=1) + params["offsets"] * torch.exp(params["scales"][:,:3]).unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = state["voxel_size"]*size_factor
            
            grid_coords = torch.round(params["anchors"] / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            new_anchors_count += candidate_anchor.shape[0]

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().to(device)*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=device).float()
                new_rotation[:,0] = 1.0

                def inverse_sigmoid(x):
                    return torch.log(x / (1 - x))

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device=device))
                
                nonzero_indices = candidate_mask.nonzero(as_tuple=True)[0]
                original_indices = nonzero_indices // state["n_offsets"]
                new_feat = params["anchor_features"][original_indices]
                new_time_features = params["time_features"][original_indices]
                new_factors = params["factors"][original_indices]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_time_features = scatter_max(new_time_features, inverse_indices.unsqueeze(1).expand(-1, new_time_features.size(1)), dim=0)[0][remove_duplicates]
                new_factors = scatter_max(new_factors, inverse_indices.unsqueeze(1).expand(-1, new_factors.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,state["n_offsets"],1]).float().to(device)

                d = {
                    "anchors": candidate_anchor,
                    "scales": new_scaling,
                    "quats": new_rotation,
                    "anchor_features": new_feat,
                    "offsets": new_offsets,
                    "opacities": new_opacities,
                    "time_features": new_time_features,
                    "factors": new_factors
                }
                

                temp_anchor_demon = torch.cat([state["anchor_demon"], torch.zeros([new_opacities.shape[0], state["anchor_demon"].shape[-1]], device=device).float()], dim=0)
                state["anchor_demon"]= temp_anchor_demon

                temp_opacity_accum = torch.cat([state["opacity_accum"], torch.zeros([new_opacities.shape[0], state["opacity_accum"].shape[-1]], device=device).float()], dim=0)
                state["opacity_accum"] = temp_opacity_accum

                torch.cuda.empty_cache()
                self.grow_anchors(params,optimizers,d)
                

        # update state
        state["offset_demon"][offset_mask] = 0
        padding_offset_demon = torch.zeros([params["anchors"].shape[0]*state["n_offsets"] - state["offset_demon"].shape[0], state["offset_demon"].shape[-1]],
                                           dtype=torch.int32, 
                                           device=device)
        state["offset_demon"] = torch.cat([state["offset_demon"], padding_offset_demon], dim=0)

        state["offset_grad2d"][offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([params["anchors"].shape[0]*state["n_offsets"] - state["offset_grad2d"].shape[0], state["offset_grad2d"].shape[-1]],
                                           dtype=torch.int32, 
                                           device=device)
        state["offset_grad2d"] = torch.cat([state["offset_grad2d"], padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        opacity_accum = peak_ratio * state["opacity_accum"].max(dim=-1, keepdim=True)[0].repeat(1,state["opacity_accum"].shape[-1]) + (1-peak_ratio) * state["opacity_accum"].mean(dim=-1,keepdim=True)
        prune_mask = (opacity_accum.sum(dim=-1,keepdim=True) < self.prune_opa*state["anchor_demon"].sum(dim=-1,keepdim=True)).squeeze(dim=1)
        anchor_demon = peak_ratio * state["anchor_demon"].max(dim=-1, keepdim=True)[0].repeat(1,state["anchor_demon"].shape[-1]) + (1-peak_ratio) * state["anchor_demon"].mean(dim=-1,keepdim=True)
        anchors_mask = (anchor_demon.sum(dim=-1,keepdim=True) > self.check_interval*self.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_demon
        offset_demon = state["offset_demon"].view([-1, state["n_offsets"],state["offset_demon"].shape[-1]])[~prune_mask]
        offset_demon = offset_demon.view([-1, state["offset_demon"].shape[-1]])
        state["offset_demon"] = offset_demon

        offset_gradient_accum = state["offset_grad2d"].view([-1, state["n_offsets"],state["offset_grad2d"].shape[-1]])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, state["offset_grad2d"].shape[-1]])
        state["offset_grad2d"] = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            state["opacity_accum"][anchors_mask] = torch.zeros([anchors_mask.sum(), state["opacity_accum"].shape[-1]], device=device).float()
            state["anchor_demon"][anchors_mask] = torch.zeros([anchors_mask.sum(), state["anchor_demon"].shape[-1]], device=device).float()
        
        temp_opacity_accum = state["opacity_accum"][~prune_mask]
        state["opacity_accum"] = temp_opacity_accum

        temp_anchor_demon = state["anchor_demon"][~prune_mask]
        state["anchor_demon"] = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchors(params,optimizers,~prune_mask)
        new_anchors_count -= prune_mask.view((-1)).nonzero(as_tuple=False).shape[0]

        return new_anchors_count
    
    def pruning_gs_according_mask(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        mask: torch.Tensor,
    ) -> Tuple[int, int]:
        prune_mask = mask
        offset_demon = state["offset_demon"].view([-1, state["n_offsets"],state["offset_demon"].shape[-1]])[~prune_mask]
        offset_demon = offset_demon.view([-1, state["offset_demon"].shape[-1]])
        state["offset_demon"] = offset_demon

        offset_gradient_accum = state["offset_grad2d"].view([-1, state["n_offsets"],state["offset_grad2d"].shape[-1]])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, state["offset_grad2d"].shape[-1]])
        state["offset_grad2d"] = offset_gradient_accum
        
        temp_opacity_accum = state["opacity_accum"][~prune_mask]
        state["opacity_accum"] = temp_opacity_accum

        temp_anchor_demon = state["anchor_demon"][~prune_mask]
        state["anchor_demon"] = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchors(params,optimizers,~prune_mask)
    
    @torch.no_grad()
    def grow_anchors(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        new_params: Dict[str, torch.Tensor],
    ):
        num_new = new_params["anchors"].shape[0]
        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name in ["anchors","scales","quats","opacities","offsets","anchor_features","time_features","factors"]:
                p_new = torch.cat([p, new_params[name]], dim=0)
            else:
                raise ValueError(f"Parameter '{name}' not recognized.")
            return torch.nn.Parameter(p_new)
        
        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            # Extend optimizer state tensors with zeros
            zeros = torch.zeros((num_new, *v.shape[1:]), device=params["anchors"].device)
            v_new = torch.cat([v, zeros], dim=0)
            return v_new
        
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    @torch.no_grad()
    def prune_anchors(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            mask,
    ):
        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(p[mask])

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        # update the parameters and the state in the optimizers
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
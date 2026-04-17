# GIFStream 解码性能实验记录

## 1. 实验目的

本实验的目标是：

1. 阅读 `GIFStream` 项目源码，梳理解码路径。
2. 分析整段视频解码中，哪些操作是一次性执行、哪些操作是逐帧执行。
3. 对解码过程做分阶段 profiling，识别主要性能瓶颈。
4. 对比不同 anchor 数量下的解码时间变化，重点观察 `N=50K` 和 `N=200K`。

需要说明的是，本次实验主要分析的是 **解码链路**，不是训练链路。

---

## 2. 代码层面的解码路径

结合源码阅读，GIFStream 的压缩后解码路径主要是：

1. `examples/simple_trainer_GIFStream.py` 中的 `run_compression()`
2. `gsplat/compression/gifstream_end2end_compression.py` 中的 `decompress()`
3. `examples/simple_trainer_GIFStream.py` 中的：
   - `decoding_features()`
   - `get_neural_gaussians()`
   - `rasterize_splats()`

逻辑上可以拆成两段：

### 2.1 一次性操作（per-GOP）

这部分只在一个 GOP 开始时做一次：

- 读取 `nets.pt`
- 恢复 entropy models 和 decoder MLP 权重
- 从 `.bin` / `.png` / `.npz` 中解出：
  - `anchor_features`
  - `time_features`
  - `scales`
  - `offsets`
  - `factors`
  - `anchors`
  - `quats`
  - `opacities`
- 做后处理：
  - re-voxelize
  - 恢复 `time_features`
  - `factors` 做 inverse sigmoid
- 构建 KNN 邻居索引（如果启用）

### 2.2 每帧操作（per-frame）

这部分每渲染一帧都要执行：

- 可见性筛选 `view_to_visible_anchors()`
- 从全局 splats 中索引可见 anchors
- 取当前时间步对应的 `time_features[:, feat_start]`
- 做 KNN 特征聚合
- 计算时间位置编码
- 执行 4 个 MLP：
  - `mlp_opacity`
  - `mlp_color`
  - `mlp_cov`
  - `mlp_motion`
- 组装最终 Gaussian 参数
- 调用 CUDA rasterization 渲染图像

---

## 3. 实验环境

实验机环境：

- OS: Windows 10
- GPU: NVIDIA GeForce RTX 3080
- Python: Conda 环境 `GIFStream`
- PyTorch: `2.10.0+cu130`
- 系统 CUDA Toolkit: `12.4`

---

## 4. 实验前准备

### 4.1 安装 Python 依赖

补充安装了项目运行 benchmark 所需的 Python 包，包括：

- `scipy`
- `imageio`
- `nerfview`
- `viser`
- `tyro`
- `torchmetrics`
- `opencv-python`
- `tensorboard`
- `pyyaml`
- `matplotlib`
- `natsort`
- `tensorly`
- `scikit-learn`
- `tqdm`
- `Pillow`

### 4.2 构建 `MLEntropy`

项目的 entropy coder 依赖 `third_party/MLEntropy` 的 C++ 扩展。

在 Windows 上额外完成了：

1. 使用 CMake 配置 `third_party/MLEntropy/build`
2. 指定 Python 3.10 的解释器、include 和 library
3. 用 VS2022 成功编译：
   - `MLCodec_CXX.cp310-win_amd64.pyd`
   - `MLCodec_rans.cp310-win_amd64.pyd`
4. 将生成的 `.pyd` 复制到 `third_party/MLEntropy/entropy_models/`

### 4.3 gsplat CUDA 扩展问题

`gsplat` 自身的 CUDA extension 在当前环境下未能成功编译，因此：

- `view_to_visible_anchors()`
- `rasterization()`

这两个依赖 gsplat CUDA extension 的阶段无法完成真实实测。

实际报错表现为：

- 系统 CUDA Toolkit 为 12.4
- PyTorch 为 cu130
- JIT 编译 gsplat CUDA extension 时失败
- Windows + 当前 PyTorch 头文件中还出现了 `compiled_autograd.h` 相关编译错误

因此本次实验中：

- **一次性解码阶段**：可以完整实测
- **逐帧 MLP 与特征组装阶段**：可以用 mock 数据实测
- **可见性检查与 rasterization**：由于 gsplat CUDA extension 编译失败，被标记为 `SKIPPED`

---

## 5. Benchmark 设计

为了在没有现成训练 checkpoint 的情况下测试解码链路，编写了 `benchmark_decode.py`。

其设计思路是：

1. 构造与真实项目形状一致的 mock splats：
   - `anchors`
   - `scales`
   - `quats`
   - `opacities`
   - `anchor_features`
   - `offsets`
   - `factors`
   - `time_features`
2. 直接调用项目里的真实压缩/解压函数：
   - `_compress_end2end`
   - `_decompress_end2end`
   - `_compress_end2end_ar`
   - `_decompress_end2end_ar`
   - `_compress_png_16bit`
   - `_decompress_png_16bit`
3. 对以下阶段分别计时：
   - 熵解码
   - 后处理
   - MLP 前向
   - 高斯参数组装
   - entropy model 推理
4. 用 warmup + repeat 的方式减少偶然误差：
   - `N_WARMUP = 3`
   - `N_REPEAT = 20`

### 5.1 Benchmark 当前核心参数

在本次实验的最终版本中，`benchmark_decode.py` 的参数为：

- `N_ANCHORS = 50_000`（第一轮）
- `N_ANCHORS = 200_000`（第二轮）
- `N_OFFSETS = 5`
- `ANCHOR_FEAT_DIM = 24`
- `C_PERFRAME = 8`
- `GOP_SIZE = 60`
- `IMAGE_W, IMAGE_H = 1014, 756`

### 5.2 与真实 `coffee_martini` 配置的差异

这里有一个重要说明：

`coffee_martini` 在项目配置 `neur3d_2` 中更接近：

- `anchor_feature_dim = 48`
- `c_perframe = 4`

而 benchmark 中为了先稳定跑通链路，沿用了：

- `anchor_feature_dim = 24`
- `c_perframe = 8`

所以本实验更适合用来比较：

- 不同 `N` 下解码复杂度的变化趋势
- 哪一类操作是主要瓶颈

而不是直接当作最终产品级的绝对时延数据。

---

## 6. 实验执行过程

### 6.1 第一轮：`N = 50K`

先将：

```python
N_ANCHORS = 50_000
```

运行：

```bash
python benchmark_decode.py
```

得到第一组基线结果。

### 6.2 第二轮：`N = 200K`

之后将：

```python
N_ANCHORS = 200_000
```

再次运行：

```bash
python benchmark_decode.py
```

并将结果写入 `benchmark_decode_results.json`。

当前仓库中的 `benchmark_decode_results.json` 对应的是 **`N=200K` 的最新结果**。

---

## 7. 实验结果

## 7.1 `N = 50K` 结果

### 一次性操作（per-GOP）

| 阶段 | mean (ms) |
|---|---:|
| Decode anchors (PNG 16-bit) | 4.003 |
| Decode anchor_features (AR entropy) | 14.249 |
| Decode scales (conditional entropy) | 3.996 |
| Decode offsets (conditional entropy) | 8.511 |
| Decode factors (conditional entropy) | 1.809 |
| Decode time_features (AR entropy) | 220.337 |
| Decode quats + opacities (NPZ) | 4.318 |
| Post-processing | 2.103 |
| **一次性合计** | **259.326** |

### 每帧相关操作（非 rasterization 部分）

| 阶段 | mean (ms) |
|---|---:|
| MLP opacity forward | 0.776 |
| MLP color forward | 0.853 |
| MLP cov forward | 0.854 |
| MLP motion forward | 1.428 |
| All MLPs combined | 4.182 |
| Feature assembly + Gaussian construction | 2.334 |

### 观察

- `time_features` 的 AR 熵解码是绝对主瓶颈：`220.337 ms`
- 在一次性开销里占比约：

```text
220.337 / 257.224 ≈ 85.7%
```

- 每帧的非 rasterization 计算量很小：
  - MLP 合计约 `4.2 ms`
  - 特征组装约 `2.3 ms`

---

## 7.2 `N = 200K` 结果

### 一次性操作（per-GOP）

| 阶段 | mean (ms) |
|---|---:|
| Decode anchors (PNG 16-bit) | 9.517 |
| Decode anchor_features (AR entropy) | 52.715 |
| Decode scales (conditional entropy) | 13.986 |
| Decode offsets (conditional entropy) | 29.557 |
| Decode factors (conditional entropy) | 11.264 |
| Decode time_features (AR entropy) | 3021.036 |
| Decode quats + opacities (NPZ) | 16.524 |
| Post-processing | 102.105 |
| **一次性合计** | **3256.704** |

### 每帧相关操作（非 rasterization 部分）

| 阶段 | mean (ms) |
|---|---:|
| MLP opacity forward | 0.396 |
| MLP color forward | 0.451 |
| MLP cov forward | 0.642 |
| MLP motion forward | 0.398 |
| All MLPs combined | 1.929 |
| Feature assembly + Gaussian construction | 5.055 |

### 观察

- `time_features` 的 AR 熵解码增长到：

```text
3021.036 ms
```

- 在一次性熵解码里占比约：

```text
3021.036 / 3154.599 ≈ 95.8%
```

- 后处理也明显增大到：

```text
102.105 ms
```

- 每帧 MLP 合计反而下降到 `1.929 ms`，说明较大 batch 的 GPU 并行度更高
- 每帧的特征组装从 `2.334 ms` 增长到 `5.055 ms`

---

## 7.3 `N = 50K` 与 `N = 200K` 对比

| 阶段 | 50K (ms) | 200K (ms) | 倍率 |
|---|---:|---:|---:|
| anchors PNG 解码 | 4.003 | 9.517 | 2.38x |
| anchor_features AR 解码 | 14.249 | 52.715 | 3.70x |
| scales 条件熵解码 | 3.996 | 13.986 | 3.50x |
| offsets 条件熵解码 | 8.511 | 29.557 | 3.47x |
| factors 条件熵解码 | 1.809 | 11.264 | 6.23x |
| **time_features AR 解码** | **220.337** | **3021.036** | **13.71x** |
| quats+opacities NPZ 解码 | 4.318 | 16.524 | 3.83x |
| 后处理 | 2.103 | 102.105 | 48.56x |
| **一次性合计** | **259.326** | **3256.704** | **12.56x** |
| All MLPs combined | 4.182 | 1.929 | 0.46x |
| Feature assembly | 2.334 | 5.055 | 2.17x |

---

## 8. 结果分析

## 8.1 最主要瓶颈：`time_features` 自回归熵解码

这是整个解码链路里最关键的瓶颈。

原因是：

1. `time_features` 覆盖整个 GOP
2. 使用了 AR 解码
3. AR 解码必须串行执行
4. 每一步又包含：
   - 条件网络推理
   - rANS 解码
   - 文件读取
   - 张量拼接

在 `N=50K` 时，它已经占一次性熵解码的 `85.7%`；
在 `N=200K` 时，这个比例进一步上升到 `95.8%`。

这说明：

- 当 anchor 数量进一步增大时，
- 一次性解码几乎完全被 `time_features` 的 AR 熵解码主导。

## 8.2 一次性操作不是线性增长

理论上，`N` 从 50K 到 200K，是 4 倍。

但实际上：

- `time_features` AR 解码是 `13.7x`
- 后处理是 `48.6x`
- 一次性总耗时是 `12.6x`

说明这里已经不只是简单的算术量增加，还包含：

- 更差的 cache locality
- 更重的 tensor 分配/回收
- 更高的 CPU / Python 调度开销
- 更大的内存带宽压力

## 8.3 每帧部分相对可控

在没有 rasterization 实测数据的情况下，能确认的是：

- MLP 推理并不重
- 特征组装有增长，但仍然远低于一次性 AR 解码

也就是说，从当前实验能看到的结论是：

- **首帧/GOP 切换延迟** 才是最严重的问题
- **连续播放时的逐帧 MLP 开销** 反而不是主要矛盾

---

## 9. 本次实验的局限性

本实验仍有几个需要明确说明的限制：

### 9.1 使用的是 synthetic benchmark

`benchmark_decode.py` 使用的是 mock splats 和 mock bitstreams，
而不是实际训练得到的 `coffee_martini` checkpoint。

优点是：

- 可以直接验证项目中的真实压缩/解压函数
- 能稳定复现实验

缺点是：

- 参数分布与真实训练结果并不完全一致
- 不代表真实模型下的最终绝对耗时

### 9.2 当前 benchmark 配置与 `neur3d_2` 不完全一致

真实 `coffee_martini` 更接近：

- `anchor_feature_dim = 48`
- `c_perframe = 4`

当前 benchmark 采用：

- `anchor_feature_dim = 24`
- `c_perframe = 8`

因此本实验更适合做：

- 相对趋势分析
- 瓶颈识别
- 不同 `N` 下复杂度对比

### 9.3 gsplat CUDA extension 未编译成功

所以以下阶段没有拿到真实实测：

- 可见性检查
- rasterization
- 全流程逐帧解码总时延

因此目前文档中的逐帧结果，只能代表：

- MLP
- 特征组装

不能代表完整 per-frame render latency。

---

## 10. 当前可得出的结论

本次实验最重要的结论有三条：

1. **GIFStream 的一次性解码瓶颈几乎完全来自 `time_features` 的 AR 熵解码。**
2. **当 `N` 从 50K 增长到 200K 时，一次性解码开销出现明显超线性增长。**
3. **逐帧部分（至少 MLP + 特征组装）相对较轻，真正阻碍实时体验的是 GOP 切换时的首帧延迟。**

换句话说：

- 如果目标是优化“开始播放/切换 GOP 的卡顿”，应优先优化 `time_features` AR 解码。
- 如果目标是优化“连续播放时的 FPS”，还需要补上 gsplat CUDA extension 的真实 rasterization 测试后再下结论。

---

## 11. 后续建议

如果继续深入，建议按以下顺序进行：

1. 修复 gsplat CUDA extension 的构建问题，拿到真实的：
   - visibility check
   - rasterization
   - full per-frame latency
2. 将 benchmark 参数改成更接近 `neur3d_2`：
   - `anchor_feature_dim = 48`
   - `c_perframe = 4`
3. 使用真实 `coffee_martini` 训练 checkpoint 做一次完整 benchmark
4. 针对 `time_features` AR 解码尝试：
   - 减少 AR 步数
   - 合并 bitstream 文件
   - 用块级并行替代全串行 AR
   - 迁移部分熵解码逻辑到 GPU


# Intel XPU Support

🔥We are excited to announce that Cache-DiT now provides **native** support for **Intel XPU** (Intel Arc GPUs, Intel Data Center GPU Flex Series, and Intel Data Center GPU Max Series). Theoretically, **nearly all** models supported by Cache-DiT can run on Intel XPU with most of Cache-DiT's optimization technologies, including:

- **Hybrid Cache Acceleration** ([**DBCache**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#dbcache-dual-block-cache), DBPrune, [**TaylorSeer**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#hybrid-taylorseer-calibrator), [**SCM**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#scm-steps-computation-masking) and more)
- **Context Parallelism** (w/ Extended Diffusers' CP APIs, [**UAA**](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#uaa-ulysses-anything-attention), Async Ulysses, ...)
- **Tensor Parallelism** (w/ PyTorch native DTensor and Tensor Parallelism APIs)
- **Text Encoder Parallelism** (w/ PyTorch native DTensor and Tensor Parallelism APIs)
- **Auto Encoder (VAE) Parallelism** (w/ Data or Tile Parallelism, avoid OOM)
- **ControlNet Parallelism** (w/ Context Parallelism for ControlNet module)
- Built-in **HTTP serving** deployment support with simple REST APIs

## Features Support

|Device|Hybrid Cache|Context Parallel|Tensor Parallel|Text Encoder Parallel|Auto Encoder(VAE) Parallel|
|:---|:---:|:---:|:---:|:---:|:---:|
|Intel Arc GPU|✅|✅|✅|✅|✅|
|Intel Data Center GPU Flex|✅|✅|✅|✅|✅|
|Intel Data Center GPU Max|✅|✅|✅|✅|✅|

## Attention backend

Cache-DiT supports multiple Attention backends for better performance. The supported attention backends for Intel XPU are listed below:

|backend|details|parallelism|attn_mask|
|:---|:---|:---|:---|
|native| Native SDPA Attention in PyTorch|✅|✅|

We recommend using the `native` (SDPA) backend as it is well-optimized for Intel XPU via Intel Extension for PyTorch.

## Environment Requirements

| Software | Supported version | Note |
|----------|------------------|-------|
| Python   | >= 3.9           | Required |
| PyTorch  | >= 2.3.0         | Built-in `torch.xpu` support |
| Intel Extension for PyTorch (IPEX) | >= 2.3.0 | Required for best performance and distributed training (`ccl` backend) |
| Intel oneAPI Base Toolkit | >= 2024.1 | Required for oneAPI DPC++/C++ Compiler and oneMKL |

## Install Intel XPU Torch

Install PyTorch with Intel XPU support. Intel XPU support is built into PyTorch natively starting from version 2.3.0.

```bash
# Install PyTorch with XPU support (check https://pytorch.org/ for latest instructions)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

### Install Intel Extension for PyTorch (Recommended)

For best performance and distributed training support (`ccl` backend), install Intel Extension for PyTorch:

```bash
pip3 install intel_extension_for_pytorch
# For distributed training (oneCCL bindings)
pip3 install oneccl_bind_pt --index-url https://developer.intel.com/ipex-whl-stable-xpu
```

### Verify XPU Installation

```python
import torch
print(torch.xpu.is_available())       # Should print True
print(torch.xpu.device_count())       # Number of XPU devices
print(torch.xpu.get_device_name())    # Device name, e.g., "Intel(R) Arc(TM) A770 Graphics"
```

## Intel XPU Environment Variables

```bash
# Control visible Intel XPU devices via Level Zero affinity mask
# e.g., expose only device 0 and device 1
export ZE_AFFINITY_MASK=0,1

# Alternatively, use ONEAPI_DEVICE_SELECTOR to filter by backend and device
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# For distributed training with oneCCL
export CCL_WORKER_COUNT=1
```

## Install Cache-DiT Library

You can install the stable release of `cache-dit` from PyPI:

```bash
pip3 install -U cache-dit
```

Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```

Please also install the latest main branch of diffusers:

```bash
pip3 install git+https://github.com/huggingface/diffusers.git # or >= 0.36.0
```

## Examples and Benchmark

After the environment configuration is complete, users can refer to the **[Quick Examples](../EXAMPLES.md)** for more details.

```bash
pip3 install opencv-python-headless einops imageio-ffmpeg ftfy
pip3 install git+https://github.com/huggingface/diffusers.git # latest or >= 0.36.0
pip3 install git+https://github.com/vipshop/cache-dit.git    # latest
```

### Single XPU Inference

The easiest way to enable hybrid cache acceleration for DiTs with cache-dit on Intel XPU is to start with single XPU inference. For examples:

```bash
# use default model path, e.g, "black-forest-labs/FLUX.1-dev"
python3 -m cache_dit.generate flux
python3 -m cache_dit.generate qwen_image
python3 -m cache_dit.generate flux --cache
python3 -m cache_dit.generate qwen_image --cache
```

### Distributed Inference

cache-dit is designed to work with 🔥Context Parallelism, 🔥Tensor Parallelism. For distributed training with Intel XPU, you need `oneccl_bind_pt` installed. For examples:

```bash
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses
torchrun --nproc_per_node=4 -m cache_dit.generate zimage --parallel ulysses
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image --parallel ulysses
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --cache
torchrun --nproc_per_node=4 -m cache_dit.generate zimage --parallel ulysses --cache
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image --parallel ulysses --cache
```

## Notes and Limitations

- **Quantization**: TorchAO quantization (FP8, INT4) is currently supported on CUDA devices only. INT8 weight-only quantization may work on XPU depending on your TorchAO version.
- **Triton Kernels**: Some optimized Triton kernels are CUDA-specific and will fall back to native PyTorch implementations on XPU.
- **Memory Profiling**: CUDA-specific memory snapshot APIs (`torch.cuda.memory._dump_snapshot`) are not available on XPU. The `ProfilerContext` will record CPU and XPU activities but skip CUDA memory snapshots.
- **Device Capability**: Intel XPU does not use CUDA compute capability. Code paths that check `get_device_capability()` (e.g., Hopper-specific optimizations) are automatically skipped on XPU.

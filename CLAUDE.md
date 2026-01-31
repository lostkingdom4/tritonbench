# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TritonBench is a benchmarking suite for PyTorch/Triton operators. The main entrypoint is `run.py` which delegates to `tritonbench.utils.run_utils.tritonbench_run`, wiring CLI parsing to operator loaders and result exporters.

## Installation & Setup

```bash
# Clone and initialize submodules
git clone https://github.com/meta-pytorch/tritonbench.git
cd tritonbench
git submodule update --init --recursive

# Install (PyTorch nightly + requirements + submodules)
python install.py

# Optional extras
python install.py --liger --fa3 --fa2 --fbgemm --mslk --jax --tk --tile --xformers --aiter --all

# Install as library
pip install -e .
```

## Running Benchmarks

```bash
# Basic usage
python run.py --op gemm
python run.py --op gemm,addmm  # Multiple operators

# Operator collections
python run.py --op-collection default
python run.py --op-collection liger
python run.py --op-collection all

# Common options
python run.py --op gemm --only kernel_a,kernel_b  # Run specific kernels
python run.py --op gemm --skip kernel_c           # Skip kernels
python run.py --op gemm --precision fp16          # Set precision (fp16, bf16, etc.)
python run.py --op gemm --device cuda             # Device (cuda, cpu, mtia)
python run.py --op gemm --metrics latency,tflops  # Metrics to collect
python run.py --op gemm --csv                     # CSV output
python run.py --op gemm --output-json results.json
python run.py --op gemm --plot                    # Generate plots
python run.py --op gemm --ci                      # CI mode

# Batch runs via YAML config
TRITONBENCH_RUN_CONFIG=benchmarks/run_config/example.yaml python run.py
```

## Testing

```bash
# CPU tests
pytest test/test_cpu -q

# GPU tests (requires proper drivers)
pytest test/test_gpu -q

# Smoke test with minimal runs
python run.py --op gemm --test-only
python run.py --op gemm --num-inputs 1 --rep 10
```

## Architecture

### Core Components

- **`run.py`**: Main entrypoint that calls `tritonbench_run()`
- **`tritonbench/utils/parser.py`**: CLI argument parsing
- **`tritonbench/utils/run_utils.py`**: Core benchmark orchestration logic
- **`tritonbench/utils/triton_op.py`**: `BenchmarkOperator` base class and registration system
- **`tritonbench/operators/<op>/operator.py`**: Individual operator implementations

### Operator Structure

Each operator lives in `tritonbench/operators/<op>/operator.py` and subclasses `BenchmarkOperator`:

```python
from tritonbench.utils.triton_op import BenchmarkOperator, register_benchmark, register_metric, register_x_val

class Operator(BenchmarkOperator):
    def get_input_iter(self):
        # Generate input shapes/data
        pass

    @register_benchmark(baseline=True)  # Mark as baseline
    def torch_impl(self, *args):
        pass

    @register_benchmark(enabled=True)  # Gate with enabled flag
    def triton_impl(self, *args):
        pass

    @register_metric()
    def tflops(self, fn_name, example_inputs, metrics):
        pass

    @register_x_val(label="Size")
    def get_x_val(self, example_inputs):
        pass
```

### Benchmark Loop

`BenchmarkOperator.run()` handles:
- Warmup/rep counts
- Entropy-based adaptive stopping
- Precision overrides
- Determinism/accuracy checks
- Result export to `.benchmarks/` (or custom `--output*` paths)

### Input Generation

- Operators define shapes via `get_input_iter()`
- Common sources: built-in lists, CSV files (`tritonbench/operators/<op>/*.csv`), YAML configs, `llama_shapes()`, `get_production_shapes()`
- Use `_scaled_randn` for numeric stability
- Honor `--input-id` and `--num-inputs` for sampling

### Kernel Registration

- Use `@register_benchmark` decorator with parameters:
  - `baseline=True`: Mark as baseline for comparison
  - `enabled=False`: Disable by default (can override with `--force`)
  - `fwd_only=True`: Only run forward pass
- Use `@register_metric` for custom metrics
- Use `@register_x_val` for x-axis values in plots
- Gate kernels by hardware: `is_cuda`, `is_fbcode`, `supports_tma`, `has_tlx`

### Feature Detection Helpers

- `tritonbench/utils/env_utils.py`: Environment and precision utilities
- `tritonbench/utils/triton_utils.py`: Triton feature detection (`has_tlx()`, etc.)
- `tritonbench/utils/gpu_utils.py`: GPU utilities (AMD-specific paths)

### Submodules

External kernels live in `submodules/`:
- xformers, flash-attention, ThunderKittens (CUDA)
- Liger-Kernel, tilelang, generative-recommenders (CUDA/HIP)
- FBGEMM (CUDA)
- AITer (HIP)

Installed by `install.py` with optional flags. Avoid modifying without need.

## Profiling & Debug Tools

```bash
python run.py --op gemm --dump-ir          # Dump IR
python run.py --op gemm --power-chart      # Power analysis
python run.py --op gemm --metrics-gpu-backend torch  # or nvml
```

Profiler integration: `tritonbench/components/ncu.py`, `nsys.py`

## Environment Variables

- `TRITONBENCH_RUN_CONFIG`: Path to YAML config in `benchmarks/run_config/`
- `TRITONBENCH_TRITON_COMMIT_HASH`: Override Triton commit hash for repro
- `TRITONBENCH_HELION_PATH`: Helion integration path
- `AMDGCN_USE_BUFFER_OPS`: AMD-specific kernel behavior

## Code Style

- Python, torch, triton only
- Keep instructions ASCII
- Prefer small, composable helpers over inlining
- Reuse existing env/dtype/precision helpers
- Keep kernel functions side-effect free
- Respect device guards and gating logic

# Agentic Benchmark Artifacts

This documents the capture/generate/validate artifacts emitted when running TritonBench with `--agentic-capture|--agentic-generate|--agentic-validate`.

## Directory layout

Artifacts live under `.benchmarks/agentic/<op>/input_<id>/<backend>/` where `<backend>` is the benchmark fn name (baseline first). Key files:

- `manifest.json`: capture manifest for this backend/input
- `ir/`: TTIR/TTGIR/LLIR/PTX/AMDGPU files from `dump_ir`
- `kernel_source.py`: best-effort raw Triton kernel source for the backend
- `generate_request.json`: emitted in generate mode to trigger an agent
- `generated/metrics.json`: agent-written latency result for generated kernel
- `validate.json`: validation summary vs baseline latency (written in validate mode)

## manifest.json schema (v1)

```json
{
  "manifest_version": 1,
  "agentic_mode": "capture" | "generate",
  "op": "<op_name>",
  "benchmark_name": "<benchmark_name>",
  "fn_name": "<backend_name>",
  "baseline": true | false,
  "input_id": <int>,
  "device": "cuda|cpu|mtia",
  "precision": "fp16|bf16|...|" ,
  "dtype": "torch.float16" | null,
  "shapes": { ... tree of tensors -> {"shape": [...], "dtype": "torch.float16", "device": "cuda:0"} },
  "metrics": {
    "latency_ms": <float|null>,
    "latency_min_ms": <float|null>,
    "latency_max_ms": <float|null>,
    "tflops": <float|null>,
    "speedup": <float|null>,
    "accuracy": <bool|null>,
    "determinism": "pass|non_deterministic|fail|null",
    "compile_time_ms": <float|null>,
    "gpu_peak_mem": <float|null>,
    "cpu_peak_mem": <float|null>,
    "extra_metrics": {"<name>": <value>|null},
    "error_msg": <str|null>
  },
  "paths": {
    "artifact_dir": ".../.benchmarks/agentic/<op>/input_<id>/<backend>",
    "ir_dir": ".../ir",
    "kernel_source": ".../kernel_source.py" | null
  }
}
```

## generate_request.json schema (v1)

Emitted in generate mode to instruct an external agent which capture to consume:

```json
{
  "request_version": 1,
  "agentic_mode": "generate",
  "op": "<op_name>",
  "fn_name": "<baseline_backend>",
  "input_id": <int>,
  "manifest": ".../manifest.json"
}
```

Agent contract:
1) Read `generate_request.json`, then `manifest.json`, then `ir/*` and `kernel_source.py` as needed (TTGIR is in `ir/` with suffix `.ttgir`).
2) Produce any generated kernels under `.../<backend>/generated/` (filename freeform).
3) Write `.../<backend>/generated/metrics.json` containing at least:

```json
{
  "latency_ms": <float>
}
```

## validate.json schema (v1)

Produced in validate mode (reads agent-produced `generated/metrics.json`):

```json
{
  "validate_version": 1,
  "agentic_mode": "validate",
  "op": "<op_name>",
  "fn_name": "<baseline_backend>",
  "input_id": <int>,
  "baseline_latency_ms": <float|null>,
  "generated_latency_ms": <float|null>,
  "speedup": <float|null>,
  "status": "pass" | "slower" | "missing-generated" | "invalid-generated",
  "generated_metrics_path": ".../generated/metrics.json"
}
```

Validation rule: pass iff generated latency ≤ baseline latency; no accuracy/other metrics are checked.

## Modes summary

- Capture (`--agentic-capture`): writes manifest + IR + kernel_source per backend/input.
- Generate (`--agentic-generate`): also writes `generate_request.json` per baseline backend/input.
- Validate (`--agentic-validate`): compares `generated/metrics.json` to captured baseline latency, writes `validate.json`.

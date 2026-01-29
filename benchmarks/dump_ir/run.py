"""
Run all available custom Triton operators and save their IRs to a directory.
For autotuned operators, we save the IRs of the best kernels.
"""

import argparse
import os
from pathlib import Path

from typing import Dict, List

try:
    from libfb.py import parutil
except ImportError:
    parutil = None
from tritonbench.operators import list_custom_triton_operators
from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.run_utils import run_in_task, run_one_operator

METADATA_DIR = (
    parutil.get_file_path("tritonbench/metadata")
    if parutil is not None and is_fbcode()
    else Path(__file__).parent.parent.parent.joinpath("tritonbench/metadata")
)

OSS_CUSTOM_TRITON_YAML = os.path.join(METADATA_DIR, "oss_triton_operators.yaml")
INTERNAL_CUSTOM_TRITON_YAML = os.path.join(
    METADATA_DIR, "fb/internal_triton_operators.yaml"
)
OSS_CUDA_KERNELS_YAML = os.path.join(METADATA_DIR, "oss_cuda_kernels.yaml")


def _filter_existing_files(paths: List[str]) -> List[str]:
    """Return only paths that exist on disk."""
    existing = []
    for p in paths:
        if Path(p).is_file():
            existing.append(p)
        else:
            print(f"[tritonbench][dump_ir] Skipping missing metadata: {p}")
    return existing


def _fallback_triton_ops_from_oss_kernels(yaml_path: str) -> Dict[str, List[str]]:
    """Fallback: derive Triton operator list from oss_cuda_kernels.yaml.

    We consider any op variant with a tag containing "triton" as a custom Triton
    benchmark and use the variant name as the sub-operator to run.
    """

    import yaml

    if not Path(yaml_path).is_file():
        return {}

    with open(yaml_path, "r") as fp:
        metadata = yaml.safe_load(fp) or {}

    triton_ops: Dict[str, List[str]] = {}
    for op_name, variants in metadata.items():
        if not isinstance(variants, dict):
            continue
        subops = [
            variant_name
            for variant_name, variant_meta in variants.items()
            if isinstance(variant_meta, dict)
            and "tags" in variant_meta
            and "triton" in variant_meta.get("tags", [])
        ]
        if subops:
            triton_ops[op_name] = subops
    return triton_ops


def get_parser():
    parser = argparse.ArgumentParser(description="Dump Triton IRs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory to save the IRs",
    )
    parser.add_argument(
        "--run-in-task", action="store_true", help="indicate running in task."
    )
    return parser


def run_operator(op: str, subop: List[str], output_dir: str):
    """Run a Tritonbench operator and save its IR to the specified directory"""
    opbench_args = [
        "--op",
        op,
        "--only",
        ",".join(subop),
        "--dump-ir",
        output_dir,
    ]
    run_in_task(op, opbench_args)


if __name__ == "__main__":
    parser = get_parser()
    args, extra_args = parser.parse_known_args()
    if args.run_in_task:
        run_one_operator(extra_args, with_bwd=True)
        exit(0)
    custom_triton_op_yamls = _filter_existing_files(
        [OSS_CUSTOM_TRITON_YAML, INTERNAL_CUSTOM_TRITON_YAML]
        if is_fbcode()
        else [OSS_CUSTOM_TRITON_YAML]
    )
    operators: Dict[str, List[str]] = {}
    if custom_triton_op_yamls:
        operators = list_custom_triton_operators(custom_triton_op_yamls)
    if not operators:
        operators = _fallback_triton_ops_from_oss_kernels(OSS_CUDA_KERNELS_YAML)
    if not operators:
        print("[tritonbench][dump_ir] No custom Triton metadata found; nothing to run.")
        exit(0)

    normalized_ops: Dict[str, List[str]] = {}
    for op_name, subops in operators.items():
        if isinstance(subops, dict):
            normalized_ops[op_name] = list(subops.keys())
        else:
            normalized_ops[op_name] = list(subops)

    [run_operator(op, normalized_ops[op], args.output_dir) for op in normalized_ops]
    print(f"[tritonbench][dump_ir] Result saved to {args.output_dir}")

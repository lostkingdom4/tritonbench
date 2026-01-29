from pathlib import Path

import tritonbench

pkg_dir = Path(tritonbench.__file__).resolve().parent
print(f"tritonbench package dir: {pkg_dir}")
print("Contents:")
for entry in sorted(pkg_dir.iterdir()):
    kind = "dir" if entry.is_dir() else "file"
    print(f"- [{kind}] {entry.name}")

print("\nTop-level callables exposed by tritonbench (non-dunder):")
for name in sorted(dir(tritonbench)):
    if name.startswith("__"):
        continue
    obj = getattr(tritonbench, name)
    if callable(obj):
        print(f"- {name}() -> {type(obj)}")
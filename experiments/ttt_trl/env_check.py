"""
Environment smoke check for TRL SDPO experiments.

Acceptance signals:
- Imports succeed
- torch.cuda.is_available() is True
- trl.__version__ is 1.4.0
- GPU 0 free VRAM >= 4.5 GB
"""

from __future__ import annotations

import sys
from importlib import import_module


REQUIRED_MODULES = [
    "torch",
    "trl",
    "peft",
    "transformers",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "wandb",
]


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _check_imports() -> int:
    _print_header("Import Check")
    failed = 0
    for module_name in REQUIRED_MODULES:
        try:
            module = import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {module_name}: {version}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            failed += 1
            print(f"[FAIL] {module_name}: {exc}")
    return failed


def _check_torch_cuda() -> int:
    import torch
    import trl

    _print_header("Runtime Check")
    failed = 0

    cuda_ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available() == {cuda_ok}")
    if not cuda_ok:
        failed += 1

    print(f"trl.__version__ == {trl.__version__!r}")
    if trl.__version__ != "1.4.0":
        failed += 1

    if cuda_ok:
        device = torch.device("cuda:0")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        print(f"GPU 0 free VRAM: {free_gb:.2f} GB / {total_gb:.2f} GB")
        print(f"VRAM free >= 4.5 GB: {free_gb >= 4.5}")
        if free_gb < 4.5:
            failed += 1
    else:
        print("GPU VRAM check skipped because CUDA is unavailable.")
        failed += 1

    return failed


def main() -> int:
    failures = 0
    failures += _check_imports()

    # Run runtime checks only if torch and trl are importable.
    try:
        failures += _check_torch_cuda()
    except Exception as exc:  # pragma: no cover - diagnostic script
        failures += 1
        print(f"[FAIL] Runtime check error: {exc}")

    _print_header("Result")
    if failures == 0:
        print("ENV CHECK PASSED")
        return 0

    print(f"ENV CHECK FAILED (issues={failures})")
    return 1


if __name__ == "__main__":
    sys.exit(main())

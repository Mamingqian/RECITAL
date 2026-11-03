"""Declared policy-training precision, and a refusal when the device cannot honour it.

``training_precision`` is a run declaration with two values: ``task_default``
keeps the task's own trainer configuration, ``fp32`` disables bf16 autocast. There
is no automatic value, so the precision a run trained in is readable from its
config rather than inferred from whichever device the scheduler assigned.

Native bf16 needs compute capability 8.0. On a device below that (for example
capability (7, 5)) autocast still runs, because PyTorch emulates bf16:
``is_bf16_supported()`` is True, ``is_bf16_supported(including_emulation=False)`` is
False, and a bf16 autocast matmul returns bfloat16 without error. Emulation is a different
computation at a different speed, so a run that declares ``task_default`` and lands
on such a device is refused here rather than trained in a precision no declaration
describes.
"""
from __future__ import annotations

DECLARED_VALUES = ("task_default", "fp32")


def training_precision(settings):
    """Validate and return the declared precision."""
    value = settings.get("training_precision", "task_default")
    if value not in DECLARED_VALUES:
        raise ValueError("training_precision must be task_default or fp32; no automatic precision fallback")
    return value


def apply_training_precision(runtime, settings):
    """Bind the declaration to the runtime the trainer reads."""
    if training_precision(settings) == "fp32":
        runtime["bf16"] = False


NATIVE_BF16_CAPABILITY = (8, 0)


def device_supports_bf16(device):
    """True when this CUDA device runs bf16 natively rather than emulated.

    ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True`` and so
    answers True on a device below capability 8.0; the capability is the predicate
    that separates native bf16 from emulation.
    """
    import torch

    if str(device) == "cpu" or not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability(torch.cuda.current_device())) >= NATIVE_BF16_CAPABILITY


def resolve_bf16(runtime, *, device):
    """The trainer's bf16 flag, refusing a declaration the device cannot honour."""
    import torch

    requested = bool(runtime.get("bf16", str(device) != "cpu"))
    if not requested or str(device) == "cpu":
        return False
    if not device_supports_bf16(device):
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        raise ValueError(
            "bf16 is emulated, not native, on compute capability "
            f"{capability[0]}.{capability[1]}; declare training_precision: fp32 for this hardware"
        )
    return True

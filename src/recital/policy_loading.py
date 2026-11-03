"""Loading of the HyenaDNA-based policies of the CRE and Loop-seq tasks."""
from __future__ import annotations


def load_remote_code_policy(checkpoint):
    """Return ``(model, tokenizer)`` from a local policy checkpoint directory.

    The HyenaDNA architecture and tokenizer are the upstream remote code that the checkpoint directory carries
    (``auto_map`` in its config), so none of it is part of this repository; transformers runs it with
    ``trust_remote_code``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    return model, tokenizer

"""ProGen3 single-sequence adapter using Transformers' native Mixtral blocks.

ProGen3's released causal architecture uses the same rotary attention, RMSNorm,
SwiGLU experts and normalized top-k routing, plus a sequence-ID embedding. This
adapter maps the public packed expert weights without changing their values.
It supports ordinary attention masks, generation and checkpoint reload through
the shared trainer, without requiring the MegaBlocks CUDA extension.

Architecture reference: https://github.com/Profluent-AI/progen3 (Apache-2.0).
The upstream checkpoint's unused masked-LM head is omitted; the causal head and
all transformer/sequence-embedding weights are retained.
"""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request

from torch import nn
from transformers import MixtralConfig, MixtralForCausalLM, PreTrainedTokenizerFast

from recital.token_sequence import RepresentationSpec
from tasks.protein.codec import PROTEIN_ALPHABET


class ProGen3CausalLM(MixtralForCausalLM):
    """Public ProGen3 weights with a standard single-sequence causal interface."""

    def __init__(self, config):
        super().__init__(config)
        self.sequence_embedding = nn.Embedding(getattr(config, "max_num_sequences", 512), config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_router_logits=None, cache_position=None,
                logits_to_keep=0, token_type_ids=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        # Independent generated proteins all have the upstream sequence ID 0.
        inputs_embeds = inputs_embeds + self.sequence_embedding.weight[0]
        return super().forward(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, labels=labels,
            use_cache=use_cache, output_router_logits=output_router_logits,
            cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs,
        )


class ProGenSequenceCodec:
    def __init__(self, tokenizer, min_length: int = 80, max_length: int = 96):
        self.tokenizer = tokenizer
        self.representation = RepresentationSpec(
            modality="protein", alphabet=PROTEIN_ALPHABET, min_length=min_length,
            max_length=max_length, variable_length=True, eos_token="2", pad_token="<pad>",
        )

    def validate(self, sequence: str) -> None:
        if not set(sequence) <= set(PROTEIN_ALPHABET):
            raise ValueError("Generated protein contains nonstandard residues")
        if not self.representation.min_length <= len(sequence) <= self.representation.max_length:
            raise ValueError("Generated protein length is outside the CM range")

    def legal_token_ids(self, include_eos: bool = False) -> list[int]:
        ids = [self.tokenizer.convert_tokens_to_ids(a) for a in PROTEIN_ALPHABET]
        return ids + ([self.tokenizer.eos_token_id] if include_eos else [])


def convert_checkpoint(source: Path, destination: Path, tokenizer_source: Path | None = None) -> Path:
    """Map each packed expert matrix to the equivalent native expert tensor."""
    from safetensors.torch import load_file
    source, destination = Path(source), Path(destination)
    if (destination / "model.safetensors").exists():
        return destination
    original = json.loads((source / "config.json").read_text())
    if original.get("fused_attention_norm") or not original.get("gated_mlp"):
        raise ValueError("This adapter supports the released unfused SwiGLU causal architecture")
    if original.get("clip_qkv") is not None:
        raise ValueError("Clipped QKV requires an explicit adapter implementation")
    config = MixtralConfig(
        vocab_size=original["vocab_size"], hidden_size=original["hidden_size"],
        intermediate_size=original["intermediate_size"], num_hidden_layers=original["num_hidden_layers"],
        num_attention_heads=original["num_attention_heads"],
        num_key_value_heads=original["num_key_value_heads"],
        num_local_experts=original["num_experts"], num_experts_per_tok=original["num_experts_per_tok"],
        max_position_embeddings=original["max_position_embeddings"],
        rms_norm_eps=original["rms_norm_eps"], rope_theta=original["rope_theta"],
        hidden_act=original["hidden_act"], attention_dropout=original["attention_dropout"],
        router_aux_loss_coef=original["router_aux_loss_coef"],
        pad_token_id=0, bos_token_id=1, eos_token_id=7, sliding_window=None,
        tie_word_embeddings=False, max_num_sequences=original["max_num_sequences"],
        source_model="Profluent-Bio/progen3-112m", adapter="progen3_single_sequence",
    )
    model = ProGen3CausalLM(config)
    weights = load_file(str(source / "model.safetensors"))
    mapped = {}
    for key, tensor in weights.items():
        if key == "mlm_head.weight":
            continue
        if key == "model.embed_seq_id.weight":
            mapped["sequence_embedding.weight"] = tensor
        elif ".experts.mlp." in key:
            prefix, matrix = key.split(".experts.mlp.")
            for expert, part in enumerate(tensor.chunk(original["num_experts"], dim=0)):
                name = {"w1": "w1", "v1": "w3", "w2": "w2"}[matrix]
                mapped[f"{prefix}.experts.{expert}.{name}.weight"] = part.T if matrix == "w2" else part
        else:
            mapped[key.replace(".router.layer.weight", ".gate.weight")] = tensor
    model.load_state_dict(mapped, strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    if tokenizer_source is None:
        tokenizer_source = source / "tokenizer.json"
        if not tokenizer_source.exists():
            url = "https://raw.githubusercontent.com/Profluent-AI/progen3/main/src/progen3/tokenizer.json"
            tokenizer_source.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_source), bos_token="<bos>", eos_token="2", pad_token="<pad>",
        model_max_length=128, clean_up_tokenization_spaces=False,
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.padding_side = "left"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(destination)
    (destination / "conversion.json").write_text(json.dumps({
        "source": config.source_model, "backend": "Transformers Mixtral blocks plus sequence embedding",
        "omitted": ["mlm_head.weight"], "mapped_tensor_count": len(mapped),
        "parameters": sum(p.numel() for p in model.parameters()),
        "generation_prompt": "<bos>1", "completion_terminator": "2",
    }, indent=2))
    return destination

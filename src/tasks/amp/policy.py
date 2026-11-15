"""AMP-Prompt as a causal LM the GRPO trainer can train.

The Zenodo record 17018363 ships AMP-Designer's soft-prompt GPT-2: ten learned vectors replace the embeddings of the first
ten input positions. The trainer feeds token ids to a `GPT2LMHeadModel`, so the ten vectors are stored as ten extra rows of
the embedding matrix (ids 25 to 34) and the prompt is the ten reserved tokens followed by [CLS]. The output embedding is tied
to those rows, which would also make the ten prompt rows output logits (norm 28 against 4 for a token row, so they took 0.8 of
the probability); `AmpPromptLM` adds -1e9 to the logits of ids 25 to 34, so the softmax over 35 ids is the soft-prompt model's
softmax over its 25 tokens, and the log-probabilities the trainer scores are those of the model that is sampled.

Importing this module loads neither torch nor transformers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROMPT_TOKENS = 10
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
RESERVED_TOKENS = tuple(f"[P{i}]" for i in range(PROMPT_TOKENS))
# Ids: [PAD] 0, [UNK] 1, [CLS] 2, [SEP] 3, [MASK] 4, residues 5 to 24 in the order of RESIDUES, prompt tokens 25 to 34.
VOCABULARY = SPECIAL_TOKENS + tuple(RESIDUES) + RESERVED_TOKENS
ORIGINAL_VOCABULARY_SIZE = len(SPECIAL_TOKENS) + len(RESIDUES)
# Added to the output logit of every prompt token.
MASKED_LOGIT = -1e9
PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3
PROMPT_TEXT = " ".join(RESERVED_TOKENS) + " [CLS]"
MIN_LENGTH, MAX_LENGTH = 5, 32
# Up to 32 residues and the [SEP] that ends them.
MAX_COMPLETION_TOKENS = MAX_LENGTH + 1


def tokenizer():
    """Word-level tokenizer over `VOCABULARY` that adds no special tokens and returns no `token_type_ids`.

    A BERT-style tokenizer wraps every prompt in [CLS] ... [SEP], and GPT-2 adds `token_type_ids` to the input
    embeddings as the embedding of token 0; GRPO feeds the policy both. Decoding joins tokens without separators.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    core = Tokenizer(models.WordLevel({token: index for index, token in enumerate(VOCABULARY)}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    core.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", bos_token="[CLS]", eos_token="[SEP]", additional_special_tokens=list(RESERVED_TOKENS),
        model_input_names=["input_ids", "attention_mask"], clean_up_tokenization_spaces=False)


def reserved_prompt_ids() -> list[int]:
    return [VOCABULARY.index(token) for token in (*RESERVED_TOKENS, "[CLS]")]


def residue_ids(sequence: str) -> list[int]:
    """[CLS], the residues, [SEP]: the input of the MIC regressors."""
    return [CLS_ID, *(len(SPECIAL_TOKENS) + RESIDUES.index(residue) for residue in sequence), SEP_ID]


_CLASSES: dict = {}


def model_classes():
    """`(AmpPromptConfig, AmpPromptLM)`, defined and registered with the Auto classes on first use.

    The config has its own `model_type`, so an Auto class in a process that has not called this refuses a checkpoint of the policy. A
    stock `GPT2LMHeadModel` does not refuse: it warns and loads the weights without the mask (probability 1.0 on the prompt tokens), so
    every path loads the policy through `load_policy`.
    """
    if not _CLASSES:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

        class AmpPromptConfig(GPT2Config):
            model_type = "amp_prompt"

            def __init__(self, unmasked_vocab_size=ORIGINAL_VOCABULARY_SIZE, **kwargs):
                super().__init__(**kwargs)
                self.unmasked_vocab_size = unmasked_vocab_size

        class MaskedHead(torch.nn.Linear):
            """The tied output projection with `MASKED_LOGIT` added to every id from `unmasked` on."""

            def __init__(self, in_features, out_features, unmasked):
                super().__init__(in_features, out_features, bias=False)
                self.unmasked = unmasked

            def forward(self, hidden):
                logits = super().forward(hidden)
                mask = torch.zeros(self.out_features, dtype=logits.dtype, device=logits.device)
                mask[self.unmasked:] = MASKED_LOGIT
                return logits + mask

        class AmpPromptLM(GPT2LMHeadModel):
            config_class = AmpPromptConfig

            def __init__(self, config):
                super().__init__(config)
                self.lm_head = MaskedHead(config.n_embd, config.vocab_size, config.unmasked_vocab_size)
                self.post_init()

        AutoConfig.register("amp_prompt", AmpPromptConfig, exist_ok=True)
        AutoModelForCausalLM.register(AmpPromptConfig, AmpPromptLM, exist_ok=True)
        _CLASSES.update(config=AmpPromptConfig, model=AmpPromptLM)
    return _CLASSES["config"], _CLASSES["model"]


def _read_state(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def soft_prompt_model(prompt_dir: Path):
    """The shipped model: GPT-2 whose first ten input embeddings are the learned prompt vectors."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    class SoftPromptEmbedding(torch.nn.Module):
        def __init__(self, wte):
            super().__init__()
            self.wte = wte
            self.learned_embedding = torch.nn.Parameter(torch.zeros(PROMPT_TOKENS, wte.embedding_dim))

        def forward(self, tokens):
            body = self.wte(tokens[:, PROMPT_TOKENS:])
            return torch.cat([self.learned_embedding.expand(body.shape[0], -1, -1), body], dim=1)

    model = GPT2LMHeadModel(GPT2Config.from_json_file(str(Path(prompt_dir) / "config.json")))
    model.set_input_embeddings(SoftPromptEmbedding(model.get_input_embeddings()))
    state = {key: value for key, value in _read_state(Path(prompt_dir) / "pytorch_model.bin").items()
             if not key.endswith((".attn.bias", ".attn.masked_bias"))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(key != "lm_head.weight" for key in missing):
        raise ValueError(f"prompt checkpoint does not fit the architecture: missing {missing}, unexpected {unexpected}")
    return model.eval()


def convert_prompt_checkpoint(prompt_dir: Path, out_dir: Path) -> Path:
    """Write the reserved-token policy and its tokenizer to `out_dir` (a new directory), with dropout off.

    `prompt_dir` holds the record's `config.json` and `pytorch_model.bin`. The saved config has 35 embedding rows, the
    prompt rows masked in the output, and all dropout probabilities zero, so the policy and the KL reference it is compared
    with run the same function.
    """
    import torch
    from transformers import GPT2Config

    config_class, model_class = model_classes()
    prompt_dir, out_dir = Path(prompt_dir), Path(out_dir)
    config = config_class(**GPT2Config.from_json_file(str(prompt_dir / "config.json")).to_dict() | {"model_type": "amp_prompt"})
    config.vocab_size = len(VOCABULARY)
    config.attn_pdrop = config.resid_pdrop = config.embd_pdrop = 0.0
    state = _read_state(prompt_dir / "pytorch_model.bin")
    learned = state["transformer.wte.learned_embedding"]
    wte = state["transformer.wte.wte.weight"]
    if tuple(learned.shape) != (PROMPT_TOKENS, config.n_embd) or wte.shape[0] != ORIGINAL_VOCABULARY_SIZE:
        raise ValueError("prompt checkpoint embedding shapes are not those of AMP-Prompt")
    kept = {key: value for key, value in state.items()
            if "wte" not in key and key != "lm_head.weight" and not key.endswith((".attn.bias", ".attn.masked_bias"))}
    kept["transformer.wte.weight"] = torch.cat([wte, learned], dim=0)
    model = model_class(config)
    missing, unexpected = model.load_state_dict(kept, strict=False)
    if unexpected or any(key != "lm_head.weight" for key in missing):
        raise ValueError(f"prompt checkpoint does not fit the architecture: missing {missing}, unexpected {unexpected}")
    model.tie_weights()
    model.generation_config.pad_token_id, model.generation_config.eos_token_id = PAD_ID, SEP_ID
    model.generation_config.bos_token_id = CLS_ID
    out_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(out_dir)
    tokenizer().save_pretrained(out_dir)
    source = prompt_dir / "pytorch_model.bin"
    (out_dir / "conversion.json").write_text(json.dumps({
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": {"original": ORIGINAL_VOCABULARY_SIZE, "prompt": PROMPT_TOKENS},
        "dropout": 0.0, "prompt_text": PROMPT_TEXT, "masked_output_logit": MASKED_LOGIT}, indent=2) + "\n")
    return out_dir


def load_policy(checkpoint, device: str = "cpu"):
    """The model and tokenizer saved by `convert_prompt_checkpoint` or by the trainer."""
    from transformers import AutoTokenizer

    model = model_classes()[1].from_pretrained(str(checkpoint), local_files_only=True)
    model.generation_config.top_k = 0
    model.generation_config.do_sample = True
    return model.to(device), AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)


def _inputs(bodies):
    import torch

    width = max(len(body) for body in bodies)
    padded = torch.tensor([[*body, *[PAD_ID] * (width - len(body))] for body in bodies])
    count = len(bodies)
    soft_in = torch.cat([torch.full((count, PROMPT_TOKENS), UNK_ID), torch.full((count, 1), CLS_ID), padded], dim=1)
    reserved_in = torch.cat([torch.tensor([reserved_prompt_ids()] * count), padded], dim=1)
    return soft_in, reserved_in


def max_logit_difference(soft, reserved, bodies) -> float:
    """Largest absolute difference, over every position of `bodies`, of the log-probabilities of the 25 original tokens.

    `bodies` is a list of token-id lists (residues and [SEP]); the soft-prompt model receives ten placeholder ids that
    its first layer replaces, the reserved-token model receives the ten prompt ids. The reserved-token model's softmax is
    over all of its 35 ids, as the trainer scores it, so a probability left on a prompt token shows as a difference.
    """
    import torch

    soft_in, reserved_in = _inputs(bodies)
    with torch.no_grad():
        first = torch.log_softmax(soft(soft_in).logits[..., :ORIGINAL_VOCABULARY_SIZE].double(), dim=-1)
        second = torch.log_softmax(reserved(reserved_in).logits.double(), dim=-1)[..., :ORIGINAL_VOCABULARY_SIZE]
    return float((first - second).abs().max())


def unsampled_mass(model, bodies) -> dict[str, float]:
    """The largest probability, over every position of `bodies`, on the ids the sampler never draws.

    `prompt` is the mass on the ten prompt ids; `unsampled` is the mass on every id that is neither a residue nor [SEP].
    """
    import torch

    _, reserved_in = _inputs(bodies)
    sampled = [SEP_ID, *range(len(SPECIAL_TOKENS), ORIGINAL_VOCABULARY_SIZE)]
    with torch.no_grad():
        probabilities = torch.softmax(model(reserved_in).logits[:, PROMPT_TOKENS:].double(), dim=-1)
    mask = torch.ones(probabilities.shape[-1], dtype=torch.bool)
    mask[sampled] = False
    return {"prompt": float(probabilities[..., ORIGINAL_VOCABULARY_SIZE:].sum(-1).max()),
            "unsampled": float(probabilities[..., mask].sum(-1).max())}

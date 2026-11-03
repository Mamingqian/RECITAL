"""Small, modality-neutral description of the token sequences a policy generates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RepresentationSpec:
    """Description of the representation consumed by a causal-LM policy.

    ``alphabet`` is the legal character/residue alphabet used by validation.
    ``token_vocabulary`` keeps tokenizer boundaries when tokens are not single
    characters (for example, a BPE protein tokenizer).  Residue/character
    adapters can leave it empty; it is populated with one-character tokens at
    construction time.  Keeping these two notions separate prevents a
    multi-character vocabulary from being silently split into characters.
    """

    modality: str
    alphabet: str | Sequence[str]
    tokenizer: str = "residue"
    min_length: int = 1
    max_length: int | None = None
    variable_length: bool = False
    pad_token: str | None = "<pad>"
    eos_token: str | None = "<eos>"
    unk_token: str | None = "<unk>"
    prefix: str = ""
    supports_edit_mask: bool = False
    token_vocabulary: tuple[str, ...] | Sequence[str] = ()

    def __post_init__(self) -> None:
        modality = str(self.modality).strip().lower()
        tokenizer = str(self.tokenizer).strip().lower()
        raw_alphabet = self.alphabet
        if isinstance(raw_alphabet, str):
            alphabet = raw_alphabet
            inferred_vocabulary = tuple(alphabet)
        else:
            inferred_vocabulary = tuple(str(token) for token in raw_alphabet)
            alphabet = "".join(inferred_vocabulary)
        vocabulary = tuple(str(token) for token in self.token_vocabulary)
        if not vocabulary:
            vocabulary = inferred_vocabulary
        if not modality:
            raise ValueError("representation modality must be non-empty")
        if not tokenizer:
            raise ValueError("representation tokenizer must be non-empty")
        if not alphabet:
            raise ValueError("representation alphabet must be non-empty")
        if not all(vocabulary):
            raise ValueError("representation token_vocabulary cannot contain empty tokens")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("representation token_vocabulary must not contain duplicates")
        if tokenizer == "residue" and any(len(token) != 1 for token in vocabulary):
            raise ValueError(
                "residue tokenizers require one-character token_vocabulary entries"
            )
        if tokenizer != "bpe" and len(set(alphabet)) != len(alphabet):
            raise ValueError("representation alphabet must not contain duplicates")
        if self.min_length < 0:
            raise ValueError("representation min_length cannot be negative")
        if self.max_length is not None and self.max_length < self.min_length:
            raise ValueError("representation max_length must be >= min_length")
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "prefix", str(self.prefix))
        object.__setattr__(self, "token_vocabulary", vocabulary)


    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Return tokenizer tokens without losing multi-character boundaries."""
        return tuple(self.token_vocabulary)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation descriptor."""
        return {
            "modality": self.modality,
            "alphabet": self.alphabet,
            "tokenizer": self.tokenizer,
            "token_vocabulary": list(self.token_vocabulary),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "variable_length": self.variable_length,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "prefix": self.prefix,
            "supports_edit_mask": self.supports_edit_mask,
        }

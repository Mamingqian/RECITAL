"""CM stage: SFT training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tasks.protein.progen import ProGen3CausalLM, convert_checkpoint


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.padding_side = "left"
    return tokenizer


def train_sft(root: Path, *, device: str, seed: int, resume: bool = False) -> dict:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    root = Path(root)
    output = root / "sft"
    if resume and (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text())
    base = convert_checkpoint(root / "assets/progen3-112m", root / "assets/progen3-hf")
    set_seed(seed)
    tokenizer = load_tokenizer(base)
    model = ProGen3CausalLM.from_pretrained(base, local_files_only=True)
    data = pd.read_csv(root / "data/sft.csv")

    def dataset(split):
        rows = []
        for sequence in data.loc[data.split.eq(split), "sequence"]:
            ids = tokenizer.encode("<bos>1" + sequence + "2", add_special_tokens=False)
            rows.append({"input_ids": ids, "labels": [-100, -100] + ids[2:]})
        return Dataset.from_list(rows)

    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            n = len(row["input_ids"])
            ids[i, :n] = torch.tensor(row["input_ids"])
            labels[i, :n] = torch.tensor(row["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.05,
        lr_scheduler_type="cosine", max_grad_norm=1.,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=20, report_to="none", seed=seed, data_seed=seed,
        use_cpu=device == "cpu", dataloader_num_workers=0,
        save_safetensors=True, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset("train"),
                      eval_dataset=dataset("valid"), data_collator=collate,
                      processing_class=tokenizer)
    initial = trainer.evaluate()
    checkpoint = get_last_checkpoint(str(output / "checkpoints")) if resume and (output / "checkpoints").exists() else None
    trained = trainer.train(resume_from_checkpoint=checkpoint)
    final = trainer.evaluate()
    model.config.use_cache = True
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(output / "model")
    result = {"base_model": "Profluent-Bio/progen3-112m", "seed": seed,
              "initial_eval_loss": initial["eval_loss"], "final_eval_loss": final["eval_loss"],
              "global_step": trainer.state.global_step, "training": trained.metrics,
              "checkpoint": str(output / "model"), "best_checkpoint": trainer.state.best_model_checkpoint,
              "sft_counts": data.split.value_counts().to_dict()}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def run_stage(args, *, device: str):
    if args.stage == "sft":
        return train_sft(args.root, device=device, seed=args.seed, resume=args.resume)
    raise ValueError(f"Unknown CM stage {args.stage!r}")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "policy"}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, default=str))
    temporary.replace(path)

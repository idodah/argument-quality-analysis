"""Fine-tune Qwen3-8B with QLoRA (4-bit NF4) for pair-wise argument ranking.

Each example produces two forward passes (delta arg / nodelta arg); each yields
a scalar score from a small head on the mean-pooled last hidden state. Training
minimizes a margin ranking loss pushing score(delta) above score(nodelta). At
inference we score both candidates and pick the higher — order-invariant by
construction (no A/B token, no two-order averaging).

Usage:
    python -m models.qwen
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from models.data import (
    RANDOM_SEED,
    evaluate,
    load_data,
    make_rank_prompt,
    save_results,
    shuffle_pairs,
    split_by_date,
)

MODEL_ID = "Qwen/Qwen3-8B"
MAX_LENGTH = 8192
RANKING_MARGIN = 0.1

LORA_CONFIG = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def _format_prompt(prompt: dict) -> str:
    return f"{prompt['system']}\n\n{prompt['user']}"


def _trim_prompt_to_length(prompt: dict, tokenizer) -> str:
    """If the formatted prompt exceeds MAX_LENGTH, trim the ### Post section to fit."""
    text = _format_prompt(prompt)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= MAX_LENGTH:
        return text

    post_marker = "### Post\n"
    split_marker = "\n\n### Response"
    full_user = prompt["user"]
    post_start = full_user.index(post_marker) + len(post_marker)
    post_end = full_user.index(split_marker)
    original_post = full_user[post_start:post_end]

    without_post = prompt["system"] + "\n\n" + full_user[:post_start] + "{POST}" + full_user[post_end:]
    overhead = len(tokenizer.encode(without_post, add_special_tokens=False))
    budget = MAX_LENGTH - overhead

    if budget <= 0:
        trimmed_post = ""
    else:
        post_ids = tokenizer.encode(original_post, add_special_tokens=False)[:budget]
        trimmed_post = tokenizer.decode(post_ids, skip_special_tokens=True)

    trimmed_user = full_user[:post_start] + trimmed_post + full_user[post_end:]
    return f"{prompt['system']}\n\n{trimmed_user}"


def _tokenize_single(text: str, tokenizer) -> dict:
    return tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)


def _build_dataset(fields: dict, labels: np.ndarray, tokenizer) -> Dataset:
    """Each row produces tokenized (delta_prompt, nodelta_prompt) pair."""
    records = []
    for i in range(len(labels)):
        topic = fields["topic"][i]
        post = fields["original_post"][i]
        # labels[i] == 1 means arg_a is the delta argument.
        if labels[i] == 1:
            delta_arg, nodelta_arg = fields["arg_a"][i], fields["arg_b"][i]
        else:
            delta_arg, nodelta_arg = fields["arg_b"][i], fields["arg_a"][i]

        delta_text = _trim_prompt_to_length(make_rank_prompt(topic, post, delta_arg), tokenizer)
        nodelta_text = _trim_prompt_to_length(make_rank_prompt(topic, post, nodelta_arg), tokenizer)
        delta_tok = _tokenize_single(delta_text, tokenizer)
        nodelta_tok = _tokenize_single(nodelta_text, tokenizer)
        records.append({
            "delta_input_ids": delta_tok["input_ids"],
            "delta_attention_mask": delta_tok["attention_mask"],
            "nodelta_input_ids": nodelta_tok["input_ids"],
            "nodelta_attention_mask": nodelta_tok["attention_mask"],
        })
    return Dataset.from_list(records)


class PairCollator:
    """Pads delta and nodelta sequences independently and stacks them."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_side(self, batch: list[dict], key_ids: str, key_mask: str) -> dict:
        features = [{"input_ids": b[key_ids], "attention_mask": b[key_mask]} for b in batch]
        padded = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return padded

    def __call__(self, batch: list[dict]) -> dict:
        delta = self._pad_side(batch, "delta_input_ids", "delta_attention_mask")
        nodelta = self._pad_side(batch, "nodelta_input_ids", "nodelta_attention_mask")
        return {
            "delta_input_ids": delta["input_ids"],
            "delta_attention_mask": delta["attention_mask"],
            "nodelta_input_ids": nodelta["input_ids"],
            "nodelta_attention_mask": nodelta["attention_mask"],
        }


class RankingModel(nn.Module):
    """PEFT base model + small scalar score head over the mean-pooled non-pad tokens."""

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.score_head = nn.Linear(hidden_size, 1, bias=False)
        # Keep the score head in fp32 for stable training; cast inputs at use time.
        self.score_head.to(torch.float32)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self, **kwargs):
        self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base.gradient_checkpointing_disable()

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]  # (B, T, H)
        # Mean-pool over non-pad tokens.
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / counts  # (B, H)
        scores = self.score_head(pooled.to(torch.float32)).squeeze(-1)  # (B,)
        return scores

    def forward(self, return_loss: bool = True, **batch):
        # `return_loss=True` default lets HF Trainer detect that this model
        # computes its own loss without a `labels` input, so eval_loss is
        # captured during evaluation.
        delta_scores = self.score(batch["delta_input_ids"], batch["delta_attention_mask"])
        nodelta_scores = self.score(batch["nodelta_input_ids"], batch["nodelta_attention_mask"])
        target = torch.ones_like(delta_scores)
        loss = F.margin_ranking_loss(delta_scores, nodelta_scores, target, margin=RANKING_MARGIN)
        return {"loss": loss, "delta_scores": delta_scores, "nodelta_scores": nodelta_scores}


class RankingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        # Default Trainer._save would call state_dict() on the full RankingModel,
        # which under QLoRA includes ~8B 4-bit-quantized base params that
        # safetensors can't round-trip. Save the same layout as save_model()
        # (adapter/, score_head.pt, tokenizer/) so load_model() can read these
        # intermediate checkpoints directly.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        model.base.save_pretrained(os.path.join(output_dir, "adapter"))
        torch.save(model.score_head.state_dict(), os.path.join(output_dir, "score_head.pt"))
        tokenizer = self.processing_class
        if tokenizer is None and self.data_collator is not None and hasattr(self.data_collator, "tokenizer"):
            tokenizer = self.data_collator.tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class BestCheckpointTracker(TrainerCallback):
    """Records the path of the checkpoint with the best (lowest) eval_loss."""

    def __init__(self):
        self.best_metric: float | None = None
        self.best_checkpoint: str | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return
        loss = metrics["eval_loss"]
        if self.best_metric is None or loss < self.best_metric:
            self.best_metric = loss
            self.best_checkpoint = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )


def _load_model():
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        token=hf_token,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base = get_peft_model(base, LORA_CONFIG)
    base.print_trainable_parameters()

    hidden_size = base.config.hidden_size
    model = RankingModel(base, hidden_size)
    # Move score head to the same device as the embedding output.
    device = next(base.parameters()).device
    model.score_head.to(device)
    return model, tokenizer


def save_model(model: "RankingModel", tokenizer, path: str | Path) -> None:
    """Save LoRA adapter + score head + tokenizer to a directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(path / "adapter")
    torch.save(model.score_head.state_dict(), path / "score_head.pt")
    tokenizer.save_pretrained(path / "tokenizer")
    print(f"Saved fine-tuned model to {path}")


def load_model(path: str | Path) -> tuple["RankingModel", "AutoTokenizer"]:
    """Re-load a fine-tuned model saved by `save_model`.

    `path` may be a local directory or a Hugging Face Hub repo id (e.g. 'user/repo').
    """
    hf_token = os.environ.get("HF_TOKEN")
    local_dir = Path(str(path))
    is_local = local_dir.exists()

    if is_local:
        adapter_src = local_dir / "adapter"
        tokenizer_src = local_dir / "tokenizer"
        score_head_path = local_dir / "score_head.pt"
    else:
        from huggingface_hub import hf_hub_download, snapshot_download

        repo_root = snapshot_download(repo_id=str(path), token=hf_token)
        adapter_src = Path(repo_root) / "adapter"
        tokenizer_src = Path(repo_root) / "tokenizer"
        score_head_path = Path(hf_hub_download(repo_id=str(path), filename="score_head.pt", token=hf_token))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        token=hf_token,
    )
    base = PeftModel.from_pretrained(base, adapter_src)
    base.eval()

    hidden_size = base.config.hidden_size
    model = RankingModel(base, hidden_size)
    state = torch.load(score_head_path, map_location="cpu")
    model.score_head.load_state_dict(state)
    device = next(base.parameters()).device
    model.score_head.to(device)
    model.eval()
    return model, tokenizer


def _score_one(model: "RankingModel", tokenizer, topic: str, post: str, arg: str) -> float:
    """Scalar score for one (topic, post, arg) under the ranker."""
    device = next(model.base.parameters()).device
    text = _trim_prompt_to_length(make_rank_prompt(topic, post, arg), tokenizer)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        return model.score(enc["input_ids"], enc["attention_mask"]).item()


def score_pair(model: "RankingModel", tokenizer, topic: str, post: str, arg_a: str, arg_b: str) -> dict:
    """Score two candidate arguments. Returns scores, winner, and P(arg_a is more persuasive)."""
    model.eval()
    s_a = _score_one(model, tokenizer, topic, post, arg_a)
    s_b = _score_one(model, tokenizer, topic, post, arg_b)
    return {
        "score_a": s_a,
        "score_b": s_b,
        "winner": "A" if s_a > s_b else "B",
        "prob_a_better": float(torch.sigmoid(torch.tensor(s_a - s_b))),
    }


def _predict(model, tokenizer, fields: dict, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, probs = [], []
    for i in range(len(labels)):
        s_a = _score_one(model, tokenizer, fields["topic"][i], fields["original_post"][i], fields["arg_a"][i])
        s_b = _score_one(model, tokenizer, fields["topic"][i], fields["original_post"][i], fields["arg_b"][i])
        # P(arg_a is delta) via sigmoid of score difference.
        probs.append(float(torch.sigmoid(torch.tensor(s_a - s_b))))
        preds.append(1 if s_a > s_b else 0)
    return np.array(preds), np.array(probs)


def _setup_wandb(model_name: str) -> str:
    """Init a Weights & Biases run if WANDB_API_KEY is set; return report_to.

    Returns "wandb" (and inits the run) when configured, else "none" so the HF
    Trainer logs nowhere extra. Project name comes from WANDB_PROJECT or a
    sensible default. Safe when wandb isn't installed."""
    if not os.environ.get("WANDB_API_KEY"):
        return "none"
    try:
        import wandb
    except ImportError:
        print("[wandb] WANDB_API_KEY set but wandb not installed; skipping (run `uv add wandb`).")
        return "none"
    project = os.environ.get("WANDB_PROJECT", "qwen-qlora-ranker")
    wandb.init(project=project, name=model_name, config={"model": model_name})
    print(f"[wandb] logging enabled (project={project!r}, run={model_name!r})")
    return "wandb"


def run(
    train_fields,
    train_labels,
    val_fields,
    val_labels,
    test_fields,
    test_labels,
    **_,
) -> list[dict]:
    model_name = "qwen_qlora_rank"

    model, tokenizer = _load_model()

    train_ds = _build_dataset(train_fields, train_labels, tokenizer)
    val_ds = _build_dataset(val_fields, val_labels, tokenizer)

    # Opt-in Weights & Biases logging: active only when WANDB_API_KEY is set,
    # so training works unchanged for anyone without a W&B account.
    report_to = _setup_wandb(model_name)

    args = TrainingArguments(
        output_dir=f"./checkpoints/{model_name}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=0.25,
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=3,
        report_to=report_to,
        run_name=model_name,
        seed=RANDOM_SEED,
        remove_unused_columns=False,
        label_names=[],
    )

    best_tracker = BestCheckpointTracker()
    trainer = RankingTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PairCollator(tokenizer),
        callbacks=[best_tracker],
    )
    trainer.train()

    final_dir = Path(f"./checkpoints/{model_name}/final")
    save_model(model, tokenizer, final_dir)

    if best_tracker.best_checkpoint is not None and os.path.isdir(best_tracker.best_checkpoint):
        print(f"Loading best checkpoint for eval: {best_tracker.best_checkpoint} (eval_loss={best_tracker.best_metric:.4f})")
        del model
        del trainer
        torch.cuda.empty_cache()
        model, tokenizer = load_model(best_tracker.best_checkpoint)
    else:
        print("No best checkpoint recorded; evaluating last-step model.")

    results = []
    for split_name, fields, labels in [("val", val_fields, val_labels), ("test", test_fields, test_labels)]:
        y_pred, y_prob = _predict(model, tokenizer, fields, labels)
        results.append({**evaluate(model_name, labels, y_pred, y_prob), "split": split_name, "n_train": len(train_labels)})
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    df = load_data()
    train_df, val_df, test_df = split_by_date(df)
    train_fields, train_labels = shuffle_pairs(train_df, seed=RANDOM_SEED)
    val_fields, val_labels = shuffle_pairs(val_df, seed=RANDOM_SEED + 1)
    test_fields, test_labels = shuffle_pairs(test_df, seed=RANDOM_SEED + 2)

    results = run(
        train_fields, train_labels,
        val_fields, val_labels,
        test_fields, test_labels,
    )
    save_results(results)
    for r in results:
        print(r)

"""
Fine-tune Llama (base) with QLoRA (4-bit NF4) for pair-wise argument-quality
ranking.

Each example produces two forward passes — one with the delta argument and one
with the nodelta argument. Each pass produces a scalar score from a small head
on top of the last non-pad hidden state. Training minimizes a margin ranking
loss that pushes score(delta) above score(nodelta).

At inference we score both candidate arguments and predict the one with the
higher score. This is order-invariant by construction — no A/B token, no
two-order averaging.

Usage:
    python -m models.llama                  # uses the full original post (default)
    python -m models.llama --use-summary    # uses the gpt-oss summary if present
"""

import argparse
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

MODEL_ID = "meta-llama/Llama-3.1-8B"
MAX_LENGTH = 4096
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


def _get_post(fields: dict, i: int, use_summary: bool) -> str:
    if use_summary:
        s = fields["summary"][i]
        if s is not None and str(s).strip():
            return str(s)
    return fields["original_post"][i]


def _tokenize_single(text: str, tokenizer) -> dict:
    return tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)


def _build_dataset(fields: dict, labels: np.ndarray, tokenizer, use_summary: bool) -> Dataset:
    """Each row produces tokenized (delta_prompt, nodelta_prompt) pair."""
    records = []
    for i in range(len(labels)):
        topic = fields["topic"][i]
        post = _get_post(fields, i, use_summary)
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
    """PEFT base model + small scalar score head over the last non-pad token."""

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
        # Index the last non-pad token per row.
        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        idx = seq_lens.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
        last_tok = last_hidden.gather(1, idx).squeeze(1)  # (B, H)
        scores = self.score_head(last_tok.to(torch.float32)).squeeze(-1)  # (B,)
        return scores

    def forward(self, **batch):
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


def _load_model():
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the "last non-pad token" indexing is unambiguous for short rows.
    # Actually we use right-padding + attention_mask.sum()-1, which handles either side.

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


def push_to_hub(local_path: str | Path, repo_id: str, private: bool = True) -> None:
    """Upload a saved model directory (adapter + score head + tokenizer) to the HF Hub."""
    from huggingface_hub import HfApi

    local_path = Path(local_path)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env file.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")


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


def score_pair(model: "RankingModel", tokenizer, topic: str, post: str, arg_a: str, arg_b: str) -> dict:
    """Score two candidate arguments. Returns scores, winner, and P(arg_a is more persuasive)."""
    model.eval()
    device = next(model.base.parameters()).device

    text_a = _trim_prompt_to_length(make_rank_prompt(topic, post, arg_a), tokenizer)
    text_b = _trim_prompt_to_length(make_rank_prompt(topic, post, arg_b), tokenizer)
    enc_a = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    enc_b = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

    with torch.no_grad():
        s_a = model.score(enc_a["input_ids"], enc_a["attention_mask"]).item()
        s_b = model.score(enc_b["input_ids"], enc_b["attention_mask"]).item()

    p_a = float(torch.sigmoid(torch.tensor(s_a - s_b)))
    return {
        "score_a": s_a,
        "score_b": s_b,
        "winner": "A" if s_a > s_b else "B",
        "prob_a_better": p_a,
    }


def _predict(model, tokenizer, fields: dict, labels: np.ndarray, use_summary: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.base.parameters()).device
    preds, probs = [], []
    for i in range(len(labels)):
        topic = fields["topic"][i]
        post = _get_post(fields, i, use_summary)
        arg_a, arg_b = fields["arg_a"][i], fields["arg_b"][i]

        text_a = _trim_prompt_to_length(make_rank_prompt(topic, post, arg_a), tokenizer)
        text_b = _trim_prompt_to_length(make_rank_prompt(topic, post, arg_b), tokenizer)
        enc_a = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        enc_b = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

        with torch.no_grad():
            s_a = model.score(enc_a["input_ids"], enc_a["attention_mask"]).item()
            s_b = model.score(enc_b["input_ids"], enc_b["attention_mask"]).item()

        # P(arg_a is delta) via sigmoid of score difference.
        p_a = float(torch.sigmoid(torch.tensor(s_a - s_b)))
        probs.append(p_a)
        preds.append(1 if s_a > s_b else 0)
    return np.array(preds), np.array(probs)


def run(
    train_fields,
    train_labels,
    val_fields,
    val_labels,
    test_fields,
    test_labels,
    use_summary: bool = False,
    **_,
) -> list[dict]:
    model_name = "llama_qlora_rank_summary" if use_summary else "llama_qlora_rank"

    model, tokenizer = _load_model()

    train_ds = _build_dataset(train_fields, train_labels, tokenizer, use_summary=use_summary)
    val_ds = _build_dataset(val_fields, val_labels, tokenizer, use_summary=use_summary)

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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=RANDOM_SEED,
        remove_unused_columns=False,
        label_names=[],
    )

    trainer = RankingTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PairCollator(tokenizer),
    )
    trainer.train()

    save_model(model, tokenizer, Path(f"./checkpoints/{model_name}/final"))

    results = []
    for split_name, fields, labels in [("val", val_fields, val_labels), ("test", test_fields, test_labels)]:
        y_pred, y_prob = _predict(model, tokenizer, fields, labels, use_summary)
        results.append({**evaluate(model_name, labels, y_pred, y_prob), "split": split_name, "n_train": len(train_labels)})
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-summary", action="store_true", help="Use the gpt-oss summary instead of original_post (default: original_post).")
    args = parser.parse_args()

    df = load_data()
    train_df, val_df, test_df = split_by_date(df)
    train_fields, train_labels = shuffle_pairs(train_df, seed=RANDOM_SEED)
    val_fields, val_labels = shuffle_pairs(val_df, seed=RANDOM_SEED + 1)
    test_fields, test_labels = shuffle_pairs(test_df, seed=RANDOM_SEED + 2)

    results = run(
        train_fields, train_labels,
        val_fields, val_labels,
        test_fields, test_labels,
        use_summary=args.use_summary,
    )
    save_results(results)
    for r in results:
        print(r)

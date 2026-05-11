"""
Fine-tune Llama-3.2-3B (base) with QLoRA (4-bit NF4) for argument quality prediction.

Usage:
    python -m models.llama
    python -m models.llama --use-summary

Training: SLURM HPC with a single GPU (~10GB VRAM).
Inference: compare logits of 'A' vs 'B' token at the answer position.
"""

import argparse
import os

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from models.data import RANDOM_SEED, evaluate, load_data, make_prompt, shuffle_pairs, split_by_date

MODEL_ID = "meta-llama/Llama-3.2-3B"
MAX_LENGTH = 4096

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

    # Measure tokens used by everything except the post body.
    post_marker = "### Post\n"
    split_marker = "\n\n### Response A"
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


def _tokenize(examples, tokenizer):
    out = tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding=False)
    out["labels"] = out["input_ids"].copy()
    return out


def _get_post(fields: dict, i: int, use_summary: bool) -> str:
    if use_summary:
        s = fields["original_post_summary"][i]
        if s is not None and str(s).strip():
            return str(s)
    return fields["original_post"][i]


def _build_dataset(fields: dict, labels: np.ndarray, tokenizer, include_answer: bool, use_summary: bool) -> Dataset:
    texts = [
        _trim_prompt_to_length(make_prompt(
            fields["topic"][i],
            _get_post(fields, i, use_summary),
            fields["arg_a"][i],
            fields["arg_b"][i],
            labels[i] if include_answer else None,
            use_summary=use_summary,
        ), tokenizer)
        for i in range(len(labels))
    ]
    ds = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    return ds.map(lambda x: _tokenize(x, tokenizer), batched=True, remove_columns=["text", "label"])


def _load_model():
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model, tokenizer


def _predict(model, tokenizer, fields: dict, labels: np.ndarray, use_summary: bool) -> tuple[np.ndarray, None]:
    model.eval()
    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    preds = []

    for i in range(len(labels)):
        prompt_text = _trim_prompt_to_length(make_prompt(
            fields["topic"][i],
            _get_post(fields, i, use_summary),
            fields["arg_a"][i],
            fields["arg_b"][i],
            use_summary=use_summary,
        ), tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        preds.append(1 if logits[a_id] >= logits[b_id] else 0)

    return np.array(preds), None


def run(train_fields, train_labels, val_fields, val_labels, test_fields, test_labels, use_summary: bool = False, **_) -> list[dict]:
    model_name = "llama_qlora_summary" if use_summary else "llama_qlora"

    model, tokenizer = _load_model()

    train_ds = _build_dataset(train_fields, train_labels, tokenizer, include_answer=True, use_summary=use_summary)
    val_ds = _build_dataset(val_fields, val_labels, tokenizer, include_answer=True, use_summary=use_summary)

    args = TrainingArguments(
        output_dir=f"./checkpoints/{model_name}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
    )
    trainer.train()

    results = []
    for split_name, fields, labels in [("val", val_fields, val_labels), ("test", test_fields, test_labels)]:
        y_pred, y_prob = _predict(model, tokenizer, fields, labels, use_summary)
        results.append({**evaluate(model_name, labels, y_pred, y_prob), "split": split_name})
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-summary", action="store_true", help="Use original_post_summary instead of original_post")
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
    for r in results:
        print(r)

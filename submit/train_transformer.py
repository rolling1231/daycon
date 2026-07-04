"""Fine-tune a small pretrained transformer encoder for action prediction (실험용, 아직 제출 파이프라인 아님)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from script import extract_text_prompt, flatten_history_compact, flatten_session_meta

sys.path.insert(0, str(Path(__file__).resolve().parent / "classical_model"))  # train_model.py 위치
from train_model import ALL_CLASSES, encode_labels, load_training_samples, log_experiment


def build_prompt_context(sample: dict) -> tuple[str, str]:
    """current_prompt는 보존 대상, context(history+meta)는 초과 시 잘려나가는 대상으로 분리."""
    prompt_text = extract_text_prompt(sample)
    context_text = (
        f"compact_history: {flatten_history_compact(sample.get('history'))} "
        f"| session_meta: {flatten_session_meta(sample.get('session_meta'))}"
    )
    return prompt_text, context_text


class ActionDataset(Dataset):
    def __init__(self, prompts: list[str], contexts: list[str], label_ids: list[int]):
        self.prompts = prompts
        self.contexts = contexts
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int):
        return self.prompts[idx], self.contexts[idx], self.label_ids[idx]


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        prompts, contexts, label_ids = zip(*batch)
        encoding = self.tokenizer(
            text=list(prompts),
            text_pair=list(contexts),
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        result = dict(encoding)
        # RoBERTa 계열은 segment(token_type) 임베딩을 안 쓰는 모델(type_vocab_size=1)이 많아서,
        # text_pair 토크나이징이 만드는 token_type_ids(0/1)를 그대로 넘기면 임베딩 인덱스 범위를 벗어난다.
        result.pop("token_type_ids", None)
        result["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return result


def save_transformer_files(model, tokenizer, output_dir: Path) -> list[str]:
    """model/ 폴더에 이미 있는 classical 모델(tfidf_logreg.pkl)을 건드리지 않도록,
    트랜스포머 파일들을 임시 폴더에 먼저 저장한 뒤 그 파일 목록만 output_dir로 복사한다."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        filenames = [f.name for f in tmp_path.iterdir() if f.is_file()]
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            shutil.copy2(tmp_path / name, output_dir / name)
    return filenames


def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(**batch, return_dict=True)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-name", default="klue/roberta-small")
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0이면 전체 사용, >0이면 스모크 테스트용으로 일부만 사용")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Load training samples...")
    samples, labels = load_training_samples(args.data_dir)
    if args.max_samples:
        samples = samples[: args.max_samples]
        labels = labels[: args.max_samples]
    print(f"samples={len(samples)} classes={len(set(labels))}")

    prompts, contexts = [], []
    for s in samples:
        p, c = build_prompt_context(s)
        prompts.append(p)
        contexts.append(c)
    label_ids = encode_labels(labels, ALL_CLASSES).tolist()

    indices = list(range(len(samples)))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.test_size, stratify=labels, random_state=args.random_state
    )

    def subset(idx_list):
        return (
            [prompts[i] for i in idx_list],
            [contexts[i] for i in idx_list],
            [label_ids[i] for i in idx_list],
        )

    train_prompts, train_contexts, train_label_ids = subset(train_idx)
    val_prompts, val_contexts, val_label_ids = subset(val_idx)
    print(f"train={len(train_prompts)} val={len(val_prompts)}")

    print(f"Load tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(ALL_CLASSES)
    ).to(device)

    collator = Collator(tokenizer, args.max_length)
    train_loader = DataLoader(
        ActionDataset(train_prompts, train_contexts, train_label_ids),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        ActionDataset(val_prompts, val_contexts, val_label_ids),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
    )

    train_label_arr = np.array(train_label_ids)
    class_counts = np.bincount(train_label_arr, minlength=len(ALL_CLASSES)).astype(np.float32)
    class_weights = train_label_arr.size / (len(ALL_CLASSES) * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_macro_f1 = -1.0
    best_weighted_f1 = -1.0
    best_filenames: list[str] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(**batch, return_dict=True)
                loss = loss_fn(outputs.logits, labels_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if step % 200 == 0:
                print(f"  epoch={epoch} step={step}/{len(train_loader)} loss={total_loss / step:.4f}")

        val_preds_idx, val_labels_idx = evaluate(model, val_loader, device)
        val_preds = [ALL_CLASSES[i] for i in val_preds_idx]
        val_labels_named = [ALL_CLASSES[i] for i in val_labels_idx]
        macro_f1 = f1_score(val_labels_named, val_preds, labels=ALL_CLASSES, average="macro", zero_division=0)
        weighted_f1 = f1_score(val_labels_named, val_preds, labels=ALL_CLASSES, average="weighted", zero_division=0)
        elapsed = time.time() - start
        print(
            f"epoch={epoch} elapsed={elapsed:.1f}s train_loss={total_loss / len(train_loader):.4f} "
            f"val_macro_f1={macro_f1:.6f} val_weighted_f1={weighted_f1:.6f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_weighted_f1 = weighted_f1
            best_filenames = save_transformer_files(model, tokenizer, args.output_dir)
            print(f"  -> new best (macro_f1={macro_f1:.6f}), saved to {args.output_dir} ({len(best_filenames)} files)")

    print(f"\nBest validation: macro_f1={best_macro_f1:.6f} weighted_f1={best_weighted_f1:.6f}")
    print(classification_report(val_labels_named, val_preds, labels=ALL_CLASSES, zero_division=0))

    mode_label = f"transformer_{args.model_name.replace('/', '_')}"
    backup_dir = Path("model_backup") / f"{mode_label}_f1{best_macro_f1:.4f}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in best_filenames:
        shutil.copy2(args.output_dir / name, backup_dir / name)
    print(f"Backup saved: {backup_dir} ({len(best_filenames)} files)")

    log_experiment(
        mode_label,
        best_macro_f1,
        best_weighted_f1,
        args.output_dir,
        backup_dir,
        extra={
            "epochs": args.epochs,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "files": best_filenames,
        },
    )


if __name__ == "__main__":
    main()

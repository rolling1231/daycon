"""Train and compare CPU-friendly TF-IDF models for action prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from baseline_submit.script import build_features, predict_adam_softmax_model


ALL_CLASSES = [
    "apply_patch",
    "ask_user",
    "edit_file",
    "glob_pattern",
    "grep_search",
    "lint_or_typecheck",
    "list_directory",
    "plan_task",
    "read_file",
    "respond_only",
    "run_bash",
    "run_tests",
    "web_search",
    "write_file",
]


def load_jsonl(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON parse failed: {exc}") from exc
    return samples


def load_labels(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def load_training_data(data_dir: Path) -> tuple[list[str], list[str]]:
    samples = load_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    ids, texts = build_features(samples)

    x_texts: list[str] = []
    y_labels: list[str] = []
    missing = 0
    for sample_id, text in zip(ids, texts):
        label = labels.get(sample_id)
        if label is None:
            missing += 1
            continue
        x_texts.append(text)
        y_labels.append(label)

    if missing:
        print(f"Warning: skipped {missing} samples without labels")
    return x_texts, y_labels


def word_vectorizer(max_features: int = 120_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )


def word_char_features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", word_vectorizer(max_features=120_000)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=160_000,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def encode_labels(labels: list[str], classes: list[str]) -> np.ndarray:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    return np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)


def balanced_sample_weights(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float32)
    weights_per_class = np.zeros(n_classes, dtype=np.float32)
    nonzero = counts > 0
    weights_per_class[nonzero] = y_idx.size / (n_classes * counts[nonzero])
    weights = weights_per_class[y_idx]
    mean_weight = float(weights.mean()) if weights.size else 1.0
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights.astype(np.float32)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    scores -= scores.max(axis=1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores


def predict_adam_softmax_matrix(x_matrix, weights: np.ndarray, bias: np.ndarray, classes: list[str], batch_size: int = 4096) -> list[str]:
    class_array = np.asarray(classes)
    preds: list[str] = []
    for start in range(0, x_matrix.shape[0], batch_size):
        x_batch = x_matrix[start:start + batch_size]
        scores = x_batch.dot(weights) + bias
        pred_idx = np.asarray(scores).argmax(axis=1)
        preds.extend(class_array[pred_idx].tolist())
    return preds


def train_adam_softmax(
    x_train,
    y_train_idx: np.ndarray,
    n_classes: int,
    *,
    sample_weight: np.ndarray | None = None,
    x_val=None,
    y_val_labels: list[str] | None = None,
    classes: list[str] | None = None,
    epochs: int = 8,
    batch_size: int = 2048,
    learning_rate: float = 0.03,
    l2: float = 1e-5,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    n_samples, n_features = x_train.shape
    weights = rng.normal(0.0, 0.001, size=(n_features, n_classes)).astype(np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)

    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    indices = np.arange(n_samples)
    best_macro_f1 = -1.0
    best_epoch = 0
    best_weights = weights.copy()
    best_bias = bias.copy()

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train_idx[batch_idx]

            scores = x_batch.dot(weights) + bias
            probs = softmax(np.asarray(scores))
            probs[np.arange(y_batch.size), y_batch] -= 1.0

            if sample_weight is not None:
                batch_weight = sample_weight[batch_idx].astype(np.float32)
                probs *= batch_weight[:, None]
                denom = max(float(batch_weight.sum()), 1.0)
            else:
                denom = float(y_batch.size)

            grad_w = (x_batch.T.dot(probs) / denom).astype(np.float32, copy=False)
            if l2:
                grad_w += l2 * weights
            grad_b = (probs.sum(axis=0) / denom).astype(np.float32, copy=False)

            step += 1
            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * (grad_w * grad_w)
            m_b = beta1 * m_b + (1.0 - beta1) * grad_b
            v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)

            m_w_hat = m_w / (1.0 - beta1 ** step)
            v_w_hat = v_w / (1.0 - beta2 ** step)
            m_b_hat = m_b / (1.0 - beta1 ** step)
            v_b_hat = v_b / (1.0 - beta2 ** step)

            weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

        if x_val is not None and y_val_labels is not None and classes is not None:
            val_pred = predict_adam_softmax_matrix(x_val, weights, bias, classes)
            macro_f1 = f1_score(y_val_labels, val_pred, labels=classes, average="macro", zero_division=0)
            print(f"epoch={epoch} val_macro_f1={macro_f1:.6f}")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_epoch = epoch
                best_weights = weights.copy()
                best_bias = bias.copy()
        else:
            best_epoch = epoch
            best_weights = weights.copy()
            best_bias = bias.copy()

    return best_weights, best_bias, best_epoch, best_macro_f1


def make_candidates() -> dict[str, Pipeline]:
    return {
        "word_logreg_balanced": Pipeline(
            [
                ("features", word_vectorizer(max_features=120_000)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=800,
                        class_weight="balanced",
                        C=2.0,
                    ),
                ),
            ]
        ),
        "word_linearsvc_balanced": Pipeline(
            [
                ("features", word_vectorizer(max_features=120_000)),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=5_000,
                    ),
                ),
            ]
        ),
        "word_linearsvc_unweighted": Pipeline(
            [
                ("features", word_vectorizer(max_features=120_000)),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight=None,
                        dual="auto",
                        max_iter=5_000,
                    ),
                ),
            ]
        ),
        "word_char_linearsvc_balanced": Pipeline(
            [
                ("features", word_char_features()),
                (
                    "clf",
                    LinearSVC(
                        C=0.8,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=5_000,
                    ),
                ),
            ]
        ),
        "word_logreg_unweighted": Pipeline(
            [
                ("features", word_vectorizer(max_features=120_000)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=800,
                        class_weight=None,
                        C=2.0,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("baseline_submit") / "model")
    parser.add_argument("--model-name", default="tfidf_logreg.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["linearsvc_final", "adam_softmax", "compare_sklearn"],
        default="linearsvc_final",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--no-balanced", action="store_true")
    args = parser.parse_args()

    print("Load training data...")
    texts, labels = load_training_data(args.data_dir)
    print(f"samples={len(texts)} classes={len(set(labels))}")

    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.random_state,
    )
    print(f"train={len(x_train)} val={len(x_val)}")

    if args.mode == "linearsvc_final":
        print("\nTrain final candidate: script.extract_text + word+char TF-IDF + LinearSVC balanced C=0.8")
        model = Pipeline(
            [
                ("features", word_char_features()),
                (
                    "clf",
                    LinearSVC(
                        C=0.8,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=7_000,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        macro_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="weighted", zero_division=0)
        print(f"LinearSVC validation: macro_f1={macro_f1:.6f} weighted_f1={weighted_f1:.6f}")
        print(classification_report(y_val, val_pred, labels=ALL_CLASSES, zero_division=0))

        print("Retrain final candidate on all training data...")
        final_model = Pipeline(
            [
                ("features", word_char_features()),
                (
                    "clf",
                    LinearSVC(
                        C=0.8,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=7_000,
                        random_state=args.random_state,
                    ),
                ),
            ]
        )
        final_model.fit(texts, labels)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.model_dir / args.model_name
        joblib.dump(final_model, model_path, compress=3)
        print(f"Saved model: {model_path}")
        return

    if args.mode == "adam_softmax":
        classes = ALL_CLASSES
        n_classes = len(classes)

        print("\nFit TF-IDF word+char features...")
        features = word_char_features()
        x_train_matrix = features.fit_transform(x_train)
        x_val_matrix = features.transform(x_val)
        print(f"x_train_shape={x_train_matrix.shape} x_val_shape={x_val_matrix.shape}")

        y_train_idx = encode_labels(y_train, classes)
        sample_weight = None if args.no_balanced else balanced_sample_weights(y_train_idx, n_classes)

        print("\nTrain Adam + Softmax classifier...")
        weights, bias, best_epoch, best_macro_f1 = train_adam_softmax(
            x_train_matrix,
            y_train_idx,
            n_classes,
            sample_weight=sample_weight,
            x_val=x_val_matrix,
            y_val_labels=y_val,
            classes=classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2=args.l2,
            random_state=args.random_state,
        )
        val_pred = predict_adam_softmax_matrix(x_val_matrix, weights, bias, classes)
        macro_f1 = f1_score(y_val, val_pred, labels=classes, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, val_pred, labels=classes, average="weighted", zero_division=0)
        print(f"\nAdam-Softmax validation: best_epoch={best_epoch} macro_f1={macro_f1:.6f} weighted_f1={weighted_f1:.6f}")
        print(classification_report(y_val, val_pred, labels=classes, zero_division=0))

        print("\nRetrain Adam-Softmax on all training data...")
        final_features = word_char_features()
        x_all_matrix = final_features.fit_transform(texts)
        y_all_idx = encode_labels(labels, classes)
        final_sample_weight = None if args.no_balanced else balanced_sample_weights(y_all_idx, n_classes)
        final_weights, final_bias, _, _ = train_adam_softmax(
            x_all_matrix,
            y_all_idx,
            n_classes,
            sample_weight=final_sample_weight,
            epochs=best_epoch or args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2=args.l2,
            random_state=args.random_state,
        )

        model = {
            "model_type": "tfidf_adam_softmax",
            "features": final_features,
            "weights": final_weights.astype(np.float32, copy=False),
            "bias": final_bias.astype(np.float32, copy=False),
            "classes": np.asarray(classes, dtype=object),
            "best_epoch": best_epoch,
            "validation_macro_f1": float(macro_f1),
            "validation_weighted_f1": float(weighted_f1),
            "optimizer": "adam",
            "distribution": "softmax",
            "balanced": not args.no_balanced,
        }

        args.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.model_dir / args.model_name
        joblib.dump(model, model_path, compress=3)
        print(f"Saved model: {model_path}")

        smoke_pred = predict_adam_softmax_model(model, texts[:5])
        print(f"Train smoke predictions (first 5): {smoke_pred}")
        return

    candidates = make_candidates()
    results: list[tuple[float, str, Pipeline]] = []
    for name, model in candidates.items():
        print(f"\nTrain candidate: {name}")
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        macro_f1 = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, pred, labels=ALL_CLASSES, average="weighted", zero_division=0)
        print(f"{name}: macro_f1={macro_f1:.6f} weighted_f1={weighted_f1:.6f}")
        results.append((macro_f1, name, model))

    best_macro_f1, best_name, _ = max(results, key=lambda item: item[0])
    print("\nValidation summary:")
    for macro_f1, name, _ in sorted(results, reverse=True):
        print(f"  {name}: macro_f1={macro_f1:.6f}")

    print(f"\nBest candidate: {best_name} macro_f1={best_macro_f1:.6f}")
    final_model = make_candidates()[best_name]
    print("Retrain best candidate on all training data...")
    final_model.fit(texts, labels)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / args.model_name
    joblib.dump(final_model, model_path, compress=3)
    print(f"Saved model: {model_path}")

    val_pred = results[[name for _, name, _ in results].index(best_name)][2].predict(x_val)
    print("\nBest validation classification report:")
    print(classification_report(y_val, val_pred, labels=ALL_CLASSES, zero_division=0))


if __name__ == "__main__":
    main()

"""Offline inference entrypoint for the action prediction competition."""

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")
MAX_CURRENT_PROMPT_CHARS = 4_000
MAX_HISTORY_ITEMS = 8
MAX_HISTORY_CHARS = 3_000
MAX_VALUE_CHARS = 800

# train_transformer.py에서 encode_labels(labels, ALL_CLASSES)로 학습할 때 쓴 것과 동일한 순서.
# 트랜스포머 분류 헤드의 출력 인덱스가 이 리스트의 인덱스와 대응된다.
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


def _clean_text(value: Any, max_chars: int = MAX_VALUE_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def _bucket_number(value: Any, bins: Iterable[int], prefix: str) -> str:
    bounds = list(bins)
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return f"{prefix}_unknown"
    for bound in bounds:
        if number <= bound:
            return f"{prefix}_le_{bound}"
    return f"{prefix}_gt_{bounds[-1]}"


def _path_tokens(paths: Any, max_paths: int = 10) -> list[str]:
    if not isinstance(paths, list):
        return []

    tokens: list[str] = []
    for raw_path in paths[-max_paths:]:
        path_text = _clean_text(raw_path, max_chars=200)
        if not path_text:
            continue
        suffix = Path(path_text).suffix.lower().lstrip(".")
        name = Path(path_text).name.lower()
        tokens.append(f"open_file {path_text}")
        if suffix:
            tokens.append(f"open_file_ext_{suffix}")
        if name:
            tokens.append(f"open_file_name_{name}")
    return tokens


def flatten_session_meta(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""

    parts: list[str] = []
    for key in ("user_tier", "language_pref"):
        value = _clean_text(meta.get(key), max_chars=80)
        if value:
            parts.append(f"{key} {value}")

    parts.append(_bucket_number(meta.get("budget_tokens_remaining"), [1_000, 5_000, 10_000, 25_000, 50_000, 100_000], "budget"))
    parts.append(_bucket_number(meta.get("turn_index"), [0, 1, 2, 4, 6, 8, 10, 12, 16], "turn"))
    parts.append(_bucket_number(meta.get("elapsed_session_sec"), [60, 180, 300, 600, 900, 1_800, 3_600], "elapsed"))

    workspace = meta.get("workspace")
    if isinstance(workspace, dict):
        parts.append(_bucket_number(workspace.get("loc"), [500, 2_000, 5_000, 10_000, 25_000, 50_000], "loc"))
        parts.append(f"git_dirty {bool(workspace.get('git_dirty'))}")

        ci_status = _clean_text(workspace.get("last_ci_status"), max_chars=80)
        if ci_status:
            parts.append(f"last_ci_status {ci_status}")

        language_mix = workspace.get("language_mix")
        if isinstance(language_mix, dict):
            try:
                lang_items = sorted(language_mix.items(), key=lambda item: float(item[1]), reverse=True)
            except (TypeError, ValueError):
                lang_items = list(language_mix.items())
            for lang, share in lang_items[:8]:
                lang_text = _clean_text(lang, max_chars=40)
                if lang_text:
                    parts.append(f"workspace_lang {lang_text}")
                try:
                    share_pct = int(round(float(share) * 100))
                    parts.append(f"workspace_lang_share_{lang_text}_{share_pct}")
                except (TypeError, ValueError):
                    pass

        open_files = workspace.get("open_files")
        open_files_count = len(open_files) if isinstance(open_files, list) else 0
        parts.append(_bucket_number(open_files_count, [0, 1, 2, 3, 5, 8], "open_files_count"))
        parts.extend(_path_tokens(open_files))

    return " | ".join(part for part in parts if part)


def _history_item_to_text(item: Any) -> str:
    if not isinstance(item, dict):
        return f"history_item {_clean_text(item)}"

    role = _clean_text(item.get("role"), max_chars=80) or "unknown"
    if role == "assistant_action":
        action = _clean_text(item.get("name"), max_chars=80)
        args = _clean_text(item.get("args"), max_chars=400)
        result = _clean_text(item.get("result_summary"), max_chars=300)
        return f"history_role {role} previous_action {action} action_args {args} action_result {result}"

    content = _clean_text(item.get("content"), max_chars=MAX_VALUE_CHARS)
    name = _clean_text(item.get("name"), max_chars=80)
    if name:
        return f"history_role {role} name {name} content {content}"
    return f"history_role {role} content {content}"


def flatten_history(history: Any) -> str:
    if not isinstance(history, list) or not history:
        return ""

    selected_reversed: list[str] = []
    used_chars = 0
    for item in reversed(history):
        text = _history_item_to_text(item)
        if not text:
            continue
        if selected_reversed and used_chars + len(text) > MAX_HISTORY_CHARS:
            break
        selected_reversed.append(text)
        used_chars += len(text)
        if len(selected_reversed) >= MAX_HISTORY_ITEMS:
            break

    return "\n".join(reversed(selected_reversed))


def flatten_history_compact(history: Any) -> str:
    if not isinstance(history, list) or not history:
        return ""

    action_names: list[str] = []
    action_counts: dict[str, int] = {}
    result_flags: list[str] = []
    arg_tokens: list[str] = []
    user_turns = 0

    for item in history:
        if not isinstance(item, dict):
            continue

        role = _clean_text(item.get("role"), max_chars=80)
        if role == "user":
            user_turns += 1
            continue

        if role != "assistant_action":
            continue

        action = _clean_text(item.get("name"), max_chars=80)
        if action:
            action_names.append(action)
            action_counts[action] = action_counts.get(action, 0) + 1

        args = item.get("args")
        if isinstance(args, dict):
            for key, value in sorted(args.items()):
                key_text = _clean_text(key, max_chars=40)
                if key_text:
                    arg_tokens.append(f"arg_key_{key_text}")
                value_text = _clean_text(value, max_chars=160).lower()
                suffix = Path(value_text).suffix.lower().lstrip(".")
                if suffix:
                    arg_tokens.append(f"arg_ext_{suffix}")
                if key_text in {"target", "path", "scope"} and value_text:
                    arg_tokens.append(f"arg_{key_text}_{value_text[:80]}")

        result = _clean_text(item.get("result_summary"), max_chars=240).lower()
        if result:
            if "pass" in result or "passed" in result:
                result_flags.append("result_pass")
            if "fail" in result or "failed" in result:
                result_flags.append("result_fail")
            if "error" in result or "traceback" in result or "exception" in result:
                result_flags.append("result_error")
            if "match" in result or "matches" in result:
                result_flags.append("result_matches")
            if "test" in result or "tests" in result:
                result_flags.append("result_tests")

    parts: list[str] = [
        f"history_items_{len(history)}",
        f"history_user_turns_{user_turns}",
        f"history_action_turns_{len(action_names)}",
    ]

    if action_names:
        parts.append(f"last_action_{action_names[-1]}")
        recent = action_names[-3:]
        parts.extend(f"recent_action_{idx}_{name}" for idx, name in enumerate(recent, start=1))
        parts.append("recent_action_seq_" + "_".join(recent))

    for action, count in sorted(action_counts.items()):
        capped = min(count, 5)
        parts.append(f"action_count_{action}_{capped}")

    parts.extend(result_flags[-8:])
    parts.extend(arg_tokens[-12:])
    return " | ".join(parts)


def extract_text(sample: dict[str, Any]) -> str:
    """Build the exact model input string used by both training and inference."""

    return extract_text_prompt_meta_compact_history(sample)


def extract_text_prompt(sample: dict[str, Any]) -> str:
    return f"current_prompt: {_clean_text(sample.get('current_prompt'), max_chars=MAX_CURRENT_PROMPT_CHARS)}"


def extract_text_prompt_history(sample: dict[str, Any]) -> str:
    sections = [
        f"recent_history: {flatten_history(sample.get('history'))}",
        f"current_prompt: {_clean_text(sample.get('current_prompt'), max_chars=MAX_CURRENT_PROMPT_CHARS)}",
    ]
    return "\n".join(section for section in sections if section.strip())


def extract_text_prompt_meta_compact_history(sample: dict[str, Any]) -> str:
    current_prompt = _clean_text(sample.get("current_prompt"), max_chars=MAX_CURRENT_PROMPT_CHARS)
    session_meta = flatten_session_meta(sample.get("session_meta"))
    compact_history = flatten_history_compact(sample.get("history"))
    sections = [
        f"session_meta: {session_meta}",
        f"compact_history: {compact_history}",
        f"current_prompt: {current_prompt}",
    ]
    return "\n".join(section for section in sections if section.strip())


def extract_text_by_mode(sample: dict[str, Any], mode: str) -> str:
    if mode == "prompt":
        return extract_text_prompt(sample)
    if mode == "prompt_history":
        return extract_text_prompt_history(sample)
    if mode == "prompt_meta":
        return extract_text(sample)
    if mode == "prompt_meta_compact_history":
        return extract_text_prompt_meta_compact_history(sample)
    raise ValueError(f"unknown feature mode: {mode}")


def build_features(samples: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    for sample in samples:
        ids.append(_clean_text(sample.get("id"), max_chars=200))
        texts.append(extract_text(sample))
    return ids, texts


def load_jsonl(path):
    """평가 데이터(jsonl) 로드. 한 줄당 샘플 하나."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}")
            samples.append(obj)
    return samples


def validate_samples(samples):
    """필수 키 존재 여부 검증 (학습 데이터와 동일 스키마)."""
    n_bad = 0
    for s in samples:
        for k in REQUIRED_KEYS:
            if k not in s:
                n_bad += 1
                break
    if n_bad:
        print(f" 경고: 필수 키 누락 샘플 {n_bad}건 (빈 텍스트로 처리)")
    return n_bad


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 id 순서/컬럼 기준."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(
            f"sample_submission 컬럼이 (id, action)이 아님: {fieldnames}")
    return fieldnames, rows


def merge_predictions(sub_rows, ids, preds, fallback_action):
    """sample_submission의 id 순서에 맞춰 예측값 병합.

    예측에 없는 id는 fallback_action으로 채워 제출 action 공백을 방지한다.
    """
    pred_map = {str(sample_id): str(pred) for sample_id, pred in zip(ids, preds)}
    n_missing = 0
    for row in sub_rows:
        p = pred_map.get(str(row.get("id", "")))
        if p is None:
            n_missing += 1
            row["action"] = fallback_action
        else:
            row["action"] = p
    if n_missing:
        print(f" 경고: 예측이 없어 fallback으로 채운 id {n_missing}건")
    return sub_rows


def save_submission(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(candidates, description):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    tried = ", ".join(str(Path(candidate)) for candidate in candidates)
    raise FileNotFoundError(f"{description}을 찾을 수 없습니다. 확인 경로: {tried}")


def predict_adam_softmax_model(model, texts, batch_size=4096):
    features = model["features"]
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = np.asarray(model["bias"], dtype=np.float32)
    classes = np.asarray(model["classes"])

    preds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        x_batch = features.transform(batch_texts)
        scores = x_batch.dot(weights) + bias
        pred_idx = np.asarray(scores).argmax(axis=1)
        preds.extend(classes[pred_idx].tolist())
    return preds


def _align_scores(scores, src_classes, dst_classes):
    src_classes = [str(label) for label in src_classes]
    aligned = np.zeros((scores.shape[0], len(dst_classes)), dtype=np.float32)
    for src_idx, label in enumerate(src_classes):
        if label in dst_classes:
            aligned[:, dst_classes.index(label)] = scores[:, src_idx]
    return aligned


def _normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    center = scores.mean(axis=1, keepdims=True)
    scale = scores.std(axis=1, keepdims=True)
    return (scores - center) / np.maximum(scale, 1e-6)


def predict_ensemble_model(model, samples):
    classes = [str(label) for label in model["classes"]]
    combined = None
    normalize = bool(model.get("normalize_scores", True))

    for component in model["components"]:
        mode = component["mode"]
        estimator = component["model"]
        weight = float(component.get("weight", 1.0))
        texts = [extract_text_by_mode(sample, mode) for sample in samples]
        scores = estimator.decision_function(texts)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        src_classes = getattr(estimator, "classes_", classes)
        aligned = _align_scores(np.asarray(scores), src_classes, classes)
        if normalize:
            aligned = _normalize_scores(aligned)
        weighted = aligned * weight
        combined = weighted if combined is None else combined + weighted

    pred_idx = combined.argmax(axis=1)
    return [classes[i] for i in pred_idx]


def predict_model(model, texts):
    if isinstance(model, dict) and model.get("model_type") == "tfidf_adam_softmax":
        return predict_adam_softmax_model(model, texts)
    return model.predict(texts)


def get_model_classes(model):
    if isinstance(model, dict) and "classes" in model:
        return list(model["classes"])
    return list(getattr(model, "classes_", []))


def find_transformer_dir(candidates):
    """model/ 밑에 config.json이 있으면 트랜스포머 체크포인트가 저장된 것으로 판단한다."""
    for candidate in candidates:
        candidate = Path(candidate)
        if (candidate / "config.json").exists():
            return candidate
    return None


def load_transformer_model(model_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_transformer_model(model, tokenizer, samples, batch_size=64, max_length=512):
    """train_transformer.py와 동일한 입력 구성(현재 프롬프트 보존, 초과분은 session_meta부터 절단)으로 추론."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        prompts = [extract_text_prompt(sample) for sample in batch_samples]
        contexts = [
            f"compact_history: {flatten_history_compact(sample.get('history'))} "
            f"| session_meta: {flatten_session_meta(sample.get('session_meta'))}"
            for sample in batch_samples
        ]
        encoding = tokenizer(
            text=prompts,
            text_pair=contexts,
            truncation="only_second",
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoding.pop("token_type_ids", None)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding, return_dict=True)
        pred_idx = outputs.logits.argmax(dim=-1).cpu().numpy()
        preds.extend(ALL_CLASSES[i] for i in pred_idx)
    return preds


def main():
    base_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    model_filename = "tfidf_logreg.pkl"

    test_path = first_existing(
        [cwd / "data" / "test.jsonl", base_dir / "data" / "test.jsonl", base_dir.parent / "data" / "test.jsonl"],
        "test.jsonl",
    )
    sample_sub_path = first_existing(
        [
            cwd / "data" / "sample_submission.csv",
            base_dir / "data" / "sample_submission.csv",
            base_dir.parent / "data" / "sample_submission.csv",
        ],
        "sample_submission.csv",
    )
    out_path = cwd / "output" / "submission.csv"

    print("Load test data...")
    samples = load_jsonl(test_path)
    validate_samples(samples)
    print(f" samples={len(samples)}")

    transformer_dir = find_transformer_dir([cwd / "model", base_dir / "model"])

    if transformer_dir is not None:
        print(f"Load transformer model from {transformer_dir}...")
        model, tokenizer = load_transformer_model(transformer_dir)
        print(" OK.")

        ids = [_clean_text(sample.get("id"), max_chars=200) for sample in samples]
        print("Inference model (transformer)...")
        preds = predict_transformer_model(model, tokenizer, samples) if samples else []
        fallback_action = ALL_CLASSES[0]
    else:
        model_filename_path = first_existing(
            [cwd / "model" / model_filename, base_dir / "model" / model_filename],
            model_filename,
        )
        print("Load model...")
        model = joblib.load(model_filename_path)
        classes = get_model_classes(model)
        fallback_action = str(classes[0]) if classes else "edit_file"
        print(f" OK. path={model_filename_path} classes={len(classes)}")

        print("Build features...")
        ids, texts = build_features(samples)
        print(f" texts={len(texts)}")

        print("Inference model...")
        if isinstance(model, dict) and model.get("model_type") == "mode_ensemble":
            preds = predict_ensemble_model(model, samples) if samples else []
        else:
            preds = predict_model(model, texts) if texts else []

    preds = [str(p) for p in preds]
    print(f" preds={len(preds)}")

    print("Build submission...")
    fieldnames, sub_rows = load_sample_submission(sample_sub_path)
    sub_rows = merge_predictions(sub_rows, ids, preds, fallback_action)
    save_submission(out_path, fieldnames, sub_rows)
    print(f"Saved: {out_path} (rows={len(sub_rows)})")


if __name__ == "__main__":
    main()

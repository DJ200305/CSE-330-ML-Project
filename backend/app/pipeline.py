from __future__ import annotations

import io
import json
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import AppConfig
from .logbert import LogBERT
from .llm import LLMRuntime, generate_analysis, load_llm, parse_llm_response


@dataclass
class Runtime:
    config: AppConfig
    event2id: Dict[str, int]
    id_to_template: Dict[str, str]
    model: LogBERT
    device: torch.device
    llm: LLMRuntime


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    config = AppConfig.load()
    event2id = _read_json(config.event2id_path)
    id_to_template = _read_json(config.id_to_template_path)

    if not event2id:
        raise RuntimeError(f"event2id not found at {config.event2id_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogBERT(len(event2id)).to(device)
    state = torch.load(str(config.logbert_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    llm = load_llm(None, config.remote_llm_url)

    return Runtime(
        config=config,
        event2id=event2id,
        id_to_template=id_to_template,
        model=model,
        device=device,
        llm=llm,
    )


def _parse_csv(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Unable to parse CSV log file.") from exc


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"EventId", "Timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _build_sequences(df: pd.DataFrame, window_size: int, step_size: int) -> List[List[str]]:
    df_sorted = df.sort_values("Timestamp").reset_index(drop=True)
    event_ids = df_sorted["EventId"].astype(str).tolist()
    sequences: List[List[str]] = []
    for i in range(0, len(event_ids) - window_size, step_size):
        seq = event_ids[i : i + window_size]
        sequences.append(seq)
    return sequences


def _mask_sequence(
    seq: List[str],
    event2id: Dict[str, int],
    mask_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_id = event2id.get("[CLS]")
    mask_id = event2id.get("[MASK]")
    unk_id = event2id.get("[UNK]")
    specials = ["[PAD]", "[CLS]", "[MASK]", "[UNK]"]

    if cls_id is None or mask_id is None or unk_id is None:
        raise RuntimeError("event2id must contain [CLS], [MASK], and [UNK] tokens.")

    encoded = [event2id.get(eid, unk_id) for eid in seq]
    tokens = [cls_id] + encoded

    input_ids: List[int] = []
    output_labels: List[int] = []

    for i, token in enumerate(tokens):
        if i == 0:
            input_ids.append(token)
            output_labels.append(-100)
            continue

        if random.random() < mask_ratio:
            p = random.random()
            if p < 0.80:
                input_ids.append(mask_id)
            elif p < 0.90:
                input_ids.append(random.randint(len(specials), len(event2id) - 1))
            else:
                input_ids.append(token)
            output_labels.append(token)
        else:
            input_ids.append(token)
            output_labels.append(-100)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(output_labels, dtype=torch.long),
        torch.tensor(tokens, dtype=torch.long),
    )


def _score_sequences(
    runtime: Runtime,
    sequences: List[List[str]],
) -> Tuple[List[Dict[str, Any]], int]:
    anomalies: List[Dict[str, Any]] = []
    skipped = 0
    g = runtime.config.top_g

    for idx, seq in enumerate(sequences):
        input_ids, labels, original_ids = _mask_sequence(
            seq, runtime.event2id, runtime.config.mask_ratio
        )
        if labels.numel() == 0 or (labels != -100).sum().item() == 0:
            continue

        input_ids = input_ids.unsqueeze(0).to(runtime.device)
        labels = labels.to(runtime.device)

        with torch.inference_mode():
            logits, _ = runtime.model(input_ids)

        probs = torch.softmax(logits, dim=-1)
        top_g_idx = torch.topk(probs, g, dim=-1).indices.cpu()
        y_cpu = labels.cpu()
        orig_cpu = original_ids.cpu()

        wrong = 0
        total_masked = 0
        for j in range(1, y_cpu.size(0)):
            if y_cpu[j].item() == -100:
                continue
            total_masked += 1
            real_token = orig_cpu[j].item()
            if real_token not in top_g_idx[0, j].tolist():
                wrong += 1

        if total_masked == 0:
            continue

        if wrong > runtime.config.threshold:
            anomalies.append(
                {
                    "sequence_id": idx,
                    "anomaly_score": float(wrong),
                    "event_ids": seq,
                    "event_templates": [
                        runtime.id_to_template.get(e, e) for e in seq
                    ],
                }
            )

    return anomalies, skipped


def _extract_verdict(text: str) -> str:
    match = re.search(r"Verdict\s*:\s*(\w+)", text, re.IGNORECASE)
    if not match:
        return "unknown"
    return match.group(1).lower()


def analyze_log_file(content: bytes, filename: str) -> Dict[str, Any]:
    runtime = get_runtime()
    # Deterministic masking for stable anomaly counts.
    seed = int(os.getenv("INFER_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    df = _parse_csv(content)
    _validate_columns(df)

    if not runtime.id_to_template and "EventTemplate" in df.columns:
        runtime.id_to_template = dict(zip(df["EventId"].astype(str), df["EventTemplate"]))

    sequences = _build_sequences(
        df,
        runtime.config.window_size,
        runtime.config.step_size,
    )
    anomalies, skipped = _score_sequences(runtime, sequences)

    results: List[Dict[str, Any]] = []
    for entry in anomalies:
        log_seq_list = entry["event_templates"]
        llm_text = generate_analysis(runtime.llm, log_seq_list)
        parsed = parse_llm_response(llm_text)
        verdict = parsed.get("verdict") or _extract_verdict(llm_text)
        results.append(
            {
                **entry,
                "llm_verdict": verdict,
                "llm_explanation": parsed.get("explanation"),
                "llm_solution": parsed.get("solution"),
                "llm_response": llm_text,
            }
        )

    return {
        "file": filename,
        "total_sequences": len(sequences),
        "skipped_sequences": skipped,
        "anomalies_found": len(anomalies),
        "results": results,
    }

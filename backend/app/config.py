from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict 


BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    # Handle UTF-8 BOM if present.
    return json.loads(path.read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class AppConfig:
    logbert_path: Path
    event2id_path: Path
    id_to_template_path: Path
    remote_llm_url: str
    window_size: int
    step_size: int
    mask_ratio: float
    top_g: int
    threshold: float
    max_anomalies: int

    @staticmethod
    def load() -> "AppConfig":
        config_path = Path(os.getenv("APP_CONFIG", ""))
        if config_path and config_path.is_dir():
            config_path = config_path / "config.json"
        if not config_path or not config_path.exists():
            config_path = RESOURCES_DIR / "config.json"
        cfg = _read_json(config_path)

        def _get_path(key: str, default: Path) -> Path:
            raw = os.getenv(key, "") or cfg.get(key, "")
            return (Path(raw) if raw else default).resolve()

        env_remote = os.getenv("REMOTE_LLM_URL")
        remote_llm_url = ""
        if env_remote and env_remote.strip():
            remote_llm_url = env_remote.strip()
        else:
            remote_llm_url = str(cfg.get("REMOTE_LLM_URL", "")).strip()

        return AppConfig(
            logbert_path=_get_path("LOGBERT_PATH", RESOURCES_DIR / "logbert.pt"),
            event2id_path=_get_path("EVENT2ID_PATH", RESOURCES_DIR / "event2id.json"),
            id_to_template_path=_get_path(
                "ID_TO_TEMPLATE_PATH", RESOURCES_DIR / "id_to_template.json"
            ),
            remote_llm_url=remote_llm_url,
            window_size=int(os.getenv("WINDOW_SIZE", cfg.get("WINDOW_SIZE", 128))),
            step_size=int(os.getenv("STEP_SIZE", cfg.get("STEP_SIZE", 32))),
            mask_ratio=float(os.getenv("MASK_RATIO", cfg.get("MASK_RATIO", 0.15))),
            top_g=int(os.getenv("TOP_G", cfg.get("TOP_G", 5))),
            threshold=float(os.getenv("THRESHOLD", cfg.get("THRESHOLD", 14))),
            max_anomalies=int(os.getenv("MAX_ANOMALIES", cfg.get("MAX_ANOMALIES", 50))),
        )

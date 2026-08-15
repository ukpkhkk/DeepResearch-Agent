from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from deep_research.utils import load_config


DEFAULT_SKILL_MEMORY_CONFIG: dict[str, Any] = {
    "enabled": True,
    "trajectory_dir": None,
    "max_retrieved_skills": 3,
    "min_quality_score": 8.0,
    "min_accuracy_score": 8,
    "require_no_open_critiques": True,
    "distill_model_role": "writer",
    "vector_store": {
        "backend": "chroma",
        "persist_directory": None,
        "collection_name": "research_skills",
        "embedding_backend": "hash",
        "embedding_model": "hash-384",
        "similarity_threshold": 0.35,
        "fetch_k": 12,
    },
    "retrieval": {
        "backend": "vector",
        "rerank_enabled": True,
        "quality_weight": 0.25,
        "success_weight": 0.15,
        "recency_weight": 0.05,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "_skill_memory"


def get_skill_memory_config(stage: str | None = None) -> dict[str, Any]:
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    stage_name = stage or os.environ.get("STAGE") or "prod"
    cfg: dict[str, Any] = {}
    try:
        cfg = load_config(stage_name=stage_name, config_path=config_path) or {}
    except Exception:
        cfg = {}

    skill_cfg = _deep_merge(DEFAULT_SKILL_MEMORY_CONFIG, cfg.get("skill_memory") or {})
    skill_cfg["_cognition_openai"] = (cfg.get("cognition") or {}).get("openai") or {}
    data_dir = _default_data_dir()
    vector_cfg = skill_cfg.setdefault("vector_store", {})
    if not skill_cfg.get("trajectory_dir"):
        skill_cfg["trajectory_dir"] = str(data_dir / "trajectories")
    if not vector_cfg.get("persist_directory"):
        vector_cfg["persist_directory"] = str(data_dir / "vector_store")
    return skill_cfg

from __future__ import annotations

import json
from pathlib import Path

from deep_research.skills.config import get_skill_memory_config
from deep_research.skills.distiller import distill_skill_from_trajectory
from deep_research.skills.extractor import extract_trajectory_from_state
from deep_research.skills.retriever import retrieve_skills_for_state
from deep_research.skills.store import get_skill_store


def retrieve_research_skills(state: dict) -> dict:
    skill_prompts, skill_ids = retrieve_skills_for_state(state)
    return {
        "retrieved_skills": skill_prompts,
        "retrieved_skill_ids": skill_ids,
    }


def extract_research_trajectory(state: dict) -> dict:
    trajectory = extract_trajectory_from_state(state)
    return {"trajectory": trajectory}


def generate_research_skill(state: dict) -> dict:
    trajectory = state.get("trajectory")
    if trajectory is None:
        trajectory = extract_trajectory_from_state(state)
    skill = distill_skill_from_trajectory(trajectory)
    return {"generated_skills": [skill]} if skill else {"generated_skills": []}


def persist_research_skills(state: dict) -> dict:
    cfg = get_skill_memory_config()
    if not cfg.get("enabled", True):
        return {}

    trajectory = state.get("trajectory")
    if trajectory is not None:
        try:
            trajectory_dir = Path(cfg["trajectory_dir"])
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            path = trajectory_dir / f"{trajectory.run_id}.json"
            path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
        except Exception:
            pass

    skills = [skill for skill in state.get("generated_skills", []) if skill is not None]
    if not skills:
        return {}
    try:
        store = get_skill_store(cfg)
        for skill in skills:
            store.add_skill(skill)
    except Exception:
        return {}
    return {"persisted_skill_ids": [skill.skill_id for skill in skills]}

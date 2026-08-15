from __future__ import annotations

from deep_research.skills.config import get_skill_memory_config
from deep_research.skills.prompting import format_skill_for_prompt
from deep_research.skills.store import get_skill_store


def _get_buffer_string(messages) -> str:
    try:
        from langchain_core.messages import get_buffer_string

        return get_buffer_string(messages)
    except Exception:
        return "\n".join(str(getattr(message, "content", message)) for message in messages)


def retrieve_skills_for_state(state: dict) -> tuple[list[str], list[str]]:
    cfg = get_skill_memory_config()
    if not cfg.get("enabled", True):
        return [], []
    vector_cfg = cfg.get("vector_store") or {}
    user_query = _get_buffer_string(state.get("messages", []))
    research_brief = state.get("research_brief", "")
    query = f"用户问题：{user_query}\n研究简报：{research_brief}\n期望输出：深度研究报告"
    try:
        store = get_skill_store(cfg)
        results = store.search(
            query,
            top_k=int(cfg.get("max_retrieved_skills", 3)),
            fetch_k=int(vector_cfg.get("fetch_k", 12)),
            min_quality_score=float(cfg.get("min_quality_score", 8.0)),
            similarity_threshold=float(vector_cfg.get("similarity_threshold", 0.35)),
        )
    except Exception:
        return [], []
    return [format_skill_for_prompt(result.skill) for result in results], [result.skill.skill_id for result in results]

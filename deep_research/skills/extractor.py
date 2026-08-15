from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from deep_research.skills.schema import (
    CritiqueSnapshot,
    EvaluationSnapshot,
    ResearchTrajectory,
)


def _get_buffer_string(messages) -> str:
    try:
        from langchain_core.messages import get_buffer_string

        return get_buffer_string(messages)
    except Exception:
        return "\n".join(str(getattr(message, "content", message)) for message in messages)


def _quality_to_snapshot(item, default_iteration: int = 0) -> EvaluationSnapshot:
    if isinstance(item, dict):
        score = float(item.get("score", 0.0))
        return EvaluationSnapshot(
            iteration=int(item.get("iteration", default_iteration)),
            comprehensiveness_score=int(round(score)),
            accuracy_score=int(round(score)),
            coherence_score=int(round(score)),
            average_score=score,
            reason=str(item.get("feedback", "")),
        )
    return EvaluationSnapshot(iteration=default_iteration)


def extract_trajectory_from_state(state: dict) -> ResearchTrajectory:
    quality_history = state.get("quality_history", []) or []
    critiques = state.get("active_critiques", []) or []
    return ResearchTrajectory(
        run_id=state.get("run_id") or str(uuid4()),
        created_at=datetime.utcnow().isoformat(),
        user_query=_get_buffer_string(state.get("messages", [])),
        research_brief=state.get("research_brief", "") or "",
        initial_draft=state.get("initial_draft_report", "") or "",
        final_draft=state.get("draft_report", "") or "",
        final_report=state.get("final_report", "") or "",
        evaluation_history=[_quality_to_snapshot(item, idx) for idx, item in enumerate(quality_history)],
        critiques=[
            CritiqueSnapshot(
                iteration=idx,
                author=getattr(item, "author", ""),
                concern=getattr(item, "concern", ""),
                addressed=bool(getattr(item, "addressed", False)),
            )
            for idx, item in enumerate(critiques)
        ],
        notes=list(state.get("notes", []) or []),
        raw_notes=list(state.get("raw_notes", []) or []),
        retrieved_skill_ids=list(state.get("retrieved_skill_ids", []) or []),
        metadata={
            "research_iterations": state.get("research_iterations", 0),
            "critique_nums": state.get("critique_nums", 0),
        },
    )

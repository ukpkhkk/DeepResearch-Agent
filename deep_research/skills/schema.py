from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationSnapshot(BaseModel):
    iteration: int = 0
    comprehensiveness_score: int = 0
    accuracy_score: int = 0
    coherence_score: int = 0
    average_score: float = 0.0
    reason: str = ""


class CritiqueSnapshot(BaseModel):
    iteration: int = 0
    author: str = ""
    concern: str = ""
    addressed: bool = False


class ResearchTaskTrace(BaseModel):
    topic: str = ""
    compressed_research: str = ""
    raw_notes: list[str] = Field(default_factory=list)
    source_count: int | None = None


class SupervisorStep(BaseModel):
    iteration: int = 0
    decision_summary: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class RefineStep(BaseModel):
    iteration: int = 0
    before_draft: str = ""
    after_draft: str = ""
    findings_summary: str = ""


class ResearchTrajectory(BaseModel):
    run_id: str
    created_at: str
    user_query: str = ""
    research_brief: str = ""
    initial_draft: str = ""
    final_draft: str = ""
    final_report: str = ""
    supervisor_steps: list[SupervisorStep] = Field(default_factory=list)
    research_tasks: list[ResearchTaskTrace] = Field(default_factory=list)
    refine_steps: list[RefineStep] = Field(default_factory=list)
    evaluation_history: list[EvaluationSnapshot] = Field(default_factory=list)
    critiques: list[CritiqueSnapshot] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    raw_notes: list[str] = Field(default_factory=list)
    retrieved_skill_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchSkill(BaseModel):
    skill_id: str
    name: str
    created_at: str
    updated_at: str
    source_run_id: str
    language: str = "unknown"
    task_type: str = "deep_research"
    task_tags: list[str] = Field(default_factory=list)
    applicability: str = ""
    not_applicable_when: list[str] = Field(default_factory=list)
    planning_guidance: list[str] = Field(default_factory=list)
    research_guidance: list[str] = Field(default_factory=list)
    writing_guidance: list[str] = Field(default_factory=list)
    evaluation_guidance: list[str] = Field(default_factory=list)
    red_team_guidance: list[str] = Field(default_factory=list)
    report_structure_pattern: str = ""
    common_failure_modes: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillSearchResult(BaseModel):
    skill: ResearchSkill
    vector_distance: float = 0.0
    vector_score: float = 0.0
    final_score: float = 0.0

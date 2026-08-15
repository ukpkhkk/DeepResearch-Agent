import operator
from typing_extensions import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from deep_research.states.critique import Critique
from deep_research.states.quality import QualityMetric


class SupervisorState(TypedDict):
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    notes: Annotated[list[str], operator.add]
    research_iterations: int
    critique_nums: int
    raw_notes: Annotated[list[str], operator.add]
    draft_report: str
    active_critiques: Annotated[List[Critique], operator.add]
    quality_history: Annotated[List[QualityMetric], operator.add]
    needs_quality_repair: bool
    retrieved_skills: Annotated[list[str], operator.add]
    retrieved_skill_ids: Annotated[list[str], operator.add]
    refine_history: Annotated[list[dict], operator.add]
    research_task_history: Annotated[list[dict], operator.add]


@tool
class ConductResearch(BaseModel):
    """Delegate one focused research task to a specialized sub-agent."""

    research_topic: str = Field(
        description="A single focused research topic with enough detail for an independent research agent.",
    )


@tool
class ResearchComplete(BaseModel):
    """Signal that the research process is complete."""

    pass

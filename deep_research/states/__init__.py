#***********************************************
#      Filename: __init__.py
#   Description: 大模型格式化字段定义  
#***********************************************

from deep_research.states.critique import Critique
from deep_research.states.quality import QualityMetric
from deep_research.states.eval_result import EvaluationResult
from deep_research.states.draft import AgentInputState, AgentState, ResearchQuestion, DraftReport
from deep_research.states.research import ResearcherState, ResearcherOutputState, Summary
from deep_research.states.supervisor import SupervisorState, ConductResearch, ResearchComplete
from deep_research.skills.schema import ResearchSkill, ResearchTrajectory, SkillSearchResult

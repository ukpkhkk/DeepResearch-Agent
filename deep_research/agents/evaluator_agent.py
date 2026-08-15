#***********************************************
#      Filename: evaluator_agent.py
#   Description: 报告评估智能体 
#***********************************************

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from deep_research.llm import get_chat_model
from deep_research.states import EvaluationResult 
from deep_research.prompts import DRAFT_EVALUATOR_PROMPT


# 初始化Judge Model 
judge_model = get_chat_model("evaluator")


def evaluate_draft_quality(research_brief: str, draft_report: str) -> EvaluationResult:
    """
    此函数实现了“self-evolution”评分机制。用另一个LLM作为评判者，以评估报告草稿相对于原始简报的质量。
    打分的分数和原因会返回给supervisor智能体。
    """

    # 组装prompt
    eval_prompt = DRAFT_EVALUATOR_PROMPT.format(
            research_brief = research_brief,
            draft_report = draft_report
    )

    # 获取结构化的分数结果（多维度打分）
    structured_judge = judge_model.with_structured_output(EvaluationResult)

    return structured_judge.invoke([HumanMessage(content=eval_prompt)])

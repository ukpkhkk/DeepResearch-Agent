#***********************************************
#      Filename: red_team_agent.py
#   Description: Red-Team智能体 
#***********************************************

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from deep_research.prompts import RED_TEAM_PROMPT
from deep_research.llm import get_chat_model
from deep_research.states import SupervisorState, Critique
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# 初始化模型 
red_team_model = get_chat_model("red_team")

# CONSTANTS
MAX_CRITIC = 3           # 红队最大批评次数，防止追求完美，无限循环
MIN_DRAFT_LEN = 50      # 报告草稿字数最少不低于50, 否则不予评判，直接返回
MIN_CRITIC = 20          # 如果红队输出很短，则判定没有缺陷（防止模型指令遵循不足输出“PASS”以外的其他字符） 


async def red_team_node(state: SupervisorState) -> dict:
    """
    这是一个红队对抗智能体，用于找出报告的逻辑缺陷，漏洞和不完善的地方。
    """

    draft = state.get("draft_report", "")
    research_brief = state.get("research_brief", "")
    critique_nums = state.get("critique_nums", 0)

    # 设置最大对抗次数
    if critique_nums >= MAX_CRITIC or not draft or len(draft) < MIN_DRAFT_LEN:
        return {}

    # 组装prompt
    prompt = RED_TEAM_PROMPT.format(research_brief=research_brief, draft_report=draft) 

    # 调用红队大模型获得对抗建议
    response = await red_team_model.ainvoke([HumanMessage(content=prompt)])
    content = response.content

    # 如果“PASS”, 则直接返回
    if "PASS" in content or len(content) < MIN_CRITIC:
        return {}

    # 如果找到报告缺陷，返回critique
    critique = Critique(
        author="Red Team Adversary",
        concern=content,
        addressed=False
    )
    logger.info(f"[RED TEAM] {content}")

    # 返回active_critiques实现动态上下文注入，并把该意见作为System Message注入到Supervisor的消息历史中
    return {
        "active_critiques": [critique],
        "critique_nums": critique_nums + 1,
        "supervisor_messages": [
            SystemMessage(content=f"ADVERSARIAL FEEDBACK DETECTED: {content}")
        ]
    }

#***********************************************
#      Filename: draft_agent.py
#   Description: 根据用户需求生成报告草稿  
#***********************************************


"""该文件的作用是澄清用户问题和生成研究简要报
主要包含以下3个模块：
1. 用户问题澄清：确认是否有必要反问
2. 确定提纲：根据问题生成一个研究的简报
3. 报告草稿：生成一个初步的报告草稿
"""

import os
from typing_extensions import Literal
from rich.markdown import Markdown
from rich.console import Console

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research import logging as dr_logging
from deep_research.llm import get_chat_model
from deep_research.prompts import RESEARCH_BRIEF_PROMPT, DRAFT_REPORT_PROMPT  
from deep_research.states import AgentState, ResearchQuestion, AgentInputState, DraftReport
from deep_research.utils import get_today_str 
from deep_research.skills.prompting import format_skills_for_prompt

logger = dr_logging.get_logger(__name__)


# 初始化模型 
draft_model = get_chat_model("draft")


# ===== Langgraph的节点 =====

def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """根据用户的query生成一个研究提纲，内容包含需要调研哪些方面，注意事项等等"""

    logger.debug(
        "write research_brief invoked with %d messages", len(state.get("messages", []))
    )

    # 组装prompt
    prompt = RESEARCH_BRIEF_PROMPT.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    logger.debug("write_research_brief invoking structured_output_model with prompt_length=%d", len(prompt))

    # 结构化输出
    structured_output_model = draft_model.with_structured_output(ResearchQuestion)
    response = structured_output_model.invoke([HumanMessage(content=prompt)])
    logger.debug("write_research_brief produced research_brief length=%d", len(response.research_brief))

    # 更新Messages并回传给主agent 
    return Command(
            goto="write_draft_report", 
            update={"research_brief": response.research_brief}
        )

def write_draft_report(state: AgentState) -> Command[Literal["__end__"]]:
    """生成提纲生成一个研究报告的草稿"""

    logger.debug(
        "write_draft_report invoked with research_brief present=%s",
        bool(state.get("research_brief")),
    )

    # 组装prompt 
    research_brief = state.get("research_brief", "")
    draft_report_prompt = DRAFT_REPORT_PROMPT.format(
        research_brief=research_brief,
        date=get_today_str()
    )

    # 结构化输出
    structured_output_model = draft_model.with_structured_output(DraftReport)
    messages = []
    skill_context = format_skills_for_prompt(state.get("retrieved_skills", []))
    if skill_context:
        messages.append(SystemMessage(content=skill_context))
    messages.append(HumanMessage(content=draft_report_prompt))
    response = structured_output_model.invoke(messages)
    logger.debug("write_draft_report produced draft_report length=%d", len(response.draft_report))

    return {
        "research_brief": research_brief,
        "draft_report": response.draft_report, 
        "initial_draft_report": response.draft_report,
        "supervisor_messages": ["Here is the draft report: " + response.draft_report, research_brief]
    }


if __name__ == "__main__":
    # 构建Graph
    deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

    # 增加节点 
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)
    deep_researcher_builder.add_node("write_draft_report", write_draft_report)

    # 增加边
    deep_researcher_builder.add_edge(START, "write_research_brief")
    deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
    deep_researcher_builder.add_edge("write_draft_report", END)

    # 编译graph
    draft_agent = deep_researcher_builder.compile()

    # 打印graph
    print(draft_agent.get_graph().draw_ascii())

    # 测试问题
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    result = draft_agent.invoke({"messages": [HumanMessage(content="帮我写一个关于英伟达最新GPU的调研报告")]}, config=thread)

    # 输出
    console = Console()
    print("=====  Research Brief ====")
    console.print(Markdown(result["research_brief"]))
    print()

    print("=====  Draft Report ====")
    console.print(Markdown(result["draft_report"]))

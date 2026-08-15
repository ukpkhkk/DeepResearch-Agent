#***********************************************
#      Filename: research_agent.py
#   Description:  研究智能体
#***********************************************

"""Research Agent核心实现
该文件实现了一个Research Agent，它可以执行迭代式网络搜索和综合分析，以回答复杂的研究问题。
"""


from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages

from deep_research.llm import get_chat_model
from deep_research.states import ResearcherState, ResearcherOutputState
from deep_research.utils import get_today_str
from deep_research.tools import _tavily_search_tool, _think_tool
from deep_research.prompts import RESEARCH_AGENT_PROMPT, COMPRESS_RESEARCH_SYSTEM_PROMPT, COMPRESS_RESEARCH_HUMAN_PROMPT 
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# ===== CONFIGURATION =====

# 初始化tools
tools = [_tavily_search_tool, _think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# 初始化模型
model = get_chat_model("researcher_main")
model_with_tools = model.bind_tools(tools)
compress_model = get_chat_model("researcher_compressor")


# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """根据当前状态决策下一步的动作"""

    msg_count = len(state.get("researcher_messages", []))
    logger.debug("llm_call invoked with %d messages", msg_count)

    # 调用大模型
    response = model_with_tools.invoke(
        [SystemMessage(content=RESEARCH_AGENT_PROMPT)] + state["researcher_messages"]
    )

    logger.info(
        "llm_call produced response tool_calls=%s num_tool_calls=%d",
        bool(response.tool_calls),
        len(response.tool_calls or []),
    )
    return {
        "researcher_messages": [response]
    }

def tool_node(state: ResearcherState):
    """根据前一次大模型结果执行所有工具调用"""

    tool_calls = state["researcher_messages"][-1].tool_calls
    logger.info("tool_node executing %d tool calls", len(tool_calls or []))

    # 调用工具
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        logger.info("Invoking tool %s with args=%s", tool_call["name"], tool_call["args"])
        observations.append(tool.invoke(tool_call["args"]))

    # 获取工具输出
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """把研究发现压缩为高价值摘要，只保留有用信息."""

    # 组装prompt
    system_message = COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) +\
            [HumanMessage(content=COMPRESS_RESEARCH_HUMAN_PROMPT.format(research_topic=state.get("research_topic", "")))]
    logger.info("compress_research invoked with %d messages", len(messages))

    # 调用summary模型
    response = compress_model.invoke(messages)

    # 从messages和tools抽取raw notes
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    logger.debug("compress_research produced raw_notes_count=%d", len(raw_notes))
    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer."""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    decision = "tool_node" if last_message.tool_calls else "compress_research"
    logger.info("should_continue decision=%s (has_tool_calls=%s)", decision, bool(last_message.tool_calls))
    return decision


# ===== GRAPH CONSTRUCTION =====

# Build the agent
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # 返回 final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # 继续搜索获得更多结果 
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()

if __name__ == "__main__":
    print(researcher_agent.get_graph().draw_ascii())

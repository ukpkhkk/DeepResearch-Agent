from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from deep_research.agents import supervisor_agent
from deep_research.agents import write_draft_report, write_research_brief
from deep_research.llm import get_chat_model
from deep_research.prompts import FINAL_REPORT_PROMPT
from deep_research.skills import (
    extract_research_trajectory,
    generate_research_skill,
    persist_research_skills,
    retrieve_research_skills,
)
from deep_research.skills.prompting import format_skills_for_prompt
from deep_research.states import AgentInputState, AgentState
from deep_research.utils import get_today_str


writer_model = get_chat_model("writer")


async def final_report_generation(state: AgentState):
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = FINAL_REPORT_PROMPT.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
    )

    messages = []
    skill_context = format_skills_for_prompt(state.get("retrieved_skills", []))
    if skill_context:
        messages.append(SystemMessage(content=skill_context))
    messages.append(HumanMessage(content=final_report_prompt))
    final_report = await writer_model.ainvoke(messages)

    return {
        "final_report": final_report.content,
        "messages": ["最终的报告: " + final_report.content],
    }


deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("retrieve_research_skills", retrieve_research_skills)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_node("extract_research_trajectory", extract_research_trajectory)
deep_researcher_builder.add_node("generate_research_skill", generate_research_skill)
deep_researcher_builder.add_node("persist_research_skills", persist_research_skills)

deep_researcher_builder.add_edge(START, "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "retrieve_research_skills")
deep_researcher_builder.add_edge("retrieve_research_skills", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", "extract_research_trajectory")
deep_researcher_builder.add_edge("extract_research_trajectory", "generate_research_skill")
deep_researcher_builder.add_edge("generate_research_skill", "persist_research_skills")
deep_researcher_builder.add_edge("persist_research_skills", END)

agent = deep_researcher_builder.compile()

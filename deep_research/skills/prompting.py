from __future__ import annotations

from deep_research.skills.schema import ResearchSkill


SKILL_CONTEXT_HEADER = """以下历史研究 skill 仅作为流程指导使用。
不要把其中的事实、URL、统计数据或结论当作当前任务的证据。
只能借鉴其中的任务规划、检索策略、报告结构和质量检查方法。
"""


def format_skill_for_prompt(skill: ResearchSkill) -> str:
    sections = [
        f"Skill 名称：{skill.name}",
        f"适用场景：{skill.applicability}",
        "规划指导：\n- " + "\n- ".join(skill.planning_guidance),
        "研究指导：\n- " + "\n- ".join(skill.research_guidance),
        "写作指导：\n- " + "\n- ".join(skill.writing_guidance),
        "评估指导：\n- " + "\n- ".join(skill.evaluation_guidance),
        "常见失败模式：\n- " + "\n- ".join(skill.common_failure_modes),
    ]
    return "\n".join(section for section in sections if section.strip())


def format_skills_for_prompt(skills: list[str]) -> str:
    if not skills:
        return ""
    return SKILL_CONTEXT_HEADER + "\n<historical_skills>\n" + "\n\n---\n\n".join(skills) + "\n</historical_skills>"

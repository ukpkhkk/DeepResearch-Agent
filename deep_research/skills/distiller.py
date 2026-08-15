from __future__ import annotations

import re
from datetime import datetime
from uuid import uuid4

from deep_research.skills.config import get_skill_memory_config
from deep_research.skills.schema import ResearchSkill, ResearchTrajectory


def _detect_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text or "") else "en"


def _trajectory_quality(trajectory: ResearchTrajectory) -> float:
    if trajectory.evaluation_history:
        return max(item.average_score for item in trajectory.evaluation_history)
    return 8.0 if len(trajectory.final_report) > 1000 and trajectory.notes else 0.0


def trajectory_passes_quality_gate(trajectory: ResearchTrajectory, cfg: dict | None = None) -> bool:
    skill_cfg = cfg or get_skill_memory_config()
    quality = _trajectory_quality(trajectory)
    if quality < float(skill_cfg.get("min_quality_score", 8.0)):
        return False
    if skill_cfg.get("require_no_open_critiques", True):
        if any(not critique.addressed for critique in trajectory.critiques):
            return False
    return bool(trajectory.final_report and trajectory.research_brief)


def _fallback_skill(trajectory: ResearchTrajectory, quality_score: float) -> ResearchSkill:
    now = datetime.utcnow().isoformat()
    language = _detect_language(trajectory.user_query + trajectory.research_brief)
    return ResearchSkill(
        skill_id=str(uuid4()),
        name="深度研究报告生成模式",
        created_at=now,
        updated_at=now,
        source_run_id=trajectory.run_id,
        language=language,
        task_type="deep_research",
        task_tags=["deep_research", "report_generation", language],
        applicability="适用于需要基于资料来源进行规划、多轮研究和结构化报告写作的开放式调研任务。",
        not_applicable_when=["用户只需要一个简短事实答案，不需要调研。", "任务依赖工具无法访问的私有数据。"],
        planning_guidance=[
            "在开始检索前，先将用户请求转化为边界清晰的研究简报。",
            "对于宽泛任务，将问题拆解为互相独立的子课题；如果子课题之间没有依赖关系，可以并行研究。",
            "每轮研究结束后，将新发现与当前草稿对照，识别仍然缺失的分析维度。",
        ],
        research_guidance=[
            "优先使用权威来源，并在压缩笔记中保留可引用的 URL。",
            "网页原文进入报告上下文前应先做摘要，降低长上下文噪声。",
            "通过反思步骤判断当前证据是否充分，以及是否需要继续检索。",
        ],
        writing_guidance=[
            "先建立草稿结构，再根据新增研究发现持续修订，不要一次性直接生成最终报告。",
            "区分有证据支撑的事实、分析判断和建议结论。",
            "如果研究过程中获得了 URL，应在报告末尾保留来源列表。",
        ],
        evaluation_guidance=[
            "定稿前检查覆盖度、事实依据和逻辑连贯性。",
            "将评估和批判反馈转化为下一轮修订的明确修复指令。",
        ],
        red_team_guidance=[
            "重点检查遗漏视角、缺少证据的断言、内部矛盾和结构松散问题。",
        ],
        report_structure_pattern="根据用户意图组织章节，常见结构包括概述、核心发现、对比或分析、影响与建议、参考来源。",
        common_failure_modes=[
            "草稿看似完整但证据覆盖不足时过早停止研究。",
            "把历史任务中的示例或事实混入当前任务结论。",
            "只罗列来源，没有把来源与正文论断对应起来。",
        ],
        quality_score=quality_score,
        metadata={"distillation": "fallback"},
    )


def distill_skill_from_trajectory(trajectory: ResearchTrajectory) -> ResearchSkill | None:
    cfg = get_skill_memory_config()
    quality_score = _trajectory_quality(trajectory)
    if not trajectory_passes_quality_gate(trajectory, cfg):
        return None

    prompt = f"""请从这次成功的 Deep Research 运行轨迹中提炼可复用的流程指导。
不要复制本次任务中的具体事实、URL、统计数据或结论。
请输出面向后续 Agent 的简洁流程经验，重点覆盖任务拆解、检索策略、报告结构、评估标准和常见错误。

研究简报：
{trajectory.research_brief}

评估历史：
{[item.model_dump() for item in trajectory.evaluation_history]}

最终报告节选：
{trajectory.final_report[:4000]}
"""
    try:
        from langchain_core.messages import HumanMessage

        from deep_research.llm import get_chat_model

        model = get_chat_model(cfg.get("distill_model_role", "writer"))
        response = model.invoke([HumanMessage(content=prompt)])
        skill = _fallback_skill(trajectory, quality_score)
        content = getattr(response, "content", "")
        if content:
            skill.metadata["llm_distilled_notes"] = str(content)[:6000]
            skill.planning_guidance.insert(0, str(content)[:800])
        return skill
    except Exception:
        return _fallback_skill(trajectory, quality_score)

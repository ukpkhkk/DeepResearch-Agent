#***********************************************
#      Filename: eval_result.py
#   Description: 报告评估结果结构化输出 
#***********************************************

from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):

    # 综合性得分：0-10分衡量是否覆盖了简报的所有方面
    comprehensiveness_score: int = Field(description="0-10 score on coverage")

    # 准确性得分：衡量报告是否有事实依据
    accuracy_score: int = Field(description="0-10 score on factual grounding")

    # 一致性得分：衡量报告是否有逻辑缺陷，以及报告是否流畅，可读性良好
    coherence_score: int = Field(description="0-10 score on flow")

    # 打分原因，用于改善报告质量
    reason: str = Field(description="Feedback for the researcher")

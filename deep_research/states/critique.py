#***********************************************
#      Filename: critique.py
#   Description: 批评Agent的格式化输出  
#***********************************************

from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field


class Critique(BaseModel):
    """用于接收来自"Red Team " 或其他质量控制Agent的对抗性反馈的结构化模型"""

    # 用于追踪生成批评的Agent（例如，"Red Team", "Safety Filter"），以便于问责。
    author: str

    # 在报告草稿中发现的具体逻辑谬误、偏见或事实错误。
    concern: str

    # 用于追踪批评是否已在后续草稿修订中得到解决的标志
    addressed: bool = Field(default=False, description="Has the supervisor fixed this?")

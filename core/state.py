"""
状态定义模块
定义工作流中的状态结构
"""
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TripState(TypedDict):
    """差旅审批工作流的状态"""

    messages: Annotated[List[BaseMessage], add_messages]

    # 差旅申请信息
    destination: Optional[str]  # 目的地
    days: Optional[int]  # 出差天数
    budget: Optional[float]  # 预算
    job_rank: Optional[str]  # 出差人的职级或职称

    # 审批过程数据
    weather: Optional[str]  # 天气状况 (如 "Sunny", "Snow")
    temp: Optional[int]  # 温度
    policy_context: Optional[str]  # 检索到的相关政策

    # 最终结果
    final_decision: Optional[str]  # 最终建议
    status: Optional[str]  # "APPROVED" (批准) / "REJECTED" (拒绝) / "PENDING"
    
    # 数据库相关
    record_id: Optional[str]  # 数据库记录ID

    # 反思机制专用字段
    decision_feedback: Optional[str]  # 审计员给出的整改意见
    revision_count: int  # 反思次数

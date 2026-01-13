"""
工具定义模块
定义 Agent 可以使用的工具
"""
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from RAG.retriever import query_policy
from tools.weather import get_real_weather


@tool
def lookup_policy_tool(query: str):
    """
    用于查询公司差旅政策、报销标准、费用限制、天气预警规定等。
    当用户询问关于规定的问题时，必须使用此工具。
    """
    return query_policy(query)


@tool
def get_destination_weather(destination: str):
    """
    用于查询目的地天气。
    当用户询问目的地天气时，必须使用此工具。
    """
    # 直接返回天气信息的字符串描述
    weather_data = get_real_weather(destination)
    return f"天气情况: {weather_data.get('weather', '未知')}, 温度: {weather_data.get('temp', '未知')}°C"


@tool
class TripSubmission(BaseModel):
    """
    差旅申请信息表格，信息不足时不要调用，先调用其它工具或者询问用户补全信息
    当用户发出差旅申请时，必须填写此表格
    """
    destination: str = Field(..., description="目的地")
    days: int = Field(..., description="出差天数")
    budget: float = Field(..., description="预算")

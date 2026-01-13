"""
工作流主模块：差旅审批智能体
构建和配置 LangGraph 工作流
"""
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import LANGCHAIN_API_KEY
from core.state import TripState
from core.tools import lookup_policy_tool, get_destination_weather, TripSubmission
from core.nodes import (
    agent_node,
    data_sync_node,
    check_weather_node,
    compliance_check_node,
    make_decision_node,
    save_db_node,
    format_result_node,
    critique_decision_node,
    should_revise,
    router_function
)

# 配置 LangChain 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = "agent-verification"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY


# ========================================
# 构建工作流
# ========================================

def build_workflow():
    """
    构建差旅审批工作流
    
    工作流结构：
    1. Agent 节点：处理用户对话，决定是否使用工具
    2. 工具执行节点：执行 RAG 查询政策
    3. 数据同步节点：提取差旅申请信息
    4. 审批流程：天气查询 -> 政策检查 -> 决策 -> 保存 -> 格式化结果
    """
    tools = [lookup_policy_tool, get_destination_weather, TripSubmission]

    workflow = StateGraph(TripState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_executor", ToolNode(tools))
    workflow.add_node("data_sync", data_sync_node)
    workflow.add_node("check_weather", check_weather_node)
    workflow.add_node("compliance_check", compliance_check_node)
    workflow.add_node("make_decision", make_decision_node)
    workflow.add_node("critique_decision", critique_decision_node)
    workflow.add_node("save_db", save_db_node)
    workflow.add_node("format_result", format_result_node)  # 新增格式化结果节点

    # 设置入口
    workflow.set_entry_point("agent")

    # 添加条件边：Agent 的三岔路口
    workflow.add_conditional_edges(
        "agent",
        router_function,
        {
            "end": END,  # 对话结束，回到用户输入
            "run_tool": "tool_executor",  # 需要执行工具
            "start_approval": "data_sync"  # 开始审批流程
        }
    )

    # 添加普通边
    # 问答闭环：Tool 执行完 -> 回到 Agent 继续说话
    workflow.add_edge("tool_executor", "agent")

    # 审批流水线：DataSync -> Check Weather -> Compliance Check -> Make Decision
    workflow.add_edge("data_sync", "check_weather")
    workflow.add_edge("check_weather", "compliance_check")
    workflow.add_edge("compliance_check", "make_decision")
    workflow.add_edge("make_decision", "critique_decision")
    workflow.add_conditional_edges(
        "critique_decision",
        should_revise,
        {
            "revise": "make_decision",
            "pass": "save_db"
        }
    )
    workflow.add_edge("save_db", "format_result")  # 添加到格式化结果节点
    workflow.add_edge("format_result", END)  # 格式化完成后结束当前流程

    return workflow


# ========================================
# 初始化应用
# ========================================

# 构建工作流
workflow = build_workflow()

# 添加内存管理
memory = MemorySaver()

# 编译成可运行应用
app = workflow.compile(checkpointer=memory)

# 保存可视化架构图
try:
    with open('workflow.png', 'wb') as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("架构图已保存为 workflow.png")
except Exception as e:
    print(f"无法保存架构图: {e}")

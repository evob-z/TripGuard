"""
工作流节点模块
定义工作流中的各个处理节点
"""
import json
from typing import Literal

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from RAG.retriever import query_policy
from config import OPENAI_API_KEY
from core.llm import get_llm_model
from core.state import TripState
from core.tools import lookup_policy_tool, get_destination_weather, TripSubmission
from database import save_trip_record
from tools.weather import get_real_weather


def agent_node(state: TripState):
    """
    Agent 节点：核心对话处理节点
    - 处理用户输入
    - 决定是否使用工具（查询政策 / 提交申请）
    - 或直接回复用户
    """
    llm = get_llm_model()

    # 绑定工具：查政策能力 + 提交申请能力
    llm_with_tools = llm.bind_tools([lookup_policy_tool, get_destination_weather, TripSubmission])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是 TripGuard，一个专业的智能差旅合规助手。\n"
            "你可以帮助用户查询差旅政策（使用 lookup_policy_tool），查询天气（get_destination_weather），或者处理差旅申请(TripSubmission)。\n"
            "原则：\n"
            "1. 如果有可以调用工具获取的信息，先调用工具，并以工具获取的信息为准\n"
            "2. 如果缺少无法通过工具获取的关键信息（如目的地），请追问用户，不要瞎编。\n"
            "3. 在提交申请前，确保你理解了用户的意图。"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm_with_tools
    try:
        response = chain.invoke({"messages": state["messages"]})
    except Exception as e:
        # 返回一个默认的 AI 消息
        from langchain_core.messages import AIMessage
        response = AIMessage(content=f"抱歉，我在处理您的请求时遇到了技术问题: {str(e)}")

    return {"messages": [response]}


def router_function(state: TripState) -> list[str]:
    """
    路由：支持并行触发多个工作流
    """
    messages = state["messages"]
    if not messages:
        return ["end"]

    last_msg = messages[-1]

    # 检查是否有工具调用
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return ["end"]

    # 获取所有被调用的工具名称
    tool_names = [tc["name"] for tc in last_msg.tool_calls]

    destinations = []

    # --- 路由逻辑匹配 ---

    # 1. 差旅审批流触发器
    if "TripSubmission" in tool_names:
        destinations.append("start_approval")  # 对应 data_sync 节点

    # 3. 通用查询工具 (查天气、查政策)
    # 如果同时包含流程工具和普通工具，通常建议并行执行，或者让普通工具在流程内被调用
    # 这里演示并行：
    common_tools = ["lookup_policy_tool", "get_destination_weather"]
    if any(name in common_tools for name in tool_names):
        destinations.append("run_tool")

    # 如果没有匹配到任何已知流，但有工具调用，默认去通用工具节点
    if not destinations and tool_names:
        return ["run_tool"]

    return destinations if destinations else ["end"]


def data_sync_node(state: TripState):
    """
    数据同步节点：提取申请信息并生成 ToolMessage
    """
    last_msg = state["messages"][-1]

    # 1. 找到 TripSubmission 的 tool_call
    # (防御性编程：虽然路由保证了这里大概率有，但防止多工具调用时的边缘情况)
    target_tool_call = next(
        (tc for tc in last_msg.tool_calls if tc["name"] == "TripSubmission"),
        None
    )

    if not target_tool_call:
        return {}

    # 2. 提取参数
    args = target_tool_call["args"]

    # 3. 构造 ToolMessage (关键！必须回填这个消息，否则 OpenAI 会报 400 错误)
    # 告诉 LLM：“你的工具调用已经收到了，我们正在后台处理”
    tool_msg = ToolMessage(
        tool_call_id=target_tool_call["id"],
        name=target_tool_call["name"],
        content=json.dumps({"status": "received", "info": "正在进行合规检查..."})
    )

    # 4. 返回状态更新
    # 注意：不要读取 args 中的 weather/temp，防止 LLM 幻觉，后续节点会去查真实的
    return {
        "destination": args.get("destination"),
        "days": args.get("days"),
        "budget": args.get("budget"),
        # 将构造好的 tool_msg 追加到历史记录中
        "messages": [tool_msg]
    }


def check_weather_node(state: TripState):
    """
    天气检查节点：查询目的地实时天气
    确保审批决策有完整的天气信息
    """
    print("--- [Weather Check] 查询目的地天气 ---")
    
    # 从状态中获取出差目的地，确保查询的是出差地点的天气
    destination = state.get('destination', '')
    
    # 如果没有目的地，尝试从消息历史中提取TripSubmission信息
    if not destination:
        for msg in reversed(state['messages']):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call['name'] == 'TripSubmission':
                        args = tool_call.get('args', {})
                        destination = args.get('destination', '')
                        if destination:
                            break
            if destination:
                break
    
    weather_data = get_real_weather(destination)
    
    # 更新状态中的天气信息
    return {
        "weather": weather_data.get('weather', '未知'),
        "temp": weather_data.get('temp', '未知')
    }


def compliance_check_node(state: TripState):
    """
    合规检查节点：检索相关政策条款
    针对具体的【地点+天气】组合检索相关规定
    """
    print("--- [Policy Check] 检索合规条款 ---")
    query = f"{state['destination']} {state['weather']} 差旅规定"
    policy_text = query_policy(query)
    return {"policy_context": policy_text}


# 定义审批结果的数据结构
class ApprovalDecision(BaseModel):
    """审批决策结果"""
    status: Literal["APPROVED", "REJECTED"] = Field(
        ...,
        description="最终审批状态。只有在完全符合政策且天气安全时才能批准。"
    )
    reason: str = Field(
        ...,
        description="详细的决策理由。如果拒绝，请说明违规条款和修改建议；如果批准，请简述符合的原因。"
    )


def make_decision_node(state: TripState):
    """
    决策节点：基于天气和政策做出审批决定
    修复：使用 PydanticOutputParser 替代不支持的 with_structured_output
    """
    print("--- [Decision] 正在呼叫 LLM 做出决策 ---")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=OPENAI_API_KEY,
        base_url="https://api.deepseek.com"
    )

    # 2. 创建解析器
    parser = PydanticOutputParser(pydantic_object=ApprovalDecision)

    # 获取之前的反馈（如果有）
    feedback = state.get("decision_feedback")
    # 3. 在 Prompt 中注入格式说明
    prompt_text = """
    你是一个严格且专业的公司差旅合规审批官。
    你的任务是根据提供的【天气信息】、【预算金额】和【公司政策】，对【用户的差旅申请】做出最终判断。

    --- 输入信息 ---
    1. 目的地: {destination}
    2. 拟出行天数: {days} 天
    3. 预算金额: {budget} 元
    4. 当地实时天气: {weather} (气温 {temp}°C)
    5. 检索到的公司政策: {policy}

    --- 决策规则 ---
    - 安全第一：如果天气恶劣（如暴雪、极寒、台风），无论政策如何，一律拒绝。
    - 合规优先：如果天气正常，但违反了费用或级别限制（根据政策判断），需要拒绝。
    - 预算逻辑：只要申请总金额【低于或等于】政策规定的【总预算上限】，即视为合规。
    - 批准条件：只有在天气安全且符合政策时，才能批准。

    --- 输出格式要求 ---
    {format_instructions}
    """

    if feedback:
        prompt_text += f"""

        ⚠️ **重要修正指示** ⚠️
        你之前的决策被审计系统驳回，原因如下：
        "{feedback}"

        请仔细阅读上述反馈，重新检查你的计算逻辑或政策解读，并生成修正后的决策。
        """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    # 4. 这里的 prompt 需要部分格式化注入 parser 的指令
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser  # 链式调用：Prompt -> LLM -> Parser

    try:
        # 执行调用
        decision = chain.invoke({
            "destination": state.get("destination", "未知"),
            "days": state.get("days", 1),
            "budget": state.get("budget", 0),
            "weather": state.get("weather", "未知"),
            "temp": state.get("temp", 0),
            "policy": state.get("policy_context", "未检索到政策")
        })

        return {
            "final_decision": decision.reason,
            "status": decision.status
        }

    except Exception as e:
        print(f"决策解析失败: {e}")
        # 兜底：如果解析还是失败（极少情况），默认拒绝
        return {
            "final_decision": f"系统错误：模型输出无法解析。原始错误: {str(e)}",
            "status": "REJECTED"
        }


class CritiqueResult(BaseModel):
    """审计结果结构"""
    is_valid: bool = Field(..., description="决策是否逻辑正确且符合政策")
    feedback: str = Field(...,
                          description="如果正确请填'通过'，如果不正确请指出具体逻辑漏洞（例如：'预算计算错误，每日800元*3天=2400元，用户申请2000元应为合规'）")


def critique_decision_node(state: TripState):
    """
    审计节点：检查 make_decision 的结果是否不仅合规，而且逻辑自洽
    """
    print("--- [Critique] 正在审计审批结果 ---")

    llm = ChatOpenAI(model="deepseek-chat",
                     api_key=OPENAI_API_KEY,
                     base_url="https://api.deepseek.com")
    parser = PydanticOutputParser(pydantic_object=CritiqueResult)

    prompt_text = """
    你是一个质量控制审计员 (QA Auditor)。
    你的任务是审查上一步【审批官】做出的【审批决策】是否正确。

    重点检查：
    1. **计算错误**：比如把“单日预算”当成了“总预算”。
    2. **事实错误**：比如天气明明是“台风”，审批官却说“天气适宜”。
    3. **逻辑冲突**：比如理由里说“符合规定”，状态却选了“REJECTED”。

    --- 原始数据 ---
    - 申请: 去 {destination} 出差 {days} 天，总预算 {budget} 元
    - 实际天气: {weather}
    - 公司政策: {policy}

    --- 待审查的决策 ---
    - 审批状态: {status}
    - 审批理由: {final_decision}

    请判断该决策是否有效。

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "destination": state["destination"],
            "days": state["days"],
            "budget": state["budget"],
            "weather": state.get("weather"),
            "policy": state.get("policy_context"),
            "status": state.get("status"),
            "final_decision": state.get("final_decision")
        })

        if result.is_valid:
            print(">>> [Audit Pass] 审计通过")
            return {"decision_feedback": None}  # 清除反馈
        else:
            print(f">>> [Audit Fail] 审计未通过: {result.feedback}")
            # 增加修订次数，并记录反馈
            return {
                "decision_feedback": result.feedback,
                "revision_count": state.get("revision_count", 0) + 1
            }

    except Exception as e:
        print(f"审计节点出错: {e}")
        # 如果审计挂了，保守起见让它通过，或者人工介入
        return {"decision_feedback": None}


def should_revise(state: TripState):
    """
    路由逻辑：决定是重修 (Revise) 还是 通过 (Pass)
    """
    feedback = state.get("decision_feedback")
    revision_count = state.get("revision_count", 0)

    # 1. 如果没有反馈（说明审计通过），或者反馈为空
    if not feedback:
        return "pass"

    # 2. 如果有反馈，但修订次数超过限制（防止死循环，比如3次）
    # 此时强制通过，或者进入人工干预节点
    if revision_count >= 3:
        print("⚠️ 达到最大修订次数，强制结束循环")
        return "pass"

    # 3. 有反馈且没超限 -> 回去重写
    return "revise"


def save_db_node(state: TripState, config=None):
    """
    数据库保存节点：保存审批结果到 SQLite
    """
    # 从 config 中获取 thread_id (作为 session_id)
    # LangGraph 运行时会自动注入 config
    thread_id = "unknown_session"  # 默认值
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id", "unknown_session")

    # 调用数据库写入函数
    record_id = save_trip_record(
        session_id=thread_id,
        destination=state["destination"],
        days=state["days"],
        weather=state.get("weather"),
        temp=state.get("temp"),
        status=state["status"],
        final_decision=state.get("final_decision", ""),
        budget=state.get("budget"),
        # 预留字段，State 里暂时没有，传 None 即可
        cost=state.get("cost")
    )

    # 返回更新后的状态
    return {"record_id": record_id}  # 将记录ID存储到状态中


def format_result_node(state: TripState):
    """
    格式化结果节点：生成最终的用户友好回复
    """
    # 告知用户存储的审批结果
    status_emoji = "✅" if state["status"] == "APPROVED" else "❌"
    status_text = "批准" if state["status"] == "APPROVED" else "拒绝"

    result_message = f"""
        {status_emoji} **审批结果：{status_text}**
    
        📋 **申请详情**
        - 目的地：{state['destination']}
        - 天数：{state['days']} 天
        - 预算：{state.get('budget', '未指定')} 元
        - 天气：{state.get('weather', '未查询')} ({state.get('temp', '--')}°C)
        
        💡 **决策说明**
        {state.get('final_decision', '无说明')}
        
        🔖 审批单已归档 (ID: {state.get('record_id', 'N/A')})
            """.strip()

    # 将结果作为 AI 消息添加到消息历史 - 使用构造函数确保不包含任何工具调用
    ai_message = AIMessage(content=result_message)
    return {"messages": [ai_message]}  # 确保返回的是没有 tool_calls 的纯 AI 消息

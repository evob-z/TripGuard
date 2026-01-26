"""
工作流节点模块
定义工作流中的各个处理节点
"""
import json
from typing import Literal

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from pydantic import BaseModel, Field
from RAG.retriever import query_policy
from core.llm import get_llm_model
from core.state import TripState
from core.tools import lookup_policy_tool, get_destination_weather, TripSubmission
from database import save_trip_record
from tools.weather import get_real_weather


def agent_node(state: TripState):
    """
    Agent 节点：核心对话处理节点（意图识别）
    - 处理用户输入
    - 决定是否使用工具（查询政策 / 提交申请）
    - 或直接回复用户
    
    使用模型：Qwen（快速响应、中文理解强）
    """
    # 使用 Qwen 模型进行意图识别
    llm = get_llm_model(model_type="intent")

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
    # 这里并行：
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
        "job_rank": args.get("job_rank"),
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
    query = f"{state['destination']} {state['weather']} {state['job_rank']} 住宿费及交通工具标准 差旅规定"
    policy_text = query_policy(query)
    return {"policy_context": policy_text}


# 定义审批结果的数据结构
class ApprovalDecision(BaseModel):
    """审批决策结果"""
    status: Literal["APPROVED", "REJECTED"] = Field(
        ...,
        description="最终审批状态。只有在符合身份对应的差旅标准且计算无误时才能批准。"
    )
    reason: str = Field(
        ...,
        description="详细的决策理由。说明人员类别、各项补贴计算过程、交通/住宿标准核对结果。"
    )


def make_decision_node(state: TripState):
    """
    决策节点：基于政策文件进行精细化审批
    
    使用模型：DeepSeek-Reasoner（深度推理、提供推理过程）
    优势：
      - 提供完整的推理链条，决策过程透明
      - 数学计算更准确（预算核算）
      - 政策分析更深入（身份映射、标准匹配）
    """
    print("--- [Decision] 正在呼叫 DeepSeek-Reasoner 进行深度推理决策 ---")

    # 使用 DeepSeek-Reasoner 模型进行审批决策（深度推理）
    llm = get_llm_model(model_type="decision")

    parser = PydanticOutputParser(pydantic_object=ApprovalDecision)
    feedback = state.get("decision_feedback")

    # 构造 Prompt
    prompt_text = """
    你是一名经验丰富的财务合规审批专员。你的任务是根据**检索到的政策 (Policy Context)** 对差旅申请进行智能审核。
    
    --- 申请信息 ---
    1. 申请人自述职级: {job_rank}
    2. 目的地: {destination}
    3. 天数: {days} 天
    4. 总预算: {budget} 元
    5. 检索到的政策条款: {policy}
    
    --- 审核逻辑（请一步步思考）---
    1. **身份智能映射**：
       - 政策中可能不会直接出现"{job_rank}"（如"博士生"、"实习生"）。
       - 规则：学生、研究生、科研助理通常对应政策中的 **"其余人员"** 或 **"等级三"** 或 **"其他人员"**。请寻找最接近的低级别分类。
    
    2. **标准提取**：
       - 在政策文本中查找该身份在"{destination}"的 **住宿费限额**。
       - 查找 **伙食补助标准** 和 **市内交通补助标准**（通常是定额，如100元/天或80元/天）。
    
    3. **预算合理性估算**（如果用户未提供明细）：
       - 计算 **理论合规上限** = (住宿限额 × (天数-1或天数)) + ((伙食补+交通补) × 天数) + 预估往返高铁二等座费用(参考值：北京上海约1300元，省会约800-2000元)。
       - 判定：如果 {budget} <= 理论合规上限 + 10%浮动，且无明显违规项，则应视为**合规**。
       - 注意：不要因为缺少具体的"交通费发票"而拒绝申请，那是报销阶段的事。审批阶段只看预算是否在合理范围内。
    
    --- 输出要求 ---
    - **批准 (APPROVED)**：如果预算在合理范围内，且身份能对应上。理由中请明确："根据政策，{job_rank} 属于 [映射后的类别]，上海住宿限额为 [XX] 元，总预算未超标。"
    - **拒绝 (REJECTED)**：只有当预算**显著超出**理论上限（例如超标50%以上），或明确违反硬性规定（如"坐头等舱"）时才拒绝。
    
    {format_instructions}
    """

    if feedback:
        prompt_text += f"\n\n⚠️ **审计反馈(上一轮错误)**: {feedback}\n请根据审计意见修正你的计算或判定逻辑。"

    prompt = ChatPromptTemplate.from_template(prompt_text)
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # 注意：这里需要确保 state 中包含 job_rank，如果 tools.py 还没改好，这里暂时给个默认值
    chain = prompt | llm | parser

    try:
        decision = chain.invoke({
            "job_rank": state.get("job_rank", "未提供(默认为最后一类人员)"),
            "destination": state.get("destination", "未知"),
            "days": state.get("days", 1),
            "budget": state.get("budget", 0),
            "weather": state.get("weather", "未知"),
            "temp": state.get("temp", 0),
            "policy": state.get("policy_context", "")
        })

        return {
            "final_decision": decision.reason,
            "status": decision.status
        }
    except Exception as e:
        print(f"决策解析失败: {e}")
        return {
            "final_decision": f"系统错误：{str(e)}",
            "status": "REJECTED"
        }


class CritiqueResult(BaseModel):
    """审计结果结构"""
    is_valid: bool = Field(..., description="决策是否完全合规且计算精确")
    feedback: str = Field(...,
                          description="通过写'通过'。不通过请指出具体的计算错误（如：'3天伙食费应为300元而非200元'）或政策引用错误。")


def critique_decision_node(state: TripState):
    """
    审计节点：重点检查数学计算和职级匹配（反思机制）
    
    使用模型：Qwen-Max（快速批判、中文理解强）
    优势：
      - 响应速度快（3-5秒），相比 Reasoner 快50%
      - 中文语境下的逻辑分析能力强
      - 能准确发现决策中的常见错误
      - 成本相对 Reasoner 更低
    """
    print("--- [Critique] 正在使用 Qwen-Max 进行快速审计 ---")

    # 使用 Qwen-Max 模型进行审计反思（快速批判）
    llm = get_llm_model(model_type="critique")
    parser = PydanticOutputParser(pydantic_object=CritiqueResult)

    prompt_text = """
    你是一名极其严苛的财务审计员。请检查上一步的审批决策是否犯了以下错误：
    
    --- 审计重点 ---
    1. **身份映射检查**：审批人是否正确将用户的自述职级（如"{job_rank}"）映射到了政策中的标准分类？（例如：博士生应映射为"其余人员"或"等级三"，如果审批人因为"找不到博士生字样"而拒绝，这是**错误**的决策，你需要驳回并纠正他）。
    2. **标准引用检查**：检查引用的住宿限额是否是"{destination}"的标准。不要把北京的标准套用到上海。
    3. **总额逻辑检查**：如果用户只提供了总预算，审批人是否进行了合理的倒推估算？如果总预算明显偏低或合理，审批人却以"缺少明细"为由拒绝，这是**过度官僚**，请驳回并要求通过。

    --- 原始数据 ---
    - 申请人: {job_rank} 去 {destination} ({days}天), 预算 {budget}
    - 政策片段: {policy}

    --- 待审决策 ---
    - 状态: {status}
    - 理由: {final_decision}

    如果发现错误，is_valid设为False，并在feedback中说明具体问题（例如："请确认博士生属于等级三人员，并按上海标准重新审核"）。
    如果决策合理（哪怕是基于估算），is_valid设为True。

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "job_rank": state.get("job_rank", "未提供"),
            "destination": state["destination"],
            "days": state["days"],
            "budget": state["budget"],
            "policy": state.get("policy_context", ""),
            "status": state.get("status"),
            "final_decision": state.get("final_decision")
        })

        if result.is_valid:
            return {"decision_feedback": None}
        else:
            return {
                "decision_feedback": result.feedback,
                "revision_count": state.get("revision_count", 0) + 1
            }
    except Exception as e:
        print(f"审计节点出错: {e}")
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
    数据库保存节点：保存审批结果
    """
    # 从 config 中获取 thread_id (作为 session_id)
    # LangGraph 运行时会自动注入 config
    thread_id = "unknown_session"  # 默认值
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id", "unknown_session")

    # 调用数据库写入函数
    record_id = save_trip_record(
        session_id=thread_id,
        job_rank=state.get("job_rank"),
        destination=state["destination"],
        days=state["days"],
        weather=state.get("weather"),
        temp=state.get("temp"),
        status=state["status"],
        final_decision=state.get("final_decision", ""),
        budget=state.get("budget"),
        # 预留字段，State 里暂时没有，传 None 即可
        # cost=state.get("cost")
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
        - 申请人职级：{state['job_rank']}
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

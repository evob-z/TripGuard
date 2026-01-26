from langchain_openai import ChatOpenAI
from typing import Literal

from config import DEEPSEEK_API_KEY
from config import QWEN_API_KEY


def get_llm_model(
    model_type: Literal["intent", "decision", "critique"] = "intent"
):
    """
    根据节点类型获取最适合的 LLM 模型
    
    参数:
        model_type: 节点类型
            - "intent": 意图识别节点 (Agent) -> 使用 Qwen-Plus (快速响应)
            - "decision": 审批决策节点 -> 使用 DeepSeek-Reasoner (深度推理)
            - "critique": 反思审计节点 -> 使用 Qwen-Max (快速批判)
    
    模型选择策略 - 方案B (平衡性能与成本):
        - Qwen-Plus: 适合快速交互、意图理解、工具调用
        - DeepSeek-Reasoner: 适合关键决策、复杂推理、深度分析
          * 提供完整的推理过程（reasoning_content）
          * 数学计算更准确
          * 政策分析更深入
        - Qwen-Max: 适合快速审计、批判性检查
          * 响应速度快（3-5秒）
          * 中文理解强，能准确发现中文表述中的逻辑问题
          * 成本相对 Reasoner 更低
    """
    if model_type == "intent":
        # 意图识别：使用 Qwen-Plus（响应快、中文理解好）
        llm = ChatOpenAI(
            model="qwen-plus",  # 可选: qwen-plus, qwen-max, qwen-turbo
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7  # 适度创造性，理解用户意图
        )
    elif model_type == "decision":
        # 决策：使用 DeepSeek-Reasoner（深度推理、关键决策）
        # ⚠️ 注意：reasoner 模型响应时间较长（5-10秒），但推理质量最高
        llm = ChatOpenAI(
            model="deepseek-reasoner",  # 思考模型，提供推理过程
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=1.0  # Reasoner 模型推荐使用 temperature=1.0
        )
    elif model_type == "critique":
        # 审计：使用 Qwen-Max（快速批判、成本优化）
        llm = ChatOpenAI(
            model="qwen-max",  # 最强 Qwen 模型，推理能力接近 Reasoner
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3  # 低温度，保证审计标准一致性
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return llm

from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY


def get_llm_model():
    """
    获取 LLM 模型
    通用函数，可在多个节点中使用
    """
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=OPENAI_API_KEY,
        base_url="https://api.deepseek.com"
    )
    return llm

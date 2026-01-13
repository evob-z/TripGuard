import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Rerank 相关组件
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 使用绝对路径，基于当前文件所在目录定位数据库路径
# 无论从哪里调用，都能正确找到数据库
CURRENT_FILE_DIR = Path(__file__).parent.resolve()
PERSIST_DIRECTORY = CURRENT_FILE_DIR / "md" / "chroma_db"

# def get_retriever():
#     """
#     直接加载已存在的向量数据库
#     """
#     if not os.path.exists(PERSIST_DIRECTORY):
#         raise FileNotFoundError(f"数据库 {PERSIST_DIRECTORY} 不存在，请先运行 python -m app.rag.build 构建数据库。")
#
#     # 创建一个嵌入（Embedding）模型实例，用于将用户的查询问题转换为向量表示
#     embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")  # 如果构建时用了这个
#
#     # 加载数据库
#     vector_db = Chroma(
#         persist_directory=PERSIST_DIRECTORY,
#         embedding_function=embedding_model
#     )
#
#     # 转换为检索器 (k=3 表示每次找最相似的3条)
#     return vector_db.as_retriever(search_kwargs={"k": 3})

# 全局变量缓存，防止每次调用都重新加载模型
_cached_retriever = None


def get_advanced_retriever():
    """
    构建【检索 + 重排序】的高级检索管道
    """
    global _cached_retriever
    if _cached_retriever:
        return _cached_retriever

    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"请先构建数据库。\n"
            f"期望的数据库路径: {PERSIST_DIRECTORY}\n"
            f"请运行: python -m RAG.build 或 cd RAG && python build.py"
        )

    # 1. 基础向量检索器 (Base Retriever) - 负责"粗召回"
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")  # 如果构建时用了这个
    vector_db = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),  # Chroma需要字符串路径
        embedding_function=embedding_model
    )
    base_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}  # 粗召回前 20 条
    )

    # 2. Reranker (重排器) - 负责“精排序”
    # print(">>> [RAG] 正在加载 BGE-Reranker 模型 (首次运行需下载)...")

    # 使用智源的 bge-reranker-base (效果好且体积适中)
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        model_kwargs={
            # "device": "cpu",
            "tokenizer_kwargs": {"use_fast": False}
        }
    )

    # 配置 Reranker: 从 20 条里挑出最准的 3 条
    reranker = CrossEncoderReranker(model=model, top_n=3)

    # 3. 组装管道
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    _cached_retriever = compression_retriever
    # print(">>> [RAG] Reranker 加载完成，高级检索管道就绪。")
    return compression_retriever


# 延迟初始化全局单例，避免导入时立即执行
_retriever_instance = None


def get_retriever():
    """获取检索器单例（懒加载）"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = get_advanced_retriever()
    return _retriever_instance


def query_policy(query: str) -> str:
    """
    对外暴露的查询函数
    """

    # 这一步会自动执行：向量检索 -> Rerank打分 -> 截取Top3
    retriever = get_retriever()  # 使用懒加载方式获取检索器
    docs = retriever.invoke(query)

    if not docs:
        return "未找到相关政策。"

    # 打印日志看看 Rerank 选了什么 (调试用)
    # print(f"--- [RAG Debug] Rerank 选出的最佳片段: {[d.page_content[:20] + '...' for d in docs]} ---")

    return "\n\n".join([doc.page_content for doc in docs])

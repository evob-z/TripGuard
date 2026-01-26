import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --- 基础依赖 ---
os.environ['HF_HUB_OFFLINE'] = '1'
# --- 路径配置 ---
CURRENT_FILE_DIR = Path(__file__).parent.resolve()
PERSIST_DIRECTORY = CURRENT_FILE_DIR / "data" / "chroma_db"


# --- 全局模型缓存 (单例模式) ---
_EMBEDDING_MODEL = None
_VECTOR_DB = None
_RERANKER = None

# --- 配置参数 ---
RERANK_SCORE_THRESHOLD = 0.5  # 重排序评分阈值，低于此分数的文档不会返回


def get_embedding_model():
    """懒加载 Embedding 模型"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("   [System] 正在初始化 Embedding 模型 (BAAI/bge-m3)...")
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _EMBEDDING_MODEL


def get_vector_db():
    """懒加载 向量数据库"""
    global _VECTOR_DB
    if _VECTOR_DB is None:
        if not PERSIST_DIRECTORY.exists():
            raise FileNotFoundError(f"数据库未找到: {PERSIST_DIRECTORY}")
        
        print("   [System] 正在连接向量数据库...")
        _VECTOR_DB = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=get_embedding_model(),
            collection_name="trip_guard_collection"
        )
    return _VECTOR_DB


def get_reranker():
    """懒加载 Rerank 模型"""
    global _RERANKER
    if _RERANKER is None:
        print("   [System] 正在初始化 Rerank 模型 (BAAI/bge-reranker-base)...")
        _RERANKER = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    return _RERANKER


def vector_search(query: str, vector_db: Chroma, k: int = 10):
    """
    向量检索 (Vector Search) - 语义召回
    """
    print("   - 执行向量检索...")
    return vector_db.similarity_search(query, k=k)


def bm25_search(query: str, vector_db: Chroma, k: int = 10):
    """
    BM25 检索 (Keyword Search) - 关键词召回
    """
    print("   - 执行关键词检索...")
    try:
        # 获取所有文档用于构建索引
        db_data = vector_db.get()
        all_docs = db_data['documents']
        all_metadatas = db_data['metadatas']

        if not all_docs:
            print("   ⚠️ 警告: 数据库为空，跳过 BM25")
            return []
        
        bm25_docs = [Document(page_content=t, metadata=m) for t, m in zip(all_docs, all_metadatas)]
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = k
        return bm25_retriever.invoke(query)
    except Exception as e:
        print(f"   ⚠️ BM25 构建失败(可能是第一次运行或依赖缺失): {e}")
        return []


def ensemble_results(vector_docs: list, keyword_docs: list):
    """
    手动去重合并 (Ensemble Logic)
    """
    unique_docs = {}
    # 先放入向量结果，再放入关键词结果
    for doc in vector_docs + keyword_docs:
        # 使用内容作为去重键 (防止同一段话被重复召回)
        key = doc.page_content.strip()
        if key not in unique_docs:
            unique_docs[key] = doc

    merged_docs = list(unique_docs.values())
    print(f"   - 召回合并后文档数: {len(merged_docs)}")
    return merged_docs


def rerank_documents(query: str, merged_docs: list, top_k: int = 3, score_threshold: float = 0.5):
    """
    手动重排序 (Rerank Logic)
    只返回重排序得分 >= score_threshold 的文档
    """
    if not merged_docs:
        return []

    print("   - 执行重排序 (Rerank)...")
    # 使用单例模式获取模型
    reranker = get_reranker()

    # 构造 Pair: [query, doc_content]
    pairs = [(query, doc.page_content) for doc in merged_docs]

    scores = reranker.score(pairs)

    # 将分数绑定到文档并排序
    scored_docs = sorted(
        zip(merged_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 取 Top K，并根据阈值过滤，将得分存储到文档metadata中
    result_docs = []
    for doc, score in scored_docs[:top_k]:
        # 只保留得分 >= score_threshold 的文档
        if score >= score_threshold:
            # 创建新文档副本，添加重排序得分
            new_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            )
            new_doc.metadata['rerank_score'] = float(score)
            result_docs.append(new_doc)
    
    print(f"   - 过滤后符合阈值(>={score_threshold})的文档数: {len(result_docs)}")
    return result_docs


def get_manual_hybrid_results(query: str):
    """
    手动执行：向量检索 + BM25检索 -> 简单合并 -> Rerank
    """
    print(f"🔍 开始执行混合检索: {query}")

    # 1. 获取模型和数据库 (懒加载/单例)
    vector_db = get_vector_db()

    # 2. 执行检索
    vector_docs = vector_search(query, vector_db, k=10)
    keyword_docs = bm25_search(query, vector_db, k=10)

    # 3. 合并去重
    merged_docs = ensemble_results(vector_docs, keyword_docs)

    # 4. 重排序
    return rerank_documents(query, merged_docs, top_k=3, score_threshold=RERANK_SCORE_THRESHOLD)


def query_policy(query: str) -> str:
    """
    对外接口
    """
    try:
        # 使用手动混合检索
        docs = get_manual_hybrid_results(query)

        if not docs:
            return "未找到相关政策信息。"

        results = []
        for i, doc in enumerate(docs):
            # 获取 source (build.py 中我们只存了文件名)
            source = doc.metadata.get("source", "未知文件")
            content = doc.page_content.strip()

            entry = f"【参考资料 {i + 1}】\n来源: {source}\n内容: {content}"
            results.append(entry)

        return "\n\n".join(results)

    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整报错堆栈
        return f"检索系统错误: {str(e)}"


if __name__ == "__main__":
    print("-" * 30)
    print("🚀 测试第一次查询（触发模型加载）:")
    print(query_policy("差旅住宿标准"))
    print("-" * 30)
    print("🚀 测试第二次查询（应使用缓存模型，速度显著提升）:")
    print(query_policy("交通补贴标准"))

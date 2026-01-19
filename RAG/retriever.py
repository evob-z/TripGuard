import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --- 基础依赖 ---
# 既然自动的 EnsembleRetriever 总是报错，我们这里手动实现“检索+去重+重排”的逻辑
# 这需要安装: pip install rank_bm25 sentence-transformers
os.environ['HF_HUB_OFFLINE'] = '1'
# --- 路径配置 ---
CURRENT_FILE_DIR = Path(__file__).parent.resolve()
PERSIST_DIRECTORY = CURRENT_FILE_DIR / "data" / "chroma_db"


def get_manual_hybrid_results(query: str):
    """
    手动执行：向量检索 + BM25检索 -> 简单合并 -> Rerank
    """
    print(f"🔍 开始执行混合检索: {query}")

    # 1. 初始化 Embedding 模型 (CPU)
    # 这一步如果不加 model_kwargs={"device": "cpu"}，在无显卡机器上可能会报错
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 向量检索 (Vector Search) - 语义召回
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(f"数据库未找到: {PERSIST_DIRECTORY}")

    vector_db = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embedding_model,
        collection_name="trip_guard_collection"
    )
    # 获取 Top 10
    print("   - 执行向量检索...")
    vector_docs = vector_db.similarity_search(query, k=10)

    # 3. BM25 检索 (Keyword Search) - 关键词召回
    print("   - 执行关键词检索...")
    try:
        # 获取所有文档用于构建索引
        all_docs = vector_db.get()['documents']
        all_metadatas = vector_db.get()['metadatas']

        if not all_docs:
            print("   ⚠️ 警告: 数据库为空，跳过 BM25")
            keyword_docs = []
        else:
            bm25_docs = [Document(page_content=t, metadata=m) for t, m in zip(all_docs, all_metadatas)]
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = 10
            keyword_docs = bm25_retriever.invoke(query)
    except Exception as e:
        print(f"   ⚠️ BM25 构建失败(可能是第一次运行或依赖缺失): {e}")
        keyword_docs = []

    # 4. 手动去重合并 (Ensemble Logic)
    unique_docs = {}
    # 先放入向量结果，再放入关键词结果
    for doc in vector_docs + keyword_docs:
        # 使用内容作为去重键 (防止同一段话被重复召回)
        key = doc.page_content.strip()
        if key not in unique_docs:
            unique_docs[key] = doc

    merged_docs = list(unique_docs.values())
    print(f"   - 召回合并后文档数: {len(merged_docs)}")

    if not merged_docs:
        return []

    # 5. 手动重排序 (Rerank Logic)
    print("   - 执行重排序 (Rerank)...")
    # 初始化打分模型
    reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    # 构造 Pair: [query, doc_content]
    pairs = [(query, doc.page_content) for doc in merged_docs]

    # 【核心修复】使用 .score() 而不是 .model.predict()
    scores = reranker.score(pairs)

    # 将分数绑定到文档并排序
    scored_docs = sorted(
        zip(merged_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 取 Top 3 (且分数不能太低，比如大于 -2)
    final_top_3 = []
    for doc, score in scored_docs[:3]:
        # print(f"      > 得分: {score:.4f} | 内容: {doc.page_content[:20]}...")
        final_top_3.append(doc)

    return final_top_3


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
    print(query_policy("差旅住宿标准"))

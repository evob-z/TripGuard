"""
TripGuard RAG系统 - Hard Negatives 深度诊断脚本
用于打印无答案查询的详细检索结果、Rerank分数和LLM裁判的完整思维链
"""
import sys
from pathlib import Path
import json

# 导入原有模块
from test_ragas import load_test_data, retrieval_mode_4_hybrid_with_rerank, get_llm_judge


def debug_hard_negatives():
    print("=" * 60)
    print("🕵️ TripGuard Hard Negatives 侦探模式")
    print("=" * 60)

    # 1. 加载数据并筛选出无答案查询
    data_path = Path(__file__).parent / "test_data.json"
    all_data = load_test_data(data_path)
    unanswerable_queries = [item for item in all_data if not item['ground_truth_context']]

    print(f"找到 {len(unanswerable_queries)} 个无答案查询，开始深度扫描...\n")

    # 2. 初始化 LLM
    llm = get_llm_judge()

    for idx, item in enumerate(unanswerable_queries, 1):
        query = item['question']
        print(f"🔍 Case {idx}: {query}")

        # 3. 执行检索 (Top-1 即可，因为我们要看最强的那个干扰项)
        # 注意：这里我们临时设阈值为 -1.0，确保拿到结果，不被原本的阈值逻辑拦截
        try:
            results = retrieval_mode_4_hybrid_with_rerank(query, k=10, top_k=1, score_threshold=-1.0)
        except Exception as e:
            print(f"   ❌ 检索出错: {e}")
            continue

        if not results:
            print("   ✅ 空结果 (已被检索器底层逻辑拒识)")
            continue

        top_doc = results[0]
        content = top_doc.page_content
        # 获取 rerank 分数 (假设存储在 metadata 中，根据你的代码逻辑调整 key)
        score = top_doc.metadata.get('rerank_score', top_doc.metadata.get('score', 'N/A'))

        print(f"   📉 Top-1 Rerank Score: {score}")
        print(f"   📄 Top-1 Content (前100字): {content[:100].replace(chr(10), ' ')}...")

        # 4. 运行 LLM 裁判（使用思维链 Prompt）
        debug_prompt = f"""
你是一个严格的逻辑合规审核员。请分析【检索内容】是否能回答【问题】。

【问题】：{query}
【检索内容】：{content}

请一步步思考：
1. 用户的核心约束是什么（如：经费类型、时间长短、地点等）？
2. 检索内容是否明确支持该约束？
3. 如果检索内容只涉及相似概念但不完全匹配（例如“科研经费”vs“行政经费”），请判定为“不相关”。

请输出你的分析过程，并最后明确结论“是”或“否”。
"""
        print("   🧠 LLM 裁判思考中...")
        response = llm.invoke(debug_prompt)
        print(f"   💬 裁判回答:\n{'-' * 20}\n{response.content.strip()}\n{'-' * 20}\n")


if __name__ == "__main__":
    debug_hard_negatives()

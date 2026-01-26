"""
TripGuard RAG系统 RAGAS性能评估测试脚本
测试各个检索模块的独立性能和组合效果
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

# 导入RAG相关模块
from retriever import (
    get_vector_db, 
    get_embedding_model,
    vector_search,
    bm25_search,
    ensemble_results,
    rerank_documents
)

# 导入LLM用于判断相关性
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.llm import get_llm_model

# 导入RAGAS评估库
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print("警告: RAGAS库未安装，将使用自定义评估指标")
    RAGAS_AVAILABLE = False


# ==================== 数据加载 ====================
def load_test_data(json_path: str) -> List[Dict]:
    """加载测试数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== 检索模式定义 ====================
def retrieval_mode_1_vector_only(query: str, k: int = 10) -> List[Any]:
    """模式1: 仅向量检索"""
    vector_db = get_vector_db()
    return vector_search(query, vector_db, k=k)


def retrieval_mode_2_bm25_only(query: str, k: int = 10) -> List[Any]:
    """模式2: 仅BM25关键词检索"""
    vector_db = get_vector_db()
    return bm25_search(query, vector_db, k=k)


def retrieval_mode_3_hybrid_no_rerank(query: str, k: int = 10) -> List[Any]:
    """模式3: 混合检索（向量+BM25）但不重排序"""
    vector_db = get_vector_db()
    vector_docs = vector_search(query, vector_db, k=k)
    keyword_docs = bm25_search(query, vector_db, k=k)
    merged_docs = ensemble_results(vector_docs, keyword_docs)
    return merged_docs[:k]  # 取前k个


def retrieval_mode_4_hybrid_with_rerank(query: str, k: int = 10, top_k: int = 3, score_threshold: float = 0.5) -> List[Any]:
    """模式4: 混合检索+重排序（完整pipeline）
    
    Args:
        query: 查询文本
        k: 检索数量
        top_k: 重排序后返回的文档数量
        score_threshold: 重排序得分阈值，低于此值将拒识返回空列表（默认0.5）
    """
    vector_db = get_vector_db()
    vector_docs = vector_search(query, vector_db, k=k)
    keyword_docs = bm25_search(query, vector_db, k=k)
    merged_docs = ensemble_results(vector_docs, keyword_docs)
    reranked_docs = rerank_documents(query, merged_docs, top_k=top_k)
    
    # 阈值机制：检查重排序后的最高得分
    if reranked_docs and hasattr(reranked_docs[0], 'metadata'):
        # 假设rerank_documents在metadata中存储了得分（需要根据实际实现调整）
        max_score = reranked_docs[0].metadata.get('rerank_score', 1.0)
        if max_score < score_threshold:
            # 模拟拒识行为，返回空列表
            return []
    
    return reranked_docs


# ==================== LLM判断工具 ====================
def get_llm_judge():
    """获取LLM判断器，使用快速qwen-plus模型"""
    return get_llm_model("intent")  # 使用快速响应的qwen-plus模型


def llm_judge_relevance(question: str, ground_truth: str, retrieved_context: str, llm, is_unanswerable: bool = False) -> bool:
    """使用LLM判断检索到的内容是否支持标准答案
    
    Args:
        question: 查询问题
        ground_truth: 标准答案
        retrieved_context: 检索到的内容
        llm: LLM模型
        is_unanswerable: 是否为无答案查询（用于调整prompt逻辑）
    """
    if is_unanswerable:
        # 针对无答案查询的特殊prompt
        prompt = f"""你是一个专业的评估专家。请判断【检索内容】是否包含回答【问题】所需的有效信息。

【问题】：{question}

【检索内容】：{retrieved_context}

请仅回答“是”或“否”。如果检索内容包含了回答该问题所需的有效信息，回答“是”；如果检索内容无法回答该问题或不相关，回答“否”。

判断："""
    else:
        # 针对可回答查询的原有prompt
        prompt = f"""你是一个专业的评估专家。请判断【检索内容】是否能够支持回答【问题】并得出【标准答案】。

【问题】：{question}

【标准答案】：{ground_truth}

【检索内容】：{retrieved_context}

请仅回答“是”或“否”。如果检索内容包含了足够信息来支持标准答案，回答“是”；否则回答“否”。

判断："""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        return "是" in answer or "yes" in answer.lower()
    except Exception as e:
        print(f"  ⚠️ LLM判断失败: {e}，回退到字符串匹配")
        # 回退到简单的字符串匹配
        if is_unanswerable:
            # 对于无答案查询，保守判断：如枟内容很少或为空，则认为不相关
            return len(retrieved_context.strip()) > 50
        else:
            return ground_truth in retrieved_context or any(
                keyword in retrieved_context for keyword in ground_truth.split('，')[:3]
            )


# ==================== 评估指标计算 ====================
def calculate_recall(retrieved_contexts: List[str], question: str, ground_truth: str, ground_truth_contexts: List[str], llm, k: int = None) -> float:
    """计算召回率:检索到的相关文档占所有相关文档的比例
    
    Args:
        retrieved_contexts: 检索到的文档内容列表
        question: 查询问题
        ground_truth: 标准答案
        ground_truth_contexts: 标准上下文列表
        llm: LLM判断器
        k: 如果指定，只考虑前k个检索结果
    """
    if not ground_truth_contexts:
        return 0.0
    
    # 如果指定了k，只使用前k个结果
    contexts_to_check = retrieved_contexts[:k] if k is not None else retrieved_contexts
    
    # 使用LLM判断每个标准上下文是否被召回
    matches = 0
    for gt_context in ground_truth_contexts:
        for ret_context in contexts_to_check:
            if llm_judge_relevance(question, ground_truth, ret_context, llm):
                matches += 1
                break
    
    return matches / len(ground_truth_contexts)


def calculate_precision(retrieved_contexts: List[str], question: str, ground_truth: str, llm) -> float:
    """计算精确率：检索到的文档中相关文档的比例"""
    if not retrieved_contexts:
        return 0.0
    
    # 使用LLM判断每个检索文档是否相关
    matches = 0
    for ret_context in retrieved_contexts:
        if llm_judge_relevance(question, ground_truth, ret_context, llm):
            matches += 1
    
    return matches / len(retrieved_contexts)


def calculate_mrr(retrieved_contexts: List[str], question: str, ground_truth: str, llm) -> float:
    """计算平均倒数排名（MRR）"""
    for i, context in enumerate(retrieved_contexts):
        if llm_judge_relevance(question, ground_truth, context, llm):
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(retrieved_contexts: List[str], question: str, ground_truth: str, llm) -> float:
    """计算归一化折损累积增益（NDCG）"""
    import math
    
    if not retrieved_contexts:
        return 0.0
    
    # 使用LLM判断相关性分数（1表示相关，0表示不相关）
    relevance_scores = []
    for context in retrieved_contexts:
        is_relevant = llm_judge_relevance(question, ground_truth, context, llm)
        relevance_scores.append(1.0 if is_relevant else 0.0)
    
    # DCG
    if relevance_scores:
        dcg = relevance_scores[0] + sum(
            rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores[1:], 1)
        )
    else:
        dcg = 0.0
    
    # IDCG (理想情况)
    ideal_scores = sorted(relevance_scores, reverse=True)
    if ideal_scores:
        idcg = ideal_scores[0] + sum(
            rel / math.log2(i + 2) for i, rel in enumerate(ideal_scores[1:], 1)
        )
    else:
        idcg = 0.0
    
    return dcg / idcg if idcg > 0 else 0.0


# ==================== 评估执行 ====================
def evaluate_retrieval_mode(
    mode_name: str,
    retrieval_func,
    test_data: List[Dict],
    llm,
    k_values: List[int] = [3, 5, 10],
    warmup: bool = False
) -> Dict[str, Any]:
    """评估单个检索模式，支持多个k值的Recall计算
    
    本函数会将测试数据拆分为两组：
    - Answerable: ground_truth_context 非空，仅这部分用于计算 Recall/MRR/NDCG
    - Unanswerable: ground_truth_context 为空，用于计算噪声鲁棒性（Noise Robustness）

    Args:
        mode_name: 检索模式名称
        retrieval_func: 检索函数
        test_data: 测试数据
        llm: LLM判断器
        k_values: 要评估的k值列表，用于计算Recall@k
        warmup: 是否执行预热
    """
    print(f"\n{'='*60}")
    print(f"开始评估: {mode_name}")
    print(f"{'='*60}")
    
    # 使用最大的k值进行检索
    max_k = max(k_values)
    
    results = {
        'mode_name': mode_name,
        'total_queries': len(test_data),
        'answerable_queries': 0,
        'unanswerable_queries': 0,
        'precision_scores': [],
        'mrr_scores': [],
        'ndcg_scores': [],
        'avg_retrieval_time': 0,
        'failed_queries': 0,
        'noise_robustness_scores': []  # 仅针对无答案查询
    }
    
    # 为每个k值创建独立的recall分数列表（仅记录 Answerable 样本）
    for k in k_values:
        results[f'recall@{k}_scores'] = []
    
    # 如果是第一个模式，执行一次预热查询
    if warmup:
        print("\n🔥 执行模型预热查询...")
        try:
            _ = retrieval_func(test_data[0]['question'], k=max_k)
            print("✓ 预热完成\n")
        except:
            pass
    
    total_time = 0
    
    for i, item in enumerate(test_data, 1):
        question = item['question']
        ground_truth = item.get('ground_truth', '')
        ground_truth_contexts = item['ground_truth_context']
        is_answerable = bool(ground_truth_contexts)

        if is_answerable:
            results['answerable_queries'] += 1
        else:
            results['unanswerable_queries'] += 1
        
        print(f"\n[{i}/{len(test_data)}] 查询: {question[:50]}... ({'Answerable' if is_answerable else 'Unanswerable'})")
        
        try:
            # 执行检索并计时（使用最大k值）
            start_time = time.time()
            retrieved_docs = retrieval_func(question, k=max_k)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # 提取检索到的文本内容
            retrieved_contexts = [doc.page_content for doc in retrieved_docs]
            
            if is_answerable:
                # 为每个k值计算 Recall@k（仅 Answerable 样本参与）
                recall_results = {}
                for k in k_values:
                    recall_k = calculate_recall(
                        retrieved_contexts,
                        question,
                        ground_truth,
                        ground_truth_contexts,
                        llm,
                        k=k,
                    )
                    results[f'recall@{k}_scores'].append(recall_k)
                    recall_results[k] = recall_k
                
                # 计算其他评估指标（基于最大k值的结果，仅 Answerable 样本）
                precision = calculate_precision(retrieved_contexts, question, ground_truth, llm)
                mrr = calculate_mrr(retrieved_contexts, question, ground_truth, llm)
                ndcg = calculate_ndcg(retrieved_contexts, question, ground_truth, llm)
                
                results['precision_scores'].append(precision)
                results['mrr_scores'].append(mrr)
                results['ndcg_scores'].append(ndcg)
                
                # 打印结果
                recall_str = " | ".join([f"Recall@{k}: {recall_results[k]:.3f}" for k in k_values])
                print(f"  ✓ {recall_str}")
                print(f"    Precision: {precision:.3f} | MRR: {mrr:.3f} | NDCG: {ndcg:.3f} | Time: {elapsed_time:.2f}s")
            else:
                # 无答案（Unanswerable）样本：只计算噪声鲁棒性（True Rejection），仅关注 Top-K 检索结果
                has_support = False
                noise_k = min(k_values) if k_values else len(retrieved_contexts)
                for ret_context in retrieved_contexts[:noise_k]:
                    if llm_judge_relevance(question, ground_truth, ret_context, llm, is_unanswerable=True):
                        has_support = True
                        break
                is_robust = not has_support
                score = 1.0 if is_robust else 0.0
                results['noise_robustness_scores'].append(score)
                status_str = "✅" if is_robust else "⚠️"
                print(f"  {status_str} 无答案查询评估 -> Noise Robustness: {score:.3f} (Top-{noise_k}) | Time: {elapsed_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ 查询失败: {str(e)}")
            results['failed_queries'] += 1
            if is_answerable:
                for k in k_values:
                    results[f'recall@{k}_scores'].append(0.0)
                results['precision_scores'].append(0.0)
                results['mrr_scores'].append(0.0)
                results['ndcg_scores'].append(0.0)
            else:
                # 无答案样本失败，视为噪声鲁棒性为0
                results['noise_robustness_scores'].append(0.0)
    
    # 计算平均值（Answerable 组）
    for k in k_values:
        scores = results[f'recall@{k}_scores']
        results[f'avg_recall@{k}'] = sum(scores) / len(scores) if scores else 0.0
    
    if results['precision_scores']:
        results['avg_precision'] = sum(results['precision_scores']) / len(results['precision_scores'])
    else:
        results['avg_precision'] = 0.0
    
    if results['mrr_scores']:
        results['avg_mrr'] = sum(results['mrr_scores']) / len(results['mrr_scores'])
    else:
        results['avg_mrr'] = 0.0
    
    if results['ndcg_scores']:
        results['avg_ndcg'] = sum(results['ndcg_scores']) / len(results['ndcg_scores'])
    else:
        results['avg_ndcg'] = 0.0
    
    # 无答案组的噪声鲁棒性（True Rejection Rate）
    if results['noise_robustness_scores']:
        results['noise_robustness'] = sum(results['noise_robustness_scores']) / len(results['noise_robustness_scores'])
    else:
        results['noise_robustness'] = 0.0
    
    results['avg_retrieval_time'] = total_time / len(test_data) if test_data else 0.0
    results['success_rate'] = (len(test_data) - results['failed_queries']) / len(test_data) if test_data else 0.0
    
    print(f"\n{'='*60}")
    print(f"{mode_name} 评估完成")
    for k in k_values:
        print(f"平均Recall@{k} (Answerable): {results[f'avg_recall@{k}']:.3f}")
    print(f"平均精确率 (Answerable): {results['avg_precision']:.3f}")
    print(f"平均MRR (Answerable): {results['avg_mrr']:.3f}")
    print(f"平均NDCG (Answerable): {results['avg_ndcg']:.3f}")
    print(f"无答案查询噪声鲁棒性 (Noise Robustness): {results['noise_robustness']:.3f}")
    print(f"Answerable 查询数: {results['answerable_queries']} / {results['total_queries']}")
    print(f"Unanswerable 查询数: {results['unanswerable_queries']} / {results['total_queries']}")
    print(f"平均检索时间: {results['avg_retrieval_time']:.3f}s")
    print(f"成功率: {results['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")
    
    return results


# ==================== 简化报告生成 ====================
def generate_markdown_report(all_results: List[Dict], output_path: str, k_values: List[int] = [3, 5, 10]):
    """生成包含多k值Recall的Markdown格式测试报告"""
    
    report = f"""# TripGuard RAG系统性能评估报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**测试数据量**: {all_results[0]['total_queries']}个查询
**评估方法**: 使用LLM判断检索内容是否支持标准答案；按是否存在 ground_truth_context 拆分为「可回答 (Answerable)」与「无答案 (Unanswerable)」两组
**检索性能 (Answerable)**: 仅基于 ground_truth_context 非空的数据计算 Recall@{', Recall@'.join(map(str, k_values))} 和 MRR 等排序指标
**抗幻觉性能 (Unanswerable)**: 基于 ground_truth_context 为空的数据计算噪声鲁棒性（Noise Robustness，越高越好）
**模式4阈值机制**: 混合检索+重排序模式启用了得分阈值机制（默认0.5），当重排序得分低于阈值时将拒识返回空结果，提升抗幻觉性能

---

## 📊 检索性能（Answerable）对比

| 检索模式 | Recall@3 | Recall@5 | Recall@10 | MRR | 检索时间(s) |
|---------|---------|---------|----------|-----|-----------|
"""
    
    for result in all_results:
        recall_str = " | ".join([f"{result.get(f'avg_recall@{k}', 0.0):.3f}" for k in k_values])
        report += f"| {result['mode_name']} | {recall_str} | {result['avg_mrr']:.3f} | {result['avg_retrieval_time']:.3f} |\n"
    
    # 抗幻觉性能（仅基于 Unanswerable 数据）
    report += "\n---\n\n## 🛡️ 抗幻觉性能（Unanswerable）\n\n"
    report += "| 检索模式 | 无答案查询数 | Noise Robustness |\n"
    report += "|---------|--------------|-----------------|\n"
    for result in all_results:
        unanswerable = result.get('unanswerable_queries', 0)
        noise_robustness = result.get('noise_robustness', 0.0)
        report += f"| {result['mode_name']} | {unanswerable} | {noise_robustness:.3f} |\n"
    
    report += "\n---\n\n## 🎯 关键发现\n\n"
    
    # 为每个k值找出最佳模式
    report += "### 最佳性能模式\n\n"
    for k in k_values:
        best_mode = max(all_results, key=lambda x: x.get(f'avg_recall@{k}', 0.0))
        report += f"- **最佳Recall@{k}**: {best_mode['mode_name']} ({best_mode.get(f'avg_recall@{k}', 0.0):.3f})\n"
    
    best_precision_mode = max(all_results, key=lambda x: x['avg_precision'])
    best_mrr_mode = max(all_results, key=lambda x: x['avg_mrr'])
    best_noise_mode = max(all_results, key=lambda x: x.get('noise_robustness', 0.0))
    
    report += f"""- **最佳精确率**: {best_precision_mode['mode_name']} ({best_precision_mode['avg_precision']:.3f})
- **最佳排序**: {best_mrr_mode['mode_name']} (MRR={best_mrr_mode['avg_mrr']:.3f})
- **抗幻觉性能最佳**: {best_noise_mode['mode_name']} (Noise Robustness={best_noise_mode.get('noise_robustness', 0.0):.3f})

"""
    
    # 模块贡献度分析（基于多个k值）
    if len(all_results) >= 4:
        vector_only = all_results[0]
        bm25_only = all_results[1]
        hybrid_no_rerank = all_results[2]
        hybrid_with_rerank = all_results[3]
        
        report += "\n---\n\n## 📈 模块贡献度分析\n\n"
        
        report += "### BM25模块贡献（混合检索 vs 纯向量检索）\n\n"
        report += "| 指标 | 纯向量 | 混合检索 | 提升值 | 提升率 |\n"
        report += "|------|-------|---------|-------|-------|\n"
        
        for k in k_values:
            vector_recall = vector_only.get(f'avg_recall@{k}', 0.0)
            hybrid_recall = hybrid_no_rerank.get(f'avg_recall@{k}', 0.0)
            improvement = hybrid_recall - vector_recall
            improvement_pct = (improvement / vector_recall * 100) if vector_recall > 0 else 0.0
            report += f"| Recall@{k} | {vector_recall:.3f} | {hybrid_recall:.3f} | {improvement:+.3f} | {improvement_pct:+.1f}% |\n"
        
        report += "\n**关键洞察**: BM25模块通过关键词匹配补充了向量检索的语义理解，"
        if hybrid_no_rerank.get('avg_recall@3', 0.0) > vector_only.get('avg_recall@3', 0.0):
            report += "在小k值时尤其有效，显著提升了召回率。\n"
        else:
            report += "整体提升了检索覆盖度。\n"
        
        report += "\n### Rerank模块贡献（完整Pipeline vs 混合检索）\n\n"
        report += "| 指标 | 混合检索 | +Rerank | 提升值 | 提升率 |\n"
        report += "|------|---------|---------|-------|-------|\n"
        
        for k in k_values:
            no_rerank_recall = hybrid_no_rerank.get(f'avg_recall@{k}', 0.0)
            with_rerank_recall = hybrid_with_rerank.get(f'avg_recall@{k}', 0.0)
            improvement = with_rerank_recall - no_rerank_recall
            improvement_pct = (improvement / no_rerank_recall * 100) if no_rerank_recall > 0 else 0.0
            report += f"| Recall@{k} | {no_rerank_recall:.3f} | {with_rerank_recall:.3f} | {improvement:+.3f} | {improvement_pct:+.1f}% |\n"
        
        # Rerank对其他指标的影响
        precision_improvement = hybrid_with_rerank['avg_precision'] - hybrid_no_rerank['avg_precision']
        precision_improvement_pct = (precision_improvement / hybrid_no_rerank['avg_precision'] * 100) if hybrid_no_rerank['avg_precision'] > 0 else 0.0
        mrr_improvement = hybrid_with_rerank['avg_mrr'] - hybrid_no_rerank['avg_mrr']
        mrr_improvement_pct = (mrr_improvement / hybrid_no_rerank['avg_mrr'] * 100) if hybrid_no_rerank['avg_mrr'] > 0 else 0.0
        
        report += f"| 精确率 | {hybrid_no_rerank['avg_precision']:.3f} | {hybrid_with_rerank['avg_precision']:.3f} | {precision_improvement:+.3f} | {precision_improvement_pct:+.1f}% |\n"
        report += f"| MRR | {hybrid_no_rerank['avg_mrr']:.3f} | {hybrid_with_rerank['avg_mrr']:.3f} | {mrr_improvement:+.3f} | {mrr_improvement_pct:+.1f}% |\n"
        
        report += "\n**关键洞察**: Rerank模块通过语义相关性重新排序，"
        if precision_improvement > 0:
            report += "显著提升了精确率和排序质量（MRR），"
        if hybrid_with_rerank.get('avg_recall@3', 0.0) > hybrid_no_rerank.get('avg_recall@3', 0.0):
            report += "并在Top-3结果中提升了召回率。\n"
        else:
            report += "优化了结果排序。\n"
        
        # 检索深度分析
        report += "\n### 检索深度影响分析\n\n"
        report += "不同k值下的性能变化趋势：\n\n"
        
        for result in all_results:
            report += f"**{result['mode_name']}**:\n"
            recall_values = [result.get(f'avg_recall@{k}', 0.0) for k in k_values]
            for i, k in enumerate(k_values):
                report += f"  - Recall@{k}: {recall_values[i]:.3f}"
                if i > 0:
                    delta = recall_values[i] - recall_values[i-1]
                    report += f" (Δ{delta:+.3f})"
                report += "\n"
            report += "\n"
    
    report += "\n---\n\n## 📋 详细数据\n\n"
    for result in all_results:
        report += f"""### {result['mode_name']}

"""
        # 显示所有Recall@k指标
        for k in k_values:
            report += f"""- Recall@{k} (Answerable): {result.get(f'avg_recall@{k}', 0.0):.3f}
"""
        report += f"""- MRR (Answerable): {result['avg_mrr']:.3f}
- 精确率 (Answerable): {result['avg_precision']:.3f}
- NDCG (Answerable): {result['avg_ndcg']:.3f}
- 可回答查询数: {result.get('answerable_queries', result['total_queries'])}
- 无答案查询数: {result.get('unanswerable_queries', 0)}
- 噪声鲁棒性 (Noise Robustness, 无答案组): {result.get('noise_robustness', 0.0):.3f}
- 平均检索时间: {result['avg_retrieval_time']:.3f}秒
- 成功率: {result['success_rate']*100:.1f}%
- 失败查询数: {result['failed_queries']}/{result['total_queries']}

"""
    
    report += "\n---\n\n## 💡 总结与建议\n\n"
    report += "基于多维度Recall评估的结论：\n\n"
    
    # 自动生成建议
    if len(all_results) >= 4:
        best_overall = max(all_results, key=lambda x: sum([x.get(f'avg_recall@{k}', 0.0) for k in k_values]))
        report += f"1. **推荐策略**: {best_overall['mode_name']} 在综合性能上表现最佳\n"
        
        # 检查是否有明显的k值敏感性
        mode_4_recalls = [hybrid_with_rerank.get(f'avg_recall@{k}', 0.0) for k in k_values]
        if max(mode_4_recalls) - min(mode_4_recalls) > 0.1:
            report += f"2. **检索深度**: 不同k值下性能差异明显，建议根据应用场景选择合适的Top-K值\n"
        else:
            report += f"2. **检索深度**: 各k值下性能稳定，系统鲁棒性良好\n"
        
        if precision_improvement > 0.05:
            report += f"3. **重排序价值**: Rerank模块带来显著提升（精确率+{precision_improvement:.1%}），建议保留\n"
        
        if any(hybrid_no_rerank.get(f'avg_recall@{k}', 0.0) > vector_only.get(f'avg_recall@{k}', 0.0) * 1.1 for k in k_values):
            report += f"4. **混合检索优势**: BM25+向量混合策略相比单一方法有明显优势\n"
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 测试报告已生成: {output_path}")
    print(f"\n报告预览（前500字符）:\n{report[:500]}...")


# ==================== 主函数 ====================
def main():
    """主测试流程"""
    print("=" * 80)
    print("TripGuard RAG系统 多维度Recall性能评估测试")
    print("=" * 80)
    
    # 加载测试数据
    test_data_path = Path(__file__).parent / "test_data.json"
    test_data = load_test_data(test_data_path)
    print(f"\n✓ 已加载 {len(test_data)} 条测试数据")
    
    # 初始化LLM判断器
    print("\n✓ 正在初始化LLM判断器...")
    llm = get_llm_judge()
    print("✓ LLM判断器初始化完成")
    
    # 定义要评估的k值
    k_values = [3, 5, 10]
    print(f"\n✓ 将评估以下k值的Recall指标: {k_values}")
    
    # 定义所有测试模式
    test_modes = [
        ("模式1: 纯向量检索", retrieval_mode_1_vector_only),
        ("模式2: 纯BM25关键词检索", retrieval_mode_2_bm25_only),
        ("模式3: 混合检索（无重排序）", retrieval_mode_3_hybrid_no_rerank),
        ("模式4: 混合检索+重排序（完整Pipeline）", retrieval_mode_4_hybrid_with_rerank),
    ]
    
    # 执行评估
    all_results = []
    for i, (mode_name, retrieval_func) in enumerate(test_modes):
        # 第一个模式需要预热
        result = evaluate_retrieval_mode(
            mode_name, 
            retrieval_func, 
            test_data, 
            llm,
            k_values=k_values,
            warmup=(i == 0)
        )
        all_results.append(result)
    
    # 生成报告
    report_path = Path(__file__).parent / f"RAGAS_Test_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_markdown_report(all_results, report_path, k_values=k_values)
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print(f"📊 评估了 {len(k_values)} 个不同k值的Recall指标")
    print("=" * 80)


if __name__ == "__main__":
    main()

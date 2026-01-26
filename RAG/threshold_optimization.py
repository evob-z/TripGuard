"""
TripGuard RAG系统 Rerank阈值优化测试脚本
通过测试不同阈值来找到召回率和抗幻觉性能的最佳平衡点
"""
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib
# 导入matplotlib用于可视化
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 从test_ragas.py导入所有需要的函数
from test_ragas import (
    load_test_data,
    get_llm_judge,
    llm_judge_relevance,
    calculate_recall,
    retrieval_mode_4_hybrid_with_rerank
)

# 导入RAG模块
from retriever import get_vector_db, vector_search, bm25_search, ensemble_results, rerank_documents


# ==================== 阈值优化测试 ====================
def evaluate_threshold(
    threshold: float,
    test_data: List[Dict],
    llm,
    k: int = 10,
    top_k: int = 3,
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, Any]:
    """评估单个阈值的性能
    
    Args:
        threshold: 要测试的阈值
        test_data: 测试数据
        llm: LLM判断器
        k: 检索数量
        top_k: 重排序后返回的文档数量
        k_values: 要评估的k值列表
    
    Returns:
        包含各项指标的字典
    """
    print(f"\n{'='*60}")
    print(f"测试阈值: {threshold:.2f}")
    print(f"{'='*60}")
    
    # 最大k值用于检索
    max_k = max(k_values)
    
    results = {
        'threshold': threshold,
        'total_queries': len(test_data),
        'answerable_queries': 0,
        'unanswerable_queries': 0,
        'rejected_queries': 0,  # 因阈值过滤被拒绝的查询数
        'avg_retrieval_time': 0,
        'failed_queries': 0,
        'noise_robustness_scores': []
    }
    
    # 为每个k值创建独立的recall分数列表
    for k_val in k_values:
        results[f'recall@{k_val}_scores'] = []
    
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
        
        try:
            # 执行检索并计时
            start_time = time.time()
            
            # 手动执行混合检索+重排序流程以获取得分信息
            vector_db = get_vector_db()
            vector_docs = vector_search(question, vector_db, k=k)
            keyword_docs = bm25_search(question, vector_db, k=k)
            merged_docs = ensemble_results(vector_docs, keyword_docs)
            reranked_docs = rerank_documents(question, merged_docs, top_k=top_k)
            
            # 应用阈值过滤
            # 注意：rerank_documents可能在metadata中存储得分
            # 这里假设第一个文档得分最高，如果得分低于阈值则拒绝
            if reranked_docs and hasattr(reranked_docs[0], 'metadata'):
                max_score = reranked_docs[0].metadata.get('rerank_score', 1.0)
                if max_score < threshold:
                    reranked_docs = []  # 拒绝返回结果
                    results['rejected_queries'] += 1
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # 提取检索到的文本内容
            retrieved_contexts = [doc.page_content for doc in reranked_docs]
            
            if is_answerable:
                # 为每个k值计算 Recall@k
                for k_val in k_values:
                    recall_k = calculate_recall(
                        retrieved_contexts,
                        question,
                        ground_truth,
                        ground_truth_contexts,
                        llm,
                        k=k_val
                    )
                    results[f'recall@{k_val}_scores'].append(recall_k)
                
                if i % 5 == 0:  # 每5个查询打印一次进度
                    print(f"  [{i}/{len(test_data)}] Answerable查询评估中...")
            else:
                # 无答案查询：计算噪声鲁棒性
                has_support = False
                noise_k = min(k_values) if k_values else len(retrieved_contexts)
                for ret_context in retrieved_contexts[:noise_k]:
                    if llm_judge_relevance(question, ground_truth, ret_context, llm, is_unanswerable=True):
                        has_support = True
                        break
                is_robust = not has_support
                score = 1.0 if is_robust else 0.0
                results['noise_robustness_scores'].append(score)
                
                if i % 5 == 0:
                    print(f"  [{i}/{len(test_data)}] Unanswerable查询评估中...")
        
        except Exception as e:
            print(f"  ✗ 查询失败: {str(e)}")
            results['failed_queries'] += 1
            if is_answerable:
                for k_val in k_values:
                    results[f'recall@{k_val}_scores'].append(0.0)
            else:
                results['noise_robustness_scores'].append(0.0)
    
    # 计算平均值
    for k_val in k_values:
        scores = results[f'recall@{k_val}_scores']
        results[f'avg_recall@{k_val}'] = sum(scores) / len(scores) if scores else 0.0
    
    if results['noise_robustness_scores']:
        results['noise_robustness'] = sum(results['noise_robustness_scores']) / len(results['noise_robustness_scores'])
    else:
        results['noise_robustness'] = 0.0
    
    results['avg_retrieval_time'] = total_time / len(test_data) if test_data else 0.0
    results['rejection_rate'] = results['rejected_queries'] / len(test_data) if test_data else 0.0
    
    print(f"\n阈值 {threshold:.2f} 评估完成:")
    for k_val in k_values:
        print(f"  Recall@{k_val}: {results[f'avg_recall@{k_val}']:.3f}")
    print(f"  噪声鲁棒性: {results['noise_robustness']:.3f}")
    print(f"  拒绝率: {results['rejection_rate']*100:.1f}%")
    print(f"{'='*60}")
    
    return results


def find_optimal_threshold(
    thresholds: List[float],
    test_data: List[Dict],
    llm,
    k_values: List[int] = [3, 5, 10],
    alpha: float = 0.6
) -> tuple[Any, list[dict[str, Any]]]:
    """寻找最优阈值
    
    Args:
        thresholds: 要测试的阈值列表
        test_data: 测试数据
        llm: LLM判断器
        k_values: 要评估的k值列表
        alpha: 召回率权重（1-alpha为抗幻觉权重）
    
    Returns:
        (最优阈值, 所有结果列表)
    """
    print("\n" + "="*80)
    print("开始阈值优化测试")
    print(f"测试阈值范围: {min(thresholds):.1f} ~ {max(thresholds):.1f}")
    print(f"评估权重: 召回率={alpha:.1%}, 抗幻觉={1-alpha:.1%}")
    print("="*80)
    
    all_results = []
    
    # 预热
    print("\n🔥 执行模型预热...")
    try:
        _ = retrieval_mode_4_hybrid_with_rerank(test_data[0]['question'], k=10, top_k=3, score_threshold=0.5)
        print("✓ 预热完成\n")
    except:
        pass
    
    # 测试每个阈值
    for threshold in thresholds:
        result = evaluate_threshold(
            threshold,
            test_data,
            llm,
            k=10,
            top_k=3,
            k_values=k_values
        )
        all_results.append(result)
        time.sleep(1)  # 避免API限流
    
    # 计算综合得分（使用Recall@3作为代表性指标）
    for result in all_results:
        recall_score = result['avg_recall@3']
        noise_score = result['noise_robustness']
        result['combined_score'] = alpha * recall_score + (1 - alpha) * noise_score
    
    # 找到最优阈值
    best_result = max(all_results, key=lambda x: x['combined_score'])
    optimal_threshold = best_result['threshold']
    
    print(f"\n🎯 找到最优阈值: {optimal_threshold:.2f}")
    print(f"   Recall@3: {best_result['avg_recall@3']:.3f}")
    print(f"   噪声鲁棒性: {best_result['noise_robustness']:.3f}")
    print(f"   综合得分: {best_result['combined_score']:.3f}")
    
    return optimal_threshold, all_results


# ==================== 可视化 ====================
def plot_threshold_analysis(all_results: List[Dict], k_values: List[int], output_dir: Path, alpha: float = 0.6):
    """生成阈值分析的可视化图表
    
    Args:
        all_results: 所有阈值的评估结果
        k_values: 评估的k值列表
        output_dir: 输出目录
        alpha: 召回率权重
    """
    thresholds = [r['threshold'] for r in all_results]
    
    # 图1: 阈值 vs 多个Recall@k
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, k_val in enumerate(k_values):
        recall_scores = [r[f'avg_recall@{k_val}'] for r in all_results]
        ax1.plot(thresholds, recall_scores, marker='o', linewidth=2, 
                label=f'Recall@{k_val}', color=colors[i % len(colors)])
    
    ax1.set_xlabel('重排序阈值', fontsize=12)
    ax1.set_ylabel('召回率 (Recall)', fontsize=12)
    ax1.set_title('重排序阈值对召回率的影响', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    plot1_path = output_dir / "threshold_vs_recall.png"
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✓ 召回率曲线图已保存: {plot1_path}")
    plt.close()
    
    # 图2: 阈值 vs 噪声鲁棒性
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    noise_scores = [r['noise_robustness'] for r in all_results]
    ax2.plot(thresholds, noise_scores, marker='s', linewidth=2, 
            color='#d62728', label='噪声鲁棒性 (Noise Robustness)')
    
    ax2.set_xlabel('重排序阈值', fontsize=12)
    ax2.set_ylabel('噪声鲁棒性', fontsize=12)
    ax2.set_title('重排序阈值对抗幻觉性能的影响', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plot2_path = output_dir / "threshold_vs_noise_robustness.png"
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✓ 抗幻觉性能曲线图已保存: {plot2_path}")
    plt.close()
    
    # 图3: 综合视图（双轴图）
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # 左轴：Recall@3
    recall3_scores = [r['avg_recall@3'] for r in all_results]
    line1 = ax3.plot(thresholds, recall3_scores, marker='o', linewidth=2, 
                     color='#1f77b4', label='Recall@3')
    ax3.set_xlabel('重排序阈值', fontsize=12)
    ax3.set_ylabel('召回率 (Recall@3)', fontsize=12, color='#1f77b4')
    ax3.tick_params(axis='y', labelcolor='#1f77b4')
    
    # 右轴：噪声鲁棒性
    ax3_right = ax3.twinx()
    line2 = ax3_right.plot(thresholds, noise_scores, marker='s', linewidth=2, 
                          color='#d62728', label='噪声鲁棒性')
    ax3_right.set_ylabel('噪声鲁棒性 (Noise Robustness)', fontsize=12, color='#d62728')
    ax3_right.tick_params(axis='y', labelcolor='#d62728')
    
    # 标记最优点
    best_result = max(all_results, key=lambda x: x['combined_score'])
    best_threshold = best_result['threshold']
    best_recall3 = best_result['avg_recall@3']
    best_noise = best_result['noise_robustness']
    
    ax3.axvline(x=best_threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.plot(best_threshold, best_recall3, 'g*', markersize=15, 
            label=f'最优阈值 ({best_threshold:.2f})')
    
    # 合并图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=10)
    
    ax3.set_title(f'召回率与抗幻觉性能的权衡 (权重: {alpha:.0%} vs {1-alpha:.0%})', 
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plot3_path = output_dir / "threshold_tradeoff.png"
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✓ 权衡分析图已保存: {plot3_path}")
    plt.close()
    
    # 图4: 综合得分曲线
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    combined_scores = [r['combined_score'] for r in all_results]
    ax4.plot(thresholds, combined_scores, marker='D', linewidth=2, 
            color='#9467bd', label=f'综合得分 (α={alpha:.1f})')
    
    # 标记最优点
    ax4.axvline(x=best_threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.plot(best_threshold, best_result['combined_score'], 'g*', markersize=15,
            label=f'最优阈值 ({best_threshold:.2f})')
    
    ax4.set_xlabel('重排序阈值', fontsize=12)
    ax4.set_ylabel('综合得分', fontsize=12)
    ax4.set_title(f'加权综合得分 (Recall权重={alpha:.0%}, 抗幻觉权重={1-alpha:.0%})', 
                 fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=10)
    
    plot4_path = output_dir / "threshold_combined_score.png"
    plt.tight_layout()
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    print(f"✓ 综合得分曲线图已保存: {plot4_path}")
    plt.close()
    
    # 图5: 拒绝率曲线
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    rejection_rates = [r['rejection_rate'] * 100 for r in all_results]
    ax5.plot(thresholds, rejection_rates, marker='^', linewidth=2, 
            color='#8c564b', label='查询拒绝率')
    
    ax5.set_xlabel('重排序阈值', fontsize=12)
    ax5.set_ylabel('拒绝率 (%)', fontsize=12)
    ax5.set_title('重排序阈值对查询拒绝率的影响', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best', fontsize=10)
    
    plot5_path = output_dir / "threshold_vs_rejection_rate.png"
    plt.tight_layout()
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    print(f"✓ 拒绝率曲线图已保存: {plot5_path}")
    plt.close()


def generate_optimization_report(all_results: List[Dict], k_values: List[int], 
                                 output_path: Path, alpha: float = 0.6):
    """生成阈值优化的Markdown报告
    
    Args:
        all_results: 所有阈值的评估结果
        k_values: 评估的k值列表
        output_path: 输出路径
        alpha: 召回率权重
    """
    best_result = max(all_results, key=lambda x: x['combined_score'])
    optimal_threshold = best_result['threshold']
    
    report = f"""# TripGuard RAG系统 Rerank阈值优化报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**测试数据量**: {all_results[0]['total_queries']}个查询
**测试阈值范围**: {min(r['threshold'] for r in all_results):.2f} ~ {max(r['threshold'] for r in all_results):.2f}
**评估权重**: 召回率={alpha:.0%}, 抗幻觉性能={1-alpha:.0%}
**评估指标**: Recall@{', Recall@'.join(map(str, k_values))}, 噪声鲁棒性 (Noise Robustness)

---

## 🎯 最优阈值建议

基于加权综合得分分析，**推荐使用阈值: {optimal_threshold:.2f}**

### 性能表现

| 指标 | 数值 |
|------|------|
"""
    
    for k_val in k_values:
        report += f"| Recall@{k_val} | {best_result[f'avg_recall@{k_val}']:.3f} |\n"
    
    report += f"""| 噪声鲁棒性 | {best_result['noise_robustness']:.3f} |
| 综合得分 | {best_result['combined_score']:.3f} |
| 拒绝率 | {best_result['rejection_rate']*100:.1f}% |
| 平均检索时间 | {best_result['avg_retrieval_time']:.3f}秒 |

---

## 📊 全量阈值对比

### 召回率对比

| 阈值 | Recall@3 | Recall@5 | Recall@10 | 综合得分 |
|------|----------|----------|-----------|---------|
"""
    
    for result in all_results:
        recall_str = " | ".join([f"{result[f'avg_recall@{k_val}']:.3f}" for k_val in k_values])
        report += f"| {result['threshold']:.2f} | {recall_str} | {result['combined_score']:.3f} |\n"
    
    report += "\n### 抗幻觉性能对比\n\n"
    report += "| 阈值 | 噪声鲁棒性 | 拒绝率 |\n"
    report += "|------|-----------|-------|\n"
    
    for result in all_results:
        report += f"| {result['threshold']:.2f} | {result['noise_robustness']:.3f} | {result['rejection_rate']*100:.1f}% |\n"
    
    report += "\n---\n\n## 📈 关键洞察\n\n"
    
    # 分析趋势
    thresholds = [r['threshold'] for r in all_results]
    recall3_scores = [r['avg_recall@3'] for r in all_results]
    noise_scores = [r['noise_robustness'] for r in all_results]
    
    # 找出召回率和抗幻觉性能的拐点
    max_recall3_idx = recall3_scores.index(max(recall3_scores))
    max_noise_idx = noise_scores.index(max(noise_scores))
    
    report += f"""### 1. 召回率趋势

- **最高召回率**: 阈值={thresholds[max_recall3_idx]:.2f}时, Recall@3={recall3_scores[max_recall3_idx]:.3f}
- **趋势分析**: {"阈值越低，召回率越高" if recall3_scores[0] > recall3_scores[-1] else "阈值越高，召回率越高"}

### 2. 抗幻觉性能趋势

- **最高噪声鲁棒性**: 阈值={thresholds[max_noise_idx]:.2f}时, Noise Robustness={noise_scores[max_noise_idx]:.3f}
- **趋势分析**: {"阈值越低，抗幻觉性能越好" if noise_scores[0] > noise_scores[-1] else "阈值越高，抗幻觉性能越好"}

### 3. 权衡点分析

最优阈值 **{optimal_threshold:.2f}** 在召回率和抗幻觉性能之间取得了良好平衡：

"""
    
    # 与极端值对比
    lowest_threshold_result = all_results[0]
    highest_threshold_result = all_results[-1]
    
    report += f"""- 相比最低阈值({lowest_threshold_result['threshold']:.2f}):
  - Recall@3 变化: {best_result['avg_recall@3'] - lowest_threshold_result['avg_recall@3']:+.3f}
  - 噪声鲁棒性变化: {best_result['noise_robustness'] - lowest_threshold_result['noise_robustness']:+.3f}

- 相比最高阈值({highest_threshold_result['threshold']:.2f}):
  - Recall@3 变化: {best_result['avg_recall@3'] - highest_threshold_result['avg_recall@3']:+.3f}
  - 噪声鲁棒性变化: {best_result['noise_robustness'] - highest_threshold_result['noise_robustness']:+.3f}

"""
    
    report += "\n---\n\n## 💡 实施建议\n\n"
    
    report += f"""### 推荐配置

```python
# 在retrieval_mode_4_hybrid_with_rerank中使用
OPTIMAL_THRESHOLD = {optimal_threshold:.2f}
```

### 场景化建议

1. **标准场景（推荐）**
   - 使用阈值: **{optimal_threshold:.2f}**
   - 适用于: 需要平衡召回率和抗幻觉性能的通用场景
   - 预期表现: Recall@3={best_result['avg_recall@3']:.3f}, 噪声鲁棒性={best_result['noise_robustness']:.3f}

"""
    
    # 找出高召回率场景的阈值
    high_recall_result = max(all_results, key=lambda x: x['avg_recall@3'])
    high_noise_result = max(all_results, key=lambda x: x['noise_robustness'])
    
    report += f"""2. **高召回场景**
   - 使用阈值: **{high_recall_result['threshold']:.2f}**
   - 适用于: 需要最大化召回率的场景（例如FAQ系统）
   - 预期表现: Recall@3={high_recall_result['avg_recall@3']:.3f}, 噪声鲁棒性={high_recall_result['noise_robustness']:.3f}

3. **高精度场景**
   - 使用阈值: **{high_noise_result['threshold']:.2f}**
   - 适用于: 需要最小化幻觉风险的场景（例如法律咨询）
   - 预期表现: Recall@3={high_noise_result['avg_recall@3']:.3f}, 噪声鲁棒性={high_noise_result['noise_robustness']:.3f}

### 调优建议

- 如果发现召回率不足，可以适当降低阈值（建议范围: {optimal_threshold-0.1:.2f} ~ {optimal_threshold:.2f}）
- 如果发现幻觉问题严重，可以适当提高阈值（建议范围: {optimal_threshold:.2f} ~ {optimal_threshold+0.1:.2f}）
- 定期使用真实用户查询重新评估阈值，建议每月或每季度执行一次优化

"""
    
    report += "\n---\n\n## 📊 可视化图表\n\n"
    report += "详细的可视化分析图表已生成，包括：\n\n"
    report += "1. `threshold_vs_recall.png` - 阈值对召回率的影响\n"
    report += "2. `threshold_vs_noise_robustness.png` - 阈值对抗幻觉性能的影响\n"
    report += "3. `threshold_tradeoff.png` - 召回率与抗幻觉性能的权衡分析\n"
    report += "4. `threshold_combined_score.png` - 加权综合得分曲线\n"
    report += "5. `threshold_vs_rejection_rate.png` - 阈值对查询拒绝率的影响\n"
    
    report += "\n---\n\n## 📝 详细数据\n\n"
    
    for result in all_results:
        report += f"""### 阈值: {result['threshold']:.2f}

"""
        for k_val in k_values:
            report += f"- Recall@{k_val}: {result[f'avg_recall@{k_val}']:.3f}\n"
        report += f"""- 噪声鲁棒性: {result['noise_robustness']:.3f}
- 综合得分: {result['combined_score']:.3f}
- 拒绝率: {result['rejection_rate']*100:.1f}%
- 可回答查询数: {result['answerable_queries']}
- 无答案查询数: {result['unanswerable_queries']}
- 拒绝查询数: {result['rejected_queries']}
- 平均检索时间: {result['avg_retrieval_time']:.3f}秒

"""
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 优化报告已生成: {output_path}")


# ==================== 主函数 ====================
def main():
    """主测试流程"""
    print("=" * 80)
    print("TripGuard RAG系统 Rerank阈值优化测试")
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
    
    # 定义要测试的阈值范围
    thresholds = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 ~ 0.9
    print(f"\n✓ 将测试以下阈值: {thresholds}")
    
    # 权重设置（可根据业务需求调整）
    alpha = 0.6  # 召回率权重60%，抗幻觉权重40%
    print(f"\n✓ 评估权重: 召回率={alpha:.0%}, 抗幻觉={1-alpha:.0%}")
    
    # 执行阈值优化
    optimal_threshold, all_results = find_optimal_threshold(
        thresholds,
        test_data,
        llm,
        k_values=k_values,
        alpha=alpha
    )
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "threshold_optimization_results"
    output_dir.mkdir(exist_ok=True)
    
    # 生成可视化图表
    print("\n" + "="*80)
    print("生成可视化图表...")
    print("="*80)
    plot_threshold_analysis(all_results, k_values, output_dir, alpha=alpha)
    
    # 生成优化报告
    report_path = output_dir / f"Threshold_Optimization_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_optimization_report(all_results, k_values, report_path, alpha=alpha)
    
    print("\n" + "="*80)
    print("✅ 阈值优化测试完成！")
    print(f"🎯 推荐阈值: {optimal_threshold:.2f}")
    print(f"📊 所有结果已保存到: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

# ⚖️ 多任务学习与列表级重排

**CIKM 2025** · **4** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [CIKM 2026 (5)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 重排 ×2

<dl>
<dt><strong><a href="gref_a_unified_generative_framework_for_efficient_reranking_via_ordered_mul/">GReF: A Unified Generative Framework for Efficient Reranking via Ordered Multi-token Prediction</a></strong></dt>
<dd>
<p>用双向编码器+动态匹配自回归解码器统一生成式重排，Rerank-DPO 端到端对齐用户偏好，OMTP 多token并行解码将延迟压至 12.97ms，快手线上转发+2.98%。</p>
</dd>
<dt><strong><a href="mpformer_adaptive_framework_for_industrial_multi_task_personalized_sequenti/">MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever</a></strong></dt>
<dd>
<p>用目标条件化 Transformer 统一多目标召回，并以排序分数监督动态配额分配，在快手 4 亿 DAU 上实现 +0.43% 总时长与 66.7% 推理资源节省。</p>
</dd>
<dt><strong><a href="orca_mitigating_over_reliance_for_multi_task_dwell_time_prediction_with_cau/">ORCA: Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling</a></strong></dt>
<dd>
<p>用因果解耦显式建模并减去CTR对停留时间预测的负迁移，恢复中时长分桶语义，DT指标平均提升10.6%且不损CTR。</p>
</dd>
<dt><strong><a href="you_only_evaluate_once_a_tree_based_rerank_method_at_meituan/">You Only Evaluate Once: A Tree-based Rerank Method at Meituan</a></strong></dt>
<dd>
<p>用树状多尺度上下文提取+上下文缓存实现单阶段全排列重排，离线GAUC显著提升并落地美团外卖。</p>
</dd>
</dl>

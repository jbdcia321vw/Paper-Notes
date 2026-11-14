# ⚖️ 多任务学习与列表级重排

**KDD 2025** · **4** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [KDD 2026 (3)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 精排 ×3

<dl>
<dt><strong><a href="beyond_item_dissimilarities_diversifying_by_intent_in_recommender_systems/">Beyond Item Dissimilarities: Diversifying by Intent in Recommender Systems</a></strong></dt>
<dd>
<p>用实时意图概率驱动整页贪心重排，以贝叶斯后验更新替代物品相似度，YouTube 线上 DAU +0.05%、用户享受度 +0.09%。</p>
</dd>
<dt><strong><a href="breaking_the_top_k_barrier_advancing_top_k_ranking_metrics_optimization_in_/">Breaking the Top-$K$Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems</a></strong></dt>
<dd>
<p>用量化分位数把 Top-K 截断转成阈值比较，推导 NDCG@K 的紧致平滑上界 SL@K，四数据集三骨干平均提升 6.03%。</p>
</dd>
<dt><strong><a href="combinatorial_optimization_perspective_based_framework_for_multi_behavior_r/">Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation</a></strong></dt>
<dd>
<p>将多行为融合视为组合优化问题施加三阶段约束，并用分布拟合多专家网络抑制负迁移，三数据集 HR@10 最高提升 49.91%。</p>
</dd>
<dt><strong><a href="mopi_hfrs_a_multi_objective_personalized_health_aware_food_recommendation_s/">MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation</a></strong></dt>
<dd>
<p>构建首个融合 NHANES 医疗数据的健康感知食物推荐基准，用符号图结构学习 + Pareto 多目标优化平衡偏好/个性化健康/营养多样性，并以知识注入 LLM 生成可解释理由。</p>
</dd>
</dl>

# ⚖️ 多任务学习与列表级重排

**SIGIR 2025** · **4** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [SIGIR 2026 (4)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 重排 ×3

<dl>
<dt><strong><a href="comprehensive_list_generation_for_multi_generator_reranking/">Comprehensive List Generation for Multi-Generator Reranking</a></strong></dt>
<dd>
<p>提出多生成器-评估器重排框架与列表全面性指标，用 CLIG 自动学习互补生成器，离线 AUC 提升 0.9%、线上观看时长 +0.48%。</p>
</dd>
<dt><strong><a href="csmf_cascaded_selective_mask_fine_tuning_for_multi_objective_embedding_base/">CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval</a></strong></dt>
<dd>
<p>借鉴 PEFT 选择性掩码思想，按曝光→点击→转化级联顺序分阶段微调双塔召回，零参数增量实现多目标融合检索。</p>
</dd>
<dt><strong><a href="nr4der_neural_re_ranking_for_diversified_exercise_recommendation/">NR4DER: Neural Re-ranking for Diversified Exercise Recommendation</a></strong></dt>
<dd>
<p>用 mLSTM 难度过滤 + 活跃学生序列增强 + 神经重排，兼顾长尾学生与个性化学习节奏的习题推荐。</p>
</dd>
<dt><strong><a href="towards_principled_learning_for_re_ranking_in_recommender_systems/">Towards Principled Learning for Re-ranking in Recommender Systems</a></strong></dt>
<dd>
<p>提出收敛一致性与对抗一致性两条重排学习原则，以自洽重排与相邻换位扰动构造额外监督，即插即用地提升多种重排基线。</p>
</dd>
</dl>

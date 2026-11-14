# ⚖️ 多任务学习与列表级重排

**WWW (TheWebConf) 2025** · **5** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (1)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/)

🔥 **高频主题：** WWW 2025 ×5 · 重排 ×3

<dl>
<dt><strong><a href="a_hybrid_cross_stage_coordination_pre_ranking_model_for_online_recommendati/">A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems</a></strong></dt>
<dd>
<p>提出 HCCP 粗排模型，融合上游召回与下游精排多级未曝光样本，用 Margin InfoNCE 联合优化一致性与长尾精度，JD 线上 UCVR+14.9%。</p>
</dd>
<dt><strong><a href="llm4rerank_llm_based_auto_reranking_framework_for_recommendations/">LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations</a></strong></dt>
<dd>
<p>将准确性/多样性/公平性抽象为全连接图节点，用零样本 LLM 以 CoT 多跳自动重排，三数据集上多准则均衡超越 SOTA。</p>
</dd>
<dt><strong><a href="nlgr_utilizing_neighbor_lists_for_generative_rerank_in_personalized_recomme/">NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems</a></strong></dt>
<dd>
<p>用邻居列表相对奖励对齐评估器-生成器目标，并以采样式非自回归生成灵活跳转组合空间，美团线上 CTR +3.25%。</p>
</dd>
<dt><strong><a href="pcl_prompt_based_continual_learning_for_user_modeling_in_recommender_system/">PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems</a></strong></dt>
<dd>
<p>用位置级+上下文提示做持续用户建模，冻结骨干仅训提示，缓解灾难性遗忘，6/8 任务超越 MTL 上界。</p>
</dd>
<dt><strong><a href="xmtf_a_formula_free_model_for_reinforcement_learning_based_multi_task_fusio/">xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems</a></strong></dt>
<dd>
<p>用可学习单调融合单元替代预定义融合公式，配合两阶段混合训练，离线总观看时长提升约7%，线上日观看时长+0.833%</p>
</dd>
</dl>

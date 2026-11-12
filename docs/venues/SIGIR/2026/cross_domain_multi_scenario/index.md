# 🌐 跨域与多场景推荐

**SIGIR 2026** · **4** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

🔥 **高频主题：** 召回 ×2 · 跨域序列推荐 ×2

<dl>
<dt><strong><a href="bridging_behavior_and_semantics_for_time_aware_cross_domain_sequential_reco/">Bridging Behavior and Semantics for Time-aware Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用神经ODE建模连续时间行为偏好演化，LLM+反事实扰动提取时间敏感语义，门控自适应控制跨域迁移权重以缓解负迁移。</p>
</dd>
<dt><strong><a href="fedmm_federated_collaborative_signal_quantization_for_multi_market_ctr_pred/">FedMM: Federated Collaborative Signal Quantization for Multi-Market CTR Prediction</a></strong></dt>
<dd>
<p>用双层 RQ-VAE 码本做联邦传输介质，全局码本聚合跨市场协同信号、本地码本保留市场特异性，在 XMarket 2/3/4 市场 CTR 任务上以 AUC 超越联邦与集中式 MMR 基线。</p>
</dd>
<dt><strong><a href="llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Models Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong></dt>
<dd>
<p>用 LLM 生成可迁移跨域行为+噪声过滤+两阶段训练+域感知画像，缓解域不平衡与域迁移问题，三数据集全排序 Recall/NDCG 显著提升。</p>
</dd>
<dt><strong><a href="reinforcing_user_interest_evolution_in_multi_scenario_learning_for_recommen/">Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems</a></strong></dt>
<dd>
<p>用 Double Q-learning 估计场景置信度作为门控，加权三元组对比损失建模跨场景兴趣演化，KuaiSAR 全库排序 NDCG@20 大幅领先。</p>
</dd>
</dl>

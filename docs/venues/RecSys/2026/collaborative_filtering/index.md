# 🤝 协同过滤与双塔召回

**RecSys 2026** · **8** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** 召回 ×6

<dl>
<dt><strong><a href="mosaic_a_fleet_of_user_embedding_specialists_for_recommendation_at_meta/">Mosaic: A Fleet of User Embedding Specialists for Recommendation at Meta</a></strong></dt>
<dd>
<p>以四类异构专家模型族构建用户表征舰队，配合 MRM 与余弦冗余损失最大化增量信息，并提出免日志 CoEval 评估，离线 NE 稳定下降、线上 A/B 显著提升。</p>
</dd>
<dt><strong><a href="multi_probe_zero_collision_hash_mpzch_mitigating_embedding_collisions_and_e/">Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders</a></strong></dt>
<dd>
<p>用线性探测+身份/元数据辅助张量与主动淘汰，在工业级嵌入表上实现零碰撞并提升新内容新鲜度，训练/推理延迟几乎不变。</p>
</dd>
<dt><strong><a href="multilingual_semantic_retrieval_for_apple_music_search/">Multilingual Semantic Retrieval for Apple Music Search</a></strong></dt>
<dd>
<p>305M 多语双塔 + 课程式多目标微调，配合分位数校准混合召回，离线 Hit@10 相对提升 69%，线上尾部查询 CR 提升 7.93% 且无结果率降 86%。</p>
</dd>
<dt><strong><a href="parameter_efficient_single_collaborative_branch_for_recommendation/">Parameter-Efficient Single Collaborative Branch for Recommendation</a></strong></dt>
<dd>
<p>用单支路权重共享替代用户/物品双塔，参数量减半，精度持平且覆盖率与流行度偏差显著改善。</p>
</dd>
<dt><strong><a href="personalizing_incremental_video_search_with_hybrid_text_and_id_embeddings/">Personalizing Incremental Video Search with Hybrid Text and ID Embeddings</a></strong></dt>
<dd>
<p>面向 Apple TV 逐键增量搜索，用文本语义与 ID 协同双嵌入注入 XGBoost 排序器，短前缀 NDCG@10 提升 8.63%，线上 CTR +1.14%。</p>
</dd>
<dt><strong><a href="recevolve_a_knowledge_driven_autonomous_agent_system_for_recommender_system/">RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems</a></strong></dt>
<dd>
<p>将双塔召回模型的架构搜索全生命周期交给多智能体闭环，41 次生产级训练换来 NDCG@50 相对 +19.9%、线上满意度 +3.77%，并意外暴露奖励黑客漏洞。</p>
</dd>
<dt><strong><a href="smart_llm_augmented_hybrid_retrieval_for_dynamic_product_ads/">SMART: LLM-Augmented Hybrid Retrieval for Dynamic Product Ads</a></strong></dt>
<dd>
<p>揭示规则查询擅长 BM25 重定向、LLM 查询擅长 ANN 拓客的检索分解，用输出审计门控仅路由 10% 用户，线上转化率 +27.6%</p>
</dd>
<dt><strong><a href="unipinrec_unifying_generative_retrieval_and_ranking_at_pinterest_scale/">UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale</a></strong></dt>
<dd>
<p>用 MAM 掩码动作建模 + 混合样本 + 跨阶段 KV 缓存复用，将生成式召回与排序全栈统一，线上延迟降 11.1%、QPS 升 63.6%。</p>
</dd>
</dl>

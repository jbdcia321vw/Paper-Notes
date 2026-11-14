# 🤝 协同过滤与双塔召回

**KDD 2025** · **7** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [KDD 2026 (3)](../../2026/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** 召回 ×6

<dl>
<dt><strong><a href="embed_progressive_implicit_preference_in_unified_space_for_deep_collaborati/">Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering</a></strong></dt>
<dd>
<p>将多类隐式反馈按稀疏度映射为有序类别，用嵌套编码+嵌套OLR把渐进偏好压入统一空间，实现单次NNS一阶段排序。</p>
</dd>
<dt><strong><a href="high_precision_audience_expansion_via_extreme_classification_in_a_two_sided/">High Precision Audience Expansion via Extreme Classification in a Two-Sided Marketplace</a></strong></dt>
<dd>
<p>将 Airbnb 位置召回从矩形边界重构为 25M 级 S2 单元极端多分类，离线精度 +11.01%、线上未取消预订者 +0.17%。</p>
</dd>
<dt><strong><a href="iu4rec_interest_unit_based_product_organization_and_recommendation_for_e_co/">IU4Rec: Interest Unit-Based Product Organization and Recommendation for E-Commerce Platform</a></strong></dt>
<dd>
<p>闲鱼C2C限库存场景下以兴趣单元(IU)重构商品组织与两阶段推荐，IU-Boosted网络离线AUC+1.91%、线上CTR+2.90%</p>
</dd>
<dt><strong><a href="on_the_role_of_weight_decay_in_collaborative_filtering_a_popularity_perspec/">On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective</a></strong></dt>
<dd>
<p>揭示权重衰减本质是将流行度编码进嵌入模长，提出 PRISM 初始化策略替代权重衰减，NDCG 最高提升 4.77% 且训练加速 38.48%。</p>
</dd>
<dt><strong><a href="optimizing_recall_or_relevance_a_multi_task_multi_head_approach_for_item_to/">Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for Item-to-Item Retrieval in Recommendation</a></strong></dt>
<dd>
<p>用多任务蒸馏损失+多头I2I架构联合优化共互动召回与语义相关性，离线召回@500提升14.4%、L2主题相关性提升56.5%，并完成线上A/B验证。</p>
</dd>
<dt><strong><a href="real_time_indexing_for_large_scale_recommendation_by_streaming_vector_quant/">Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever</a></strong></dt>
<dd>
<p>用流式 VQ 码本替代 HNSW/DR 静态索引，实现实时挂载、索引均衡与可修复，抖音全量上线显著提升用户参与。</p>
</dd>
<dt><strong><a href="synergizing_implicit_and_explicit_user_interests_a_multi_embedding_retrieva/">Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest</a></strong></dt>
<dd>
<p>Pinterest 工业级多嵌入召回框架，用可微聚类建模隐式兴趣、条件召回建模显式兴趣，线上 HF Repins +1.09%、A-Pincepts +0.81%。</p>
</dd>
</dl>

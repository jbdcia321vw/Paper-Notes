# 🤝 协同过滤与双塔召回

**SIGIR 2026** · **5** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [KDD 2026 (1)](../../../KDD/2026/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** 召回 ×4

<dl>
<dt><strong><a href="balanced_co_clustering_of_users_and_items_for_embedding_table_compression_i/">Balanced Co-Clustering of Users and Items for Embedding Table Compression in Recommender Systems</a></strong></dt>
<dd>
<p>BACO 以平衡共聚类构建用户/物品草图矩阵压缩嵌入表，75% 参数削减下 Recall 仅降 1.85%，比最强基线快 346 倍</p>
</dd>
<dt><strong><a href="diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong></dt>
<dd>
<p>融合原型意图与分布意图重构表征，以对齐-均匀性为主目标联合优化，三数据集全库排序显著超越15个基线。</p>
</dd>
<dt><strong><a href="ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong></dt>
<dd>
<p>在 Stiefel 流形上约束 Item-ID 投影矩阵，保持奇异值谱等距以缓解稀疏扩展下的表征坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt><strong><a href="silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom 索引 + Int8 融合核，吞吐提升 23.7×、成本效率 13.35×。</p>
</dd>
<dt><strong><a href="zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong></dt>
<dd>
<p>零权重免训练新闻推荐框架，用时间衰减×表征相似度打分，离线AUC超越NRMS且推理快600倍，线上CTR接近NRMS</p>
</dd>
</dl>

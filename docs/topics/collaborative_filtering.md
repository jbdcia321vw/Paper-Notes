# 🤝 协同过滤与双塔召回 (Collaborative Filtering & Two-Tower Retrieval)

> **研究范畴**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

当前方向共收录 **9** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/balanced_co_clustering_of_users_and_items_for_embedding_table_compression_i/">Balanced Co-Clustering of Users and Items for Embedding Table Compression in Recommender Systems</a></strong></dt>
<dd>
<p>BACO 以平衡共聚类构建用户/物品草图矩阵压缩嵌入表，75% 参数削减下 Recall 仅降 1.85%，比最强基线快 346 倍</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong></dt>
<dd>
<p>提出 CAS/CTS/CMS 三模块协同增强双塔召回，兼容在线学习，离线三数据集 AUC 提升，线上广告收入最高 +8.36% 且保持毫秒级延迟。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong></dt>
<dd>
<p>融合原型意图与分布意图重构表征，以对齐-均匀性为主目标联合优化，三数据集全库排序显著超越15个基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong></dt>
<dd>
<p>在 Stiefel 流形上约束 Item-ID 投影矩阵，保持奇异值谱等距以缓解稀疏扩展下的表征坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/promax_exploring_the_potential_of_llm_derived_profiles_with_distribution_sh/">ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems</a></strong></dt>
<dd>
<p>将 LLM 画像视为检索信号，用监督/自监督双重分布塑形引导基座推荐模型，全库排序下 NDCG@10 最高提升 8.44%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/rethinking_semantic_collaborative_integration_why_alignment_is_not_enough/">Rethinking Semantic–Collaborative Integration: Why Alignment Is Not Enough</a></strong></dt>
<dd>
<p>论证语义-协同表征仅部分共享，全局对齐会压制视图私有信号；提出互补融合视角与诊断指标，最小融合在稀疏基准上超越对齐范式。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom 索引 + Int8 融合核，吞吐提升 23.7×、成本效率 13.35×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动信号仅作相关物品内偏好精修，融合交叉编码器分级标签、多渠道检索先验与去偏互动构建统一监督目标训练双编码器，离线 P@25 +10.5%、NDCG@25 +5.7%，线上曝光 +0.60%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong></dt>
<dd>
<p>零权重免训练新闻推荐框架，用时间衰减×表征相似度打分，离线AUC超越NRMS且推理快600倍，线上CTR接近NRMS</p>
</dd>
</dl>

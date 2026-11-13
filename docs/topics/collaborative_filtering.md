# 🤝 协同过滤与双塔召回 (Collaborative Filtering & Two-Tower Retrieval)

> **研究范畴**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

当前方向共收录 **18** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/CIKM/2026/collaborative_filtering/dualspectralcf_training_free_sign_aware_spectral_collaborative_filtering/">DualSpectralCF: Training-Free Sign-Aware Spectral Collaborative Filtering</a></strong></dt>
<dd>
<p>在任意谱滤波骨干上以带符号输入信号与带符号物品算子注入显式负反馈，免训练即插即用，全库排序下 R@20 最高提升 32.6%，比 SIGformer 快 7.7–155.3 倍。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/think_to_personalize_unifying_reasoning_and_retrieval_for_user_centric_pers/">Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval</a></strong></dt>
<dd>
<p>用 SFT+GRPO 两阶段让 LLM 显式推理用户历史生成意图增强查询并统一编码，Full-Ranking 下 Broad/LongTail 召回显著提升，线上订单量 +0.46%</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/training_free_llm_based_recommendation_with_post_llm_item_refinement_using_/">Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals</a></strong></dt>
<dd>
<p>提出 post-LLM 范式 CoRRe，用共购图方向细化与流行度幅度校准改造 LLM 物品表征，免训练即在三域 Amazon 上超越全部免训练基线并多数超越训练式方法。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/collaborative_filtering/transretrieval_scaling_up_transformer_based_retrieval_for_industrial_recomm/">TransRetrieval: Scaling Up Transformer-Based Retrieval for Industrial Recommendation</a></strong></dt>
<dd>
<p>用加权平均聚合+目标Token压缩+位置式域嵌入，让Transformer在召回阶段实现对数线性Scaling，线上收入+2.53%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/collaborative_memory_augmentation_for_generative_recommendation/">Collaborative Memory Augmentation for Generative Recommendation</a></strong></dt>
<dd>
<p>为生成式推荐引入跨用户协同记忆库，用可学习查询token压缩序列、目标感知检索与门控交叉注意力融合，Full-Ranking下NDCG@5最高提升23.7%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/rethinking_popularity_bias_in_collaborative_filtering_via_analytical_vector/">Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition</a></strong></dt>
<dd>
<p>揭示 BPR 优化在嵌入空间刻出“流行度方向”的几何偏差，提出非对称方向分解校正 DDC，在 5 个骨干上 MRR@10 提升 6%–29% 且 AvgPop@10 降 21%–47%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/balanced_co_clustering_of_users_and_items_for_embedding_table_compression_i/">Balanced Co-Clustering of Users and Items for Embedding Table Compression in Recommender Systems</a></strong></dt>
<dd>
<p>BACO 以平衡共聚类构建用户/物品草图矩阵压缩嵌入表，75% 参数削减下 Recall 仅降 1.85%，比最强基线快 346 倍</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong></dt>
<dd>
<p>融合原型意图与分布意图重构表征，以对齐-均匀性为主目标联合优化，三数据集全库排序显著超越15个基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong></dt>
<dd>
<p>在 Stiefel 流形上约束 Item-ID 投影矩阵，保持奇异值谱等距以缓解稀疏扩展下的表征坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom 索引 + Int8 融合核，吞吐提升 23.7×、成本效率 13.35×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong></dt>
<dd>
<p>零权重免训练新闻推荐框架，用时间衰减×表征相似度打分，离线AUC超越NRMS且推理快600倍，线上CTR接近NRMS</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/collaborative_filtering/memory_is_no_longer_a_bottleneck_memory_efficient_graph_filtering_for_scala/">Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering</a></strong></dt>
<dd>
<p>用逐用户 Krylov 子空间替代全量物品相似度图，实现免训练多项式图滤波，内存降 5.74×、推理加速 26× 且精度反超 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/debias_fairness_industrial/mitigating_popularity_bias_in_recommendation_with_global_listwise_learning_/">Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting</a></strong></dt>
<dd>
<p>用 IPS 加权多项似然实现列表级全局无偏学习，并以 Bi-Weighting 平滑倾向估计与渐进式去偏，六数据集 Recall@20 平均提升 9.43%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/collaborative_filtering/have_we_really_understood_collaborative_information_an_empirical_investigat/">Have We Really Understood Collaborative Information? An Empirical Investigation</a></strong></dt>
<dd>
<p>以物品共现路径长度量化协作信息为0/1/2-hop协作关系，系统实证其在六个基准上的分布、对12种推荐算法的影响与预测依赖，揭示模型强于简单CR、弱于复杂CR的核心规律。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/identifying_and_upweighting_power_niche_users_to_mitigate_popularity_bias_i/">Identifying and Upweighting Power-Niche Users to Mitigate Popularity Bias in Recommendations</a></strong></dt>
<dd>
<p>识别高活跃-小众偏好用户并对其在BPR损失中双重加权，在三个基准上实现性能与流行度去偏的帕累托双赢。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/spattack_subgroup_poisoning_attacks_on_federated_recommender_systems/">Spattack: Subgroup Poisoning Attacks on Federated Recommender Systems</a></strong></dt>
<dd>
<p>首个联邦推荐子群投毒攻击：近似-提升范式+对比排斥与自适应加权，0.1%恶意客户端即达100% ER@5且几乎不伤非目标群。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/talos_optimizing_top_k_accuracy_in_recommender_systems/">Talos: Optimizing Top-$K$ Accuracy in Recommender Systems</a></strong></dt>
<dd>
<p>用量化阈值把 Top-K 截断改写为分数-阈值比较，配合采样分位回归与约束项，直接优化 Precision@K/Recall@K 并具备 DRO 鲁棒性。</p>
</dd>
</dl>

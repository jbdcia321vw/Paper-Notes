# 🤝 协同过滤与双塔召回

**SIGIR 2026** · **6** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** 召回 ×6

<dl>
<dt><strong><a href="cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong></dt>
<dd>
<p>以塔内自校正、跨塔EMA同步与级联模型共享三机制增强双塔召回，离线三数据集AUC一致提升，线上广告收入+8.36%且QPS仅降0.59%。</p>
</dd>
<dt><strong><a href="diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong></dt>
<dd>
<p>以原型意图与分布意图重构表征，用对齐-均匀性+粗细粒度匹配联合优化，Amazon-book R@20 相对提升 10.6%</p>
</dd>
<dt><strong><a href="ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong></dt>
<dd>
<p>在 Item-ID 投影层上施加 Stiefel 流形正交约束，抑制稀疏缩放下的嵌入坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt><strong><a href="silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom Index 与 Int8 融合 ANN 协同设计，吞吐提升 23.7×、成本效率提升 13.35×。</p>
</dd>
<dt><strong><a href="unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动为相关项内偏好信号，融合交叉编码器分级标签、多通道检索先验与去偏互动构建统一双塔训练目标，离线 P@25 +10.5%、NDCG@25 +5.7%。</p>
</dd>
<dt><strong><a href="zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong></dt>
<dd>
<p>零权重免训练新闻推荐框架，以时间衰减×表征相似度打分，离线全库排序超越 NRMS，线上 CTR 逼近 SOTA 且推理快 600 倍。</p>
</dd>
</dl>

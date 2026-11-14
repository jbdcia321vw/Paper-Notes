# 🤝 协同过滤与双塔召回

**WWW (TheWebConf) 2025** · **7** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (5)](../../2026/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/)

🔥 **高频主题：** WWW 2025 ×7 · 召回 ×6

<dl>
<dt><strong><a href="criteria_aware_graph_filtering_extremely_fast_yet_accurate_multi_criteria_r/">Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation</a></strong></dt>
<dd>
<p>免训练免矩阵分解的多准则图滤波推荐，按准则定制多项式低通滤波器并融合用户准则偏好，BA 数据集 0.2 秒推理、NDCG@5 最高提升 24%。</p>
</dd>
<dt><strong><a href="does_weighting_improve_matrix_factorization_for_recommender_systems/">Does Weighting Improve Matrix Factorization for Recommender Systems?</a></strong></dt>
<dd>
<p>系统实证加权隐式反馈矩阵分解：大秩模型下不加权常持平或更优，并给出加权目标闭式解与预条件共轭梯度高效求解。</p>
</dd>
<dt><strong><a href="frequency_aware_graph_signal_processing_for_collaborative_filtering/">Frequency-aware Graph Signal Processing for Collaborative Filtering</a></strong></dt>
<dd>
<p>用级联高通-低通滤波捕获独有/共性特征、并行低通滤波捕获高阶邻域，免训练线性组合在六个数据集上 Full-Ranking 提升 2-5%。</p>
</dd>
<dt><strong><a href="graph_neural_controlled_differential_equations_for_collaborative_filtering/">Graph Neural Controlled Differential Equations For Collaborative Filtering</a></strong></dt>
<dd>
<p>用 MLP 控制器生成连续时变权重矩阵替代 GODE-CF 的离散权重，在四个 Amazon 5-core 数据集 Full-Ranking 下 Recall@20 全面超越 LightGCN/GODE-CF。</p>
</dd>
<dt><strong><a href="graphhash_graph_clustering_enables_parameter_efficiency_in_recommender_syst/">GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems</a></strong></dt>
<dd>
<p>用模块度二部图聚类替代 ID 哈希，将图平滑前置到预处理阶段，嵌入表缩减 75%+ 时 Recall@20 平均提升 101.52%。</p>
</dd>
<dt><strong><a href="interactrank_personalized_web_scale_search_pre_ranking_with_cross_interacti/">InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features</a></strong></dt>
<dd>
<p>在双塔预排序点积之上叠加历史交互统计特征 IQP 与实时用户序列，以 1.1× FLOPs 实现线上 SIFR +6.5%、F1S +9.2%。</p>
</dd>
<dt><strong><a href="rankformer_a_graph_transformer_for_recommendation_based_on_ranking_objectiv/">Rankformer: A Graph Transformer for Recommendation based on Ranking Objective</a></strong></dt>
<dd>
<p>从 BPR 梯度下降推导出图 Transformer 层，正负样本分离聚合+基准项+偏移项，线性加速后 4 数据集平均提升 4.48%</p>
</dd>
</dl>

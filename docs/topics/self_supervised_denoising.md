# 🔄 自监督对比学习与交互去噪 (Self-Supervised, Contrastive & Denoising Rec)

> **研究范畴**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

当前方向共收录 **21** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_modality_harmony_orthogonal_purification_and_topology_guided_moe_for/">Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation</a></strong></dt>
<dd>
<p>以纯协同锚点为几何参考轴正交分解并保能截断模态噪声，配合解耦 Sigmoid 路由 MoE 与冲突感知 safe-SSL，在 Amazon 三数据集 8:1:1 划分下 R@20 最高提升 6.55%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/graph_rec/do_all_nodes_benefit_equally_from_knowledge_graphs_adaptive_node_aware_kg_f/">Do All Nodes Benefit Equally from Knowledge Graphs? Adaptive Node-Aware KG Fusion for Recommendation</a></strong></dt>
<dd>
<p>以对抗扰动估计节点级 CF 不稳定性，自适应门控 KG 注入强度并配合自适应对比对齐，四数据集 Full-Ranking 平均排名第一。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/collaborative_filtering/dualspectralcf_training_free_sign_aware_spectral_collaborative_filtering/">DualSpectralCF: Training-Free Sign-Aware Spectral Collaborative Filtering</a></strong></dt>
<dd>
<p>在任意谱滤波骨干上以带符号输入信号与带符号物品算子注入显式负反馈，免训练即插即用，全库排序下 R@20 最高提升 32.6%，比 SIGformer 快 7.7–155.3 倍。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/from_overlooked_to_explored_recovering_item_relations_via_mixture_of_perspe/">From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation</a></strong></dt>
<dd>
<p>揭示自注意力相似性偏置导致异构关系被系统性忽略，提出 K 个视角透镜（亲和/对比双视图）校准注意力，7 个基准全库排序平均提升 3–11%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/god_enhancing_generalization_via_deep_grafting_for_sequential_recommendatio/">GOD: Enhancing Generalization via Deep Grafting for Sequential Recommendation</a></strong></dt>
<dd>
<p>用「嫁接」构造教师条件混合源模型，对学生嵌入表/编码器做组件级蒸馏，Full-Ranking 下最高相对提升 13.92%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/information_guided_selective_modality_interest_alignment_for_multimodal_rec/">Information-Guided Selective Modality-Interest Alignment for Multimodal Recommendation</a></strong></dt>
<dd>
<p>以互信息/条件熵分解为理论视角，用行为校准图精炼+兴趣对比+共享子空间选择性对齐，三数据集 Full-Ranking 全面 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/graph_rec/neural_tree_collaborative_filtering_rethinking_graph_collaborative_filterin/">Neural Tree Collaborative Filtering: Rethinking Graph Collaborative Filtering as Tree Collaborative Filtering with Curvature-Aware Propagation Depth</a></strong></dt>
<dd>
<p>以局部度不平衡曲率代理为每节点分配自适应传播深度，将 GCF 重释为节点根树集成，Full-Ranking 下 R@20 稳定超越 LightGCN 与 SSL 骨干。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cold_start_rec/denoising_implicit_feedback_for_cold_start_recommendation/">Denoising Implicit Feedback for Cold-start Recommendation</a></strong></dt>
<dd>
<p>用内容相似热物品协同表征为冷物品生成置信加权伪标签，并以相对熵×冷启动状态估计样本不确定性做自适应软标签纠正，快手十亿级 A/B 全指标提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong></dt>
<dd>
<p>借鉴最近发展区理论提出 Teacher-Peer-Self 多源负采样框架，以分歧重排提升多样性、共识蒸馏复用打分，六数据集五骨干稳定超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/from_raw_features_to_effective_embeddings_a_three_stage_approach_for_multim/">From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation</a></strong></dt>
<dd>
<p>用 VLM/LLM 摘要 + LightGCN 传播 + 跨视图对比学习三阶段增强多模态特征，全库排序下 Recall@10 提升 7-15%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/multi_modal_relational_item_representation_learning_for_inferring_substitut/">Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items</a></strong></dt>
<dd>
<p>MMSC 融合冻结多模态基座与元路径行为编码器，用对比去噪与 LLM 辅助监督推断替代/互补关系，五数据集提升 26.1%/39.2%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/qagcf_graph_collaborative_filtering_for_q_a_recommendation/">QAGCF: Graph Collaborative Filtering for Q&amp;A Recommendation</a></strong></dt>
<dd>
<p>解耦问答对协同/语义双视图并融合全局图，用 Jacobi 多项式图滤波器处理异质性，对比学习增强鲁棒性，全库排序下显著超越基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong></dt>
<dd>
<p>提出 TIDE，用 Hawkes 增强傅里叶时间编码 + 双专家解耦习惯/探索意图，在四个 Amazon NBR 数据集上显著超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/graph_rec/hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/generative_rec/continuous_time_discrete_space_diffusion_model_for_recommendation/">Continuous-time Discrete-space Diffusion Model for Recommendation</a></strong></dt>
<dd>
<p>将推荐历史交互建模为连续时间离散空间（吸收态掩码）扩散，配合流行度感知噪声调度、一致性参数化与多跳对比引导，在三个数据集 Full-Ranking 下 Recall@10 相对提升 4.2%–5.2%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/self_supervised_denoising/simdiffrec_semantic_similarity_guided_diffusion_for_contrastive_sequential_/">SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义相似物品均值替代高斯随机噪声做扩散增强，按去噪置信度选增强位置并配难负采样，五数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/efficient_content_based_recommendation_model_training_via_noise_aware_cores/">Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection</a></strong></dt>
<dd>
<p>用末层梯度子模优化选核集，配合渐进标签自校正与 MC-Dropout 去噪，1% 数据恢复 93-95% 全量性能。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/mclmr_a_model_agnostic_causal_learning_framework_for_multi_behavior_recomme/">MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用后门干预+MoE自适应聚合+偏差感知对比学习，作为即插即用框架增强多行为推荐，三数据集全库排序显著提升。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/radar_relation_aware_diffusion_asymmetric_graph_contrastive_learning_for_re/">RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>以 VGAE 生成视图与关系感知去噪视图构成双视图，叠加扩散引导增广与非对称对比目标，在 Full-Ranking 下 Recall@20 相对 SOTA 提升 3.8%–4.7%。</p>
</dd>
</dl>

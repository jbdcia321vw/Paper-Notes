# 🔄 自监督对比学习与交互去噪 (Self-Supervised, Contrastive & Denoising Rec)

> **研究范畴**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

当前方向共收录 **113** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/sequential_rec/augmenting_sequential_recommendation_with_balanced_relevance_and_diversity/">Augmenting Sequential Recommendation with Balanced Relevance and Diversity</a></strong></dt>
<dd>
<p>提出即插即用平衡增强插件 BASRec，以单序列随机算子+表征级 mixup 与跨序列非线性 mixup 兼顾相关性与多样性，全库排序下 SASRec 平均提升 33.8%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/rl_bandit_rec/contrastive_representation_for_interactive_recommendation/">Contrastive Representation for Interactive Recommendation</a></strong></dt>
<dd>
<p>用偏好排序对比学习(PRCL)为 DRL 交互推荐提炼状态表征，冷启动下样本效率显著提升。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/social_group_rec/disentangled_modeling_of_preferences_and_social_influence_for_group_recomme/">Disentangled Modeling of Preferences and Social Influence for Group Recommendation</a></strong></dt>
<dd>
<p>解耦用户偏好与社会影响的双通道(超)图传播，配合社交重要性引导的对比学习，缓解群组推荐的偏好偏差与数据稀疏。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/collaborative_filtering/from_pairwise_to_ranking_climbing_the_ladder_to_ideal_collaborative_filteri/">From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking</a></strong></dt>
<dd>
<p>用噪声注入构造伪排序并以多序分类排序损失替代成对损失，在四个数据集 Full-Ranking 下 NDCG@10 最高提升 38.9%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/sequential_rec/future_sight_and_tough_fights_revolutionizing_sequential_recommendation_wit/">Future Sight and Tough Fights: Revolutionizing Sequential Recommendation with FENRec</a></strong></dt>
<dd>
<p>用时间相关软标签挖掘未来交互、用锚-负混合构造持久困难负样本，四数据集全库排序平均提升 6.16%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/sequential_rec/lightweight_yet_fine_grained_a_graph_capsule_convolutional_network_with_sub/">Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性注意力胶囊图+低秩子空间对齐，在共享账户混合序列上做细粒度潜在用户解耦，精度大幅领先且参数/耗时更低。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/mentor_multi_level_self_supervised_learning_for_multimodal_recommendation/">MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation</a></strong></dt>
<dd>
<p>以 ID 嵌入为锚做四级跨模态分布对齐，叠加特征掩码与图扰动自监督，Amazon 三集 Recall@20 最高提升 74.73%</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/sequential_rec/exploiting_inter_session_information_with_frequency_enhanced_dual_path_netw/">Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation</a></strong></dt>
<dd>
<p>FreqRec 用可学习复数频域 MLP 双路建模跨会话群体节奏与用户局部高频，并加频域一致性损失，全库排序下 NDCG@10 最高提升 7.38%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/self_supervised_denoising/g_ubs_towards_robust_understanding_of_implicit_feedback_via_group_aware_use/">G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation</a></strong></dt>
<dd>
<p>用 LLM 群组画像 + 群组感知 GRPO 强化学习为个体隐式反馈去噪，构建 IF-VR 多模态基准，推理准确率提升 14.9%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/self_supervised_denoising/hard_vs_noise_resolving_hard_noisy_sample_confusion_in_recommender_systems_/">Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 语义/逻辑双相关性信号区分困难样本与噪声样本，配合目标对齐与图对比抑制幻觉，全库排序下 Recall@20 平均提升约 45%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/generating_negative_samples_for_multi_modal_recommendation/">Generating Negative Samples for Multi-Modal Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 三步提示生成兼具凝聚力与难度的属性级负样本，并以因果学习解耦关键特征，全库排序下 Recall@10 最高提升 9.02%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/i_3_mrec_invariant_learning_with_information_bottleneck_for_incomplete_moda/">I$^3$-MRec: Invariant Learning with Information Bottleneck for Incomplete Modality Recommendation</a></strong></dt>
<dd>
<p>将各模态视为独立语义环境做 IRM 不变学习，并用 IB 压缩模态特有信息，在模态缺失场景下 Full-Ranking Recall@20 稳定提升 2%–4%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/refining_contrastive_learning_and_homography_relations_for_multi_modal_reco/">Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation</a></strong></dt>
<dd>
<p>用元网络+正交约束精炼多模态对比学习，并补建用户兴趣图与物品共现图，全库排序下 R@20 最高提升约 9.8%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_modality_harmony_orthogonal_purification_and_topology_guided_moe_for/">Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation</a></strong></dt>
<dd>
<p>以纯协同锚点为几何参考轴正交分解并保能截断模态噪声，配合解耦 Sigmoid 路由 MoE 与冲突感知 safe-SSL，在 Amazon 三数据集 8:1:1 划分下 R@20 最高提升 6.55%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/a_self_supervised_mixture_of_experts_framework_for_multi_behavior_recommend/">A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation</a></strong></dt>
<dd>
<p>揭示多行为推荐在已访问/未访问物品上的性能鸿沟，提出双专家 MoE + 专家自适应 SSL（访问过滤/新颖性推断对比 + 行为生成）框架，全库排序下 HR@20 最高提升 65.46%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/empowering_denoising_sequential_recommendation_with_large_language_model_em/">Empowering Denoising Sequential Recommendation with Large Language Model Embeddings</a></strong></dt>
<dd>
<p>用 LLM 语义嵌入与协同嵌入做跨模态兴趣对齐，缓解冷门物品过度去噪，四数据集全排序 HR/NDCG 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/generative_data_augmentation_in_graph_contrastive_learning_for_recommendati/">Generative Data Augmentation in Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用 VAE 式生成器自适应生成噪声替代固定随机扰动构造对比视图，并引入物品互补矩阵，在三个稀疏数据集上 Full-Ranking 提升约 3–6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/modality_alignment_with_multi_scale_bilateral_attention_for_multimodal_reco/">Modality Alignment with Multi-scale Bilateral Attention for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 MambaRec，用多尺度空洞卷积+双注意力做局部图文对齐、MMD+InfoNCE 做全局分布对齐，Amazon 三数据集 Recall@20 最高提升约 5.4%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/sequential_rec/multi_item_query_attention_for_stable_sequential_recommendation/">Multi-Item-Query Attention for Stable Sequential Recommendation</a></strong></dt>
<dd>
<p>用多物品查询窗口+虚拟物品替代单查询掩码注意力，缓解序列噪声敏感，ML-1M HR@5 提升 21.2%，LastFM 达 SOTA。</p>
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
<dt><strong><a href="../../venues/DASFAA/2025/collaborative_filtering/learning_recommender_systems_with_soft_target_a_decoupled_perspective/">Learning Recommender Systems with Soft Target: A Decoupled Perspective</a></strong></dt>
<dd>
<p>将 Softmax 软标签目标解耦为目标置信度与未观测项兴趣分布两项，配合 K-means 邻居标签传播生成软标签，在四个骨干模型三个数据集上稳定提升。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/self_supervised_denoising/post_training_denoising_of_user_profiles_with_llms_in_collaborative_filteri/">Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation</a></strong></dt>
<dd>
<p>用 LLM 零样本/少样本提示在推理前清洗用户画像，无需重训 CF 模型即获最高 13% NDCG/HR 提升。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2025/llm_rec/darec_a_disentangled_alignment_framework_for_large_language_model_and_recom/">DaRec: A Disentangled Alignment Framework for Large Language Model and Recommender System</a></strong></dt>
<dd>
<p>从信息论证明零间隙对齐次优，提出解耦共享/特有表征+全局局部结构对齐的即插即用框架，全库排序下 Recall@20 最高提升 5.33%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/sequential_rec/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommenda/">Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义检索构建用户/物品候选池并配可学习样本合成器，替代随机增广与 K-Means 硬规则，全库排序下 HR@20 最高提升 11.82%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/can_recommender_systems_teach_themselves_a_recursive_self_improving_framewo/">Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control</a></strong></dt>
<dd>
<p>提出 RSIR 递归自我改进框架，用模型自生成序列+保真度质量控制稠密化稀疏交互，理论证明其等价于流形切向隐式正则，多骨干多数据集 Full-Ranking 下 NDCG/Recall 稳定提升 3%–14%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/gcib_graph_contrastive_information_bottleneck_for_multi_behavior_recommenda/">GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用图信息瓶颈在结构层剪枝辅助行为噪声边、用跨行为 InfoNCE 在特征层对齐目标语义，Leave-One-Out 全库排序下 NDCG@10 最高提升 37.2%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitatio/">Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy</a></strong></dt>
<dd>
<p>展开 GCF 预测为邻居对加权聚合，揭示 SSM 仅按物品侧相似度更新且忽略邻居对类型，提出类型感知 NT-SSM，NDCG@20 最高提升 21.45%。</p>
</dd>
<dt><strong><a href="../../venues/IJCAI/2025/sequential_rec/towards_automatic_sampling_of_user_behaviors_for_sequential_recommender_sys/">Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems</a></strong></dt>
<dd>
<p>提出 AutoSAM，用轻量采样器+强化学习多目标奖励自适应丢弃低信息历史行为，在四个真实数据集全库排序下 NDCG@10 平均提升约 8%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/sequential_rec/bridging_textual_collaborative_gap_through_semantic_codes_for_sequential_re/">Bridging Textual-Collaborative Gap through Semantic Codes for Sequential Recommendation</a></strong></dt>
<dd>
<p>用多视图文本量化语义码作桥梁，以码为 Query 交叉注意力融合文本，配合双重掩码自监督，Full-Ranking 下 R@10 最高提升 20.1%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/self_supervised_denoising/enhancing_graph_contrastive_learning_with_reliable_and_informative_augmenta/">Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation</a></strong></dt>
<dd>
<p>用端到端多级向量量化学习离散码，将码作为虚拟邻居与语义正样本构建三重对比视图，Full-Ranking 下 NDCG@10 最高提升 17.5%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/graph_rec/lightkg_efficient_knowledge_aware_recommendations_with_simplified_gnn_archi/">LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture</a></strong></dt>
<dd>
<p>用标量对编码关系+线性聚合+免子图对比层，在稀疏场景超越12个KGRS基线5.8%并省84.3%训练时间</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multimodal_rec/mdvt_enhancing_multimodal_recommendation_with_model_agnostic_multimodal_dri/">MDVT: Enhancing Multimodal Recommendation with Model-Agnostic Multimodal-Driven Virtual Triplets</a></strong></dt>
<dd>
<p>用多模态融合表征动态构造虚拟三元组作为额外监督信号，配合三种预热阈值策略与增强 pairwise 损失，即插即用缓解数据稀疏，稀疏用户组 NDCG@10 最高提升 12.85%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/self_supervised_denoising/omnisage_large_scale_multi_entity_heterogeneous_graph_representation_learni/">OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning</a></strong></dt>
<dd>
<p>统一异构图 GNN、内容特征与用户序列的多任务对比学习框架，支撑 Pinterest 十亿级实体通用表征，线上全站 Repin 提升约 2.5%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cold_start_rec/online_item_cold_start_recommendation_with_popularity_aware_meta_learning/">Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning</a></strong></dt>
<dd>
<p>按流行度固定切分元学习任务并配冷启动增强器，流式在线场景下冷启动召回相对 SOTA 提升 11%–74%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/self_supervised_denoising/shapley_value_driven_data_pruning_for_recommender_systems/">Shapley Value-driven Data Pruning for Recommender Systems</a></strong></dt>
<dd>
<p>用 FastSHAP 实时估计交互对训练损失的边际贡献，按 Shapley 值剪枝低效用交互并重训，四数据集 Recall/NDCG 平均提升 3.72%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cold_start_rec/denoising_implicit_feedback_for_cold_start_recommendation/">Denoising Implicit Feedback for Cold-start Recommendation</a></strong></dt>
<dd>
<p>用内容相似热物品协同表征为冷物品生成置信加权伪标签，并以相对熵×冷启动状态估计样本不确定性做自适应软标签纠正，快手十亿级 A/B 全指标提升。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/self_supervised_denoising/dynamic_spectral_denoising_with_global_context_attention_for_multi_behavior/">Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>沿特征维做 RFFT 学习视图自适应谱调制净化表征，再以净化全局锚做可靠性感知注意力融合，全库排序下 Taobao HR@10 相对提升 26.17%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/proex_a_unified_framework_leveraging_large_language_model_with_profile_extr/">ProEx: A Unified Framework Leveraging Large Language Model with Profile Extrapolation for Recommendation</a></strong></dt>
<dd>
<p>用四步 CoT 为每个用户/物品生成多面画像并做环境不变性融合，在三个数据集上对 6 个基模型平均提升 5%+。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multimodal_rec/vi_mmrec_similarity_aware_training_cost_free_virtual_user_item_interactions/">VI-MMRec: Similarity-Aware Training Cost-free Virtual User-Item Interactions for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用模态相似度构造虚拟交互矩阵并以统计权重融合，零训练开销即插即用地缓解多模态推荐数据稀疏，Full-Ranking 下 Recall@10 最高提升 22.25%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/decoupled_entity_representation_learning_for_pinterest_ads_ranking/">Decoupled Entity Representation Learning for Pinterest Ads Ranking</a></strong></dt>
<dd>
<p>提出解耦式上游-下游实体表征框架 DERM，三塔+对比自监督+滑动平均聚合，离线 AUC 提升 0.19%、线上 CTR +1.38%、CPA -1.61%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/nlgcl_naturally_existing_neighbor_layers_graph_contrastive_learning_for_rec/">NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用 GNN 相邻层天然异构节点作正样本对替代图增广，免视图构建，全库排序下 NDCG@10 最高提升 12.41% 且训练最快。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/sgcl_unifying_self_supervised_and_supervised_learning_for_graph_recommendat/">SGCL: Unifying Self-Supervised and Supervised Learning for Graph Recommendation</a></strong></dt>
<dd>
<p>将 BPR 与 InfoNCE 合并为单一监督图对比损失，免增广免负采样，精度与训练效率双超 SGL/SimGCL。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/distribution_level_contrastive_supervision_for_generative_recommendation/">Distribution-Level Contrastive Supervision for Generative Recommendation</a></strong></dt>
<dd>
<p>用分词器多层码本概率分布作软语义目标，以负对称KL+BPR对比监督对齐推荐器表征，即插即用提升多种生成式骨干。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/sequential_rec/data_augmentation_as_free_lunch_exploring_the_test_time_augmentation_for_se/">Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation</a></strong></dt>
<dd>
<p>将数据增强从训练期迁移到推理期，提出 TNoise 与 TMask 两个免训练、免参数的测试时增强算子，在 Full-Ranking 协议下平均提升 14%–30%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/disentangling_and_generating_modalities_for_recommendation_in_missing_modal/">Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios</a></strong></dt>
<dd>
<p>将模态特征解耦为通用/特有分量并用互信息损失分离，再以跨模态对齐特征与用户模态偏好生成缺失模态，缺失场景全库排序 R@20 相对提升 4.7%–8.2%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/llm_rec/intent_representation_learning_with_large_language_model_for_recommendation/">Intent Representation Learning with Large Language Model for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 思维链抽取文本意图，与交互意图双塔对齐+动量蒸馏匹配，作为模型无关插件在三个数据集上稳定提升 Recall/NDCG。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/sequential_rec/intent_aware_diffusion_with_contrastive_learning_for_sequential_recommendat/">Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 K-means 意图原型引导条件扩散生成意图对齐正视图替代随机增强，在 5 个数据集 Full-Ranking 下 NDCG 平均提升 20.68%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/social_group_rec/invariance_matters_empowering_social_recommendation_via_graph_invariant_lea/">Invariance Matters: Empowering Social Recommendation via Graph Invariant Learning</a></strong></dt>
<dd>
<p>用图不变学习替代社交图硬过滤：多环境生成器+不变风险最小化+对抗环境探索，Full-Ranking 下 NDCG@20 较 GBSR 提升 5.4%–8.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/self_supervised_denoising/squeeze_and_excitation_a_weighted_graph_contrastive_learning_for_collaborat/">Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>仅扰动末层GCN视图并用SENet动态加权特征，在三个数据集上实现最高24.45%的NDCG@20提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 学习用户自适应增强，四数据集全排序下 HR/NDCG 平均提升约 7-8%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/differentiable_semantic_id_for_generative_recommendation/">Differentiable Semantic ID for Generative Recommendation</a></strong></dt>
<dd>
<p>用 Gumbel 噪声+不确定性衰减实现可微语义ID联合训练，Leave-One-Out 全库排序下 R@10 相对 TIGER 提升约 12%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong></dt>
<dd>
<p>借鉴 ZPD 理论构建 Teacher-Peer-Self 多源负采样，用分歧重排提升多样性、共识蒸馏复用打分开销，六数据集全库排序下 Recall@20 最高提升 27.3%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong></dt>
<dd>
<p>识别地理约束下 in-batch 负样本中的“不可能负样本”问题，提出区域同质批采样 RCBS，仅改批构造即显著提升用户表征与线上指标。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结+可学习双网络的子模块组合模拟指数级集成多样性，推理仅用单网络，NDCG@20 最高提升 9.70% 且收敛快 7.69 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/from_raw_features_to_effective_embeddings_a_three_stage_approach_for_multim/">From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation</a></strong></dt>
<dd>
<p>用 VLM/LLM 摘要 + LightGCN 传播 + 跨视图对比学习三阶段精炼食谱多模态特征，Full-Ranking 下 Recall@10 提升 7–15%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/multi_modal_relational_item_representation_learning_for_inferring_substitut/">Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items</a></strong></dt>
<dd>
<p>融合多模态基础模型与元路径行为编码，以自监督对比去噪与 LLM 校验弱监督推断可替代/互补关系，M@10 平均提升 26.1%/39.2%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/qagcf_graph_collaborative_filtering_for_q_a_recommendation/">QAGCF: Graph Collaborative Filtering for Q&amp;A Recommendation</a></strong></dt>
<dd>
<p>将问答对解耦为用户-问题/用户-答案双协同图与问题-答案/问题-问题双语义图，融合为全局异质图后用 Jacobi 多项式图滤波器分离高低频信号，配合噪声对比学习，在知乎与商业数据集 Full-Ranking 下 Recall@10 提升 5.16%–5.34%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/poi_spatiotemporal_rec/rest_a_plug_and_play_spatially_constrained_representation_enhancement_frame/">ReST: A Plug-and-Play Spatially-Constrained Representation Enhancement Framework for Local-Life Recommendation</a></strong></dt>
<dd>
<p>以物品为中心，用属性语义预热+空间约束难负采样对比学习增强长尾物品ID表征，离线AUC最高+5.90%，线上GMV+2.804%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cold_start_rec/sparse_contrastive_learning_for_content_based_cold_item_recommendation/">Sparse Contrastive Learning for Content-Based Cold Item Recommendation</a></strong></dt>
<dd>
<p>抛弃 CF 对齐、纯内容空间 item-item 相似度建模，用 α-entmax 稀疏化 sampled softmax 并支持内容空间蒸馏，冷物品排序与曝光公平性双赢。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong></dt>
<dd>
<p>TIDE 用 Hawkes 增强傅里叶时间编码刻画物品级非单调补货周期，并以双专家解耦习惯与探索意图，Leave-One-Out 全库排序下 R@10 最高提升 18.4%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/dual_test_time_training_for_out_of_distribution_recommender_system/">Dual Test-time Training for Out-of-distribution Recommender System</a></strong></dt>
<dd>
<p>首个将测试时训练引入 OOD 推荐，用自蒸馏+对比双任务在测试阶段适配用户/物品特征漂移，全库排序下 R@50 最高提升 11.56%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/graph_rec/dual_channel_multiplex_graph_neural_networks_for_recommendation/">Dual-Channel Multiplex Graph Neural Networks for Recommendation</a></strong></dt>
<dd>
<p>双通道多路图网络显式建模行为模式、隐式挖掘关系链依赖，三数据集全指标平均提升 10%+</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/llm_rec/enhancing_collaborative_semantics_of_language_model_driven_recommendations_/">Enhancing Collaborative Semantics of Language Model-Driven Recommendations via Graph-Aware Learning</a></strong></dt>
<dd>
<p>用多跳提示+图感知对比学习把 GNN 聚合意图注入 LLM，在三个 Amazon/Yelp 数据集 Full-Ranking 下 HR@10 最高提升约 15%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/fairdgcl_fairness_aware_recommendation_with_dynamic_graph_contrastive_learn/">FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning</a></strong></dt>
<dd>
<p>用对抗式视图生成器+判别器动态生成公平增强视图，在四个数据集上同时提升准确率与群体公平性。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/sequential_rec/improving_sequential_recommendations_via_bidirectional_temporal_data_augmen/">Improving Sequential Recommendations via Bidirectional Temporal Data Augmentation with Pre-training</a></strong></dt>
<dd>
<p>BARec 用双向时序增强预训练生成保偏好伪先验物品，并以 KL 知识增强微调对齐原/增广序列分布，全库排序下 Beauty Recall@5 提升 16.6%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/collaborative_filtering/intent_propagation_contrastive_collaborative_filtering/">Intent Propagation Contrastive Collaborative Filtering</a></strong></dt>
<dd>
<p>双螺旋消息传播融合高阶关系与意图解耦，对比学习对齐结构/意图视图，Full-Ranking 下 NDCG 平均提升 9.36%</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/graph_rec/intent_guided_heterogeneous_graph_contrastive_learning_for_recommendation/">Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用元路径意图构建双对比学习视图，并以信息瓶颈掩码自编码器去噪，六数据集全库排序 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/collaborative_filtering/latent_factor_modeling_with_expert_network_for_multi_behavior_recommendatio/">Latent Factor Modeling with Expert Network for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用门控专家网络显式建模共享潜在因子，配合对比学习保证因子独立与一致，在三个多行为数据集 Full-Ranking 下 HR@10 最高提升 3.95%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/sequential_rec/pattern_wise_transparent_sequential_recommendation/">Pattern-wise Transparent Sequential Recommendation</a></strong></dt>
<dd>
<p>用多尺度滑窗切出点级/联合级模式，以概率嵌入+逻辑合取建模模式，KL散度+距离权重实现可解释且超越黑盒的序列推荐。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/social_group_rec/score_based_generative_diffusion_models_for_social_recommendations/">Score-based Generative Diffusion Models for Social Recommendations</a></strong></dt>
<dd>
<p>用 SDE 扩散模型在隐空间直接生成协同一致的社交表征，配合课程训练与跨域对比学习，Full-Ranking 下 Recall@10 相对提升 1.85%–3.07%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/towards_robust_recommendation_a_review_and_an_adversarial_robustness_evalua/">Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library</a></strong></dt>
<dd>
<p>系统综述推荐系统对抗/非对抗鲁棒性，提出 ShillingREC 评测库统一攻击与防御模型评估。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
</dd>
<dt><strong><a href="../../venues/TMM/2025/multimodal_rec/knowledge_aware_diffusion_enhanced_multimedia_recommendation/">Knowledge-aware Diffusion-Enhanced Multimedia Recommendation</a></strong></dt>
<dd>
<p>用重启随机游走 Jaccard 注意力矩阵替代可学习注意力，并以用户嵌入引导扩散模型重建低噪知识图谱构建对比视图，三数据集 Recall@20 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/TMM/2026/multimodal_rec/cross_modal_attention_network_with_dual_graph_learning_in_multimodal_recomm/">Cross-Modal Attention Network with Dual Graph Learning in Multimodal Recommendation</a></strong></dt>
<dd>
<p>递归跨模态注意力 RCA 迭代对齐图文特征，配合用户-物品异构图与物品-物品语义图双图 + 对比学习，四数据集平均提升约 5%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/cross_domain_multi_scenario/coherence_guided_preference_disentanglement_for_cross_domain_recommendation/">Coherence-guided Preference Disentanglement for Cross-domain Recommendations</a></strong></dt>
<dd>
<p>用全局-局部图 + 一致性约束抽取跨域共享物品属性，并以流行度加权 Softmax 将共享用户偏好解耦为兴趣/从众，六组双目标 CDR 任务 HR@10 最高提升约 37%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/sequential_rec/learning_robust_sequential_recommenders_through_confident_soft_labels/">Learning Robust Sequential Recommenders through Confident Soft Labels</a></strong></dt>
<dd>
<p>用模型级/数据级/训练级三种教师模块生成高置信软标签，与 one-hot 联合训练学生序列推荐器，全库排序下 Recall@10 最高提升约 11%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/self_supervised_denoising/symmetric_graph_contrastive_learning_against_noisy_views_for_recommendation/">Symmetric Graph Contrastive Learning against Noisy Views for Recommendation</a></strong></dt>
<dd>
<p>定义图增广产生的噪声视图，用对称指数对比损失替代 InfoNCE，使 GCL 对噪声视图鲁棒，全库排序下 NDCG@20 最高提升 12.25%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/multi_task_reranking/user_invariant_preference_learning_for_multi_behavior_recommendation/">User Invariant Preference Learning for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用 IRM 范式 + VAE 从多行为交互中抽取用户不变偏好以抑制辅助行为噪声，Leave-One-Out + Full-Ranking 下 HR@10 最高提升 3.98%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/graph_rec/hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/collaborative_filtering/balancing_embedding_spectrum_for_recommendation/">Balancing Embedding Spectrum for Recommendation</a></strong></dt>
<dd>
<p>从谱分析视角揭示成对学习中的嵌入坍缩，提出 DirectSpec 全通滤波直接均衡奇异值谱，MF/LightGCN 上 nDCG@10 最高提升 52.6%/42.4%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/cold_start_rec/data_scarcity_in_recommendation_systems_a_survey/">Data Scarcity in Recommendation Systems: A Survey</a></strong></dt>
<dd>
<p>系统梳理数据稀缺下五类缓解范式（增广/自监督/迁移/广度学习/知识图谱），并给出挑战与未来方向。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/collaborative_filtering/enhancing_item_level_bundle_representation_for_bundle_recommendation/">Enhancing Item-level Bundle Representation for Bundle Recommendation</a></strong></dt>
<dd>
<p>用 B-U-I 高阶关联与预训练伪交互增强捆绑包物品级表征，全库排序下平均提升 7.5%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/self_supervised_denoising/a_contrastive_framework_with_user_item_and_review_alignment_for_recommendat/">A Contrastive Framework with User, Item and Review Alignment for Recommendation</a></strong></dt>
<dd>
<p>将评论从“特征”升级为“对比信号”，用评论增广+三方对齐双对比任务缓解稀疏与评论缺失，Full-Ranking 下 NDCG@5 最高提升 7.7%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/graph_rec/lightgnn_simple_graph_neural_network_for_recommendation/">LightGNN: Simple Graph Neural Network for Recommendation</a></strong></dt>
<dd>
<p>用可学习边/嵌入剪枝+分层知识蒸馏压缩 GNN 推荐模型，80% 边与 90% 嵌入裁剪下仍超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/graph_rec/mixrec_heterogeneous_graph_collaborative_filtering/">MixRec: Heterogeneous Graph Collaborative Filtering</a></strong></dt>
<dd>
<p>参数化异质超图解耦多行为意图，配合节点/图级自适应对比学习，Leave-One-Out 全库排序下 HR@10 最高提升 9.8%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/sequential_rec/oracle_guided_dynamic_user_preference_modeling_for_sequential_recommendatio/">Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation</a></strong></dt>
<dd>
<p>用未来交互作为后验“Oracle”引导过去编码器学习前瞻表征，配无参低通滤波去噪与两阶段训练，六数据集采样排序全面领先。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/sequential_rec/personalised_outfit_recommendation_via_history_aware_transformers/">Personalised Outfit Recommendation via History-aware Transformers</a></strong></dt>
<dd>
<p>双层 Transformer 堆叠建模套装内部兼容性与用户购买史，配合监督对比与自适应间隔损失实现个性化套装打分，CP-Hard AUC 提升 15.7%/19.4%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/self_supervised_denoising/scone_a_novel_stochastic_sampling_to_generate_contrastive_views_and_hard_ne/">SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation</a></strong></dt>
<dd>
<p>用分数生成模型统一生成用户-物品特定对比视图与难负样本，6 数据集全库排序 Recall@20 最高提升 1.70%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/multimodal_rec/teach_me_how_to_denoise_a_universal_framework_for_denoising_multi_modal_rec/">Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration</a></strong></dt>
<dd>
<p>揭示 ID 空间去噪假设在多模态空间失效，提出 AMSC 重校准 + DBPR + OT 蒸馏的即插即用去噪框架，全库排序下 Recall@20 最高提升约 11%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/sequential_rec/your_causal_self_attentive_recommender_hosts_a_lonely_neighborhood/">Your Causal Self-Attentive Recommender Hosts a Lonely Neighborhood</a></strong></dt>
<dd>
<p>从稀疏局部归纳偏置与低秩近似两角度理论剖析 AE/AR 自注意力，并在 5 个基准上以 ModSAR 流水线实证 AR 全面占优。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/sequential_rec/capturing_user_interests_from_data_streams_for_continual_sequential_recomme/">Capturing User Interests from Data Streams for Continual Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性注意力参数化记忆替代回放/正则，配合 Cauchy-Schwarz 归一化与共享兴趣池，在流式数据块上兼顾历史保留与新兴趣获取。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/generative_rec/continuous_time_discrete_space_diffusion_model_for_recommendation/">Continuous-time Discrete-space Diffusion Model for Recommendation</a></strong></dt>
<dd>
<p>将推荐历史交互建模为连续时间离散空间（吸收态掩码）扩散，配合流行度感知噪声调度、一致性参数化与多跳对比引导，在三个数据集 Full-Ranking 下 Recall@10 相对提升 4.2%–5.2%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/self_supervised_denoising/simdiffrec_semantic_similarity_guided_diffusion_for_contrastive_sequential_/">SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义相似物品均值替代高斯随机噪声做扩散增强，按去噪置信度选增强位置并配难负采样，五数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/sequential_rec/time_aware_adaptive_side_information_fusion_for_sequential_recommendation/">Time-Aware Adaptive Side Information Fusion for Sequential Recommendation</a></strong></dt>
<dd>
<p>提出 TSP 时间跨度划分 + AFF 自适应频域滤波 + ASIF「引导不混合」融合层，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 10.28%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/self_supervised_denoising/mixrec_individual_and_collective_mixing_empowers_data_augmentation_for_reco/">MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems</a></strong></dt>
<dd>
<p>用 Beta/Dirichlet 混合在嵌入空间零成本造正负视图，配合双混合对比损失，全库排序下 Recall@20 最高提升 5.05%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/self_supervised_denoising/personalized_denoising_implicit_feedback_for_robust_recommender_system/">Personalized Denoising Implicit Feedback for Robust Recommender System</a></strong></dt>
<dd>
<p>揭示全局损失分布中正常/噪声交互严重重叠，提出基于用户个人损失分布的重采样去噪 PLD，BPR 下 Recall@20 最高提升 7.36%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/rallrec_improving_retrieval_augmented_large_language_model_recommendation_w/">RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning</a></strong></dt>
<dd>
<p>用 LLM 生成物品详述并做文本-协同表征对比对齐，配合时间感知重排器提升 RAG 式 LLM 推荐的检索质量。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/separated_contrastive_learning_for_matching_in_cross_domain_recommendation_/">Separated Contrastive Learning for Matching in Cross-domain Recommendation with Curriculum Scheduling</a></strong></dt>
<dd>
<p>将 intra-CL 与 inter-CL 解耦为两阶段课程训练，配合 stop-gradient 与基于 Katz 中心性的负样本难度调度，稳定跨域召回训练并显著提升 HIT@100。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/sequential_rec/td3_tucker_decomposition_based_dataset_distillation_method_for_sequential_r/">TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 Tucker 分解把合成序列摘要解耦为用户/时间/共享物品因子与关系核，配合特征空间对齐双层优化，在 0.2%–7.4% 数据量下逼近甚至超越全量训练。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/self_supervised_denoising/unleashing_the_power_of_large_language_model_for_denoising_recommendation/">Unleashing the Power of Large Language Model for Denoising Recommendation</a></strong></dt>
<dd>
<p>LLaRD 用 LLM 生成偏好知识与图 CoT 关系知识，再以信息瓶颈对齐去噪图，Full-Ranking 下 Recall@20 最高提升约 14%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/alphafree_recommendation_free_from_users_ids_and_gnns/">AlphaFree: Recommendation Free from Users, IDs, and GNNs</a></strong></dt>
<dd>
<p>用语言表征替代ID、在线推断用户偏好、以相似物品增强+对比对齐替代GNN，实现无用户/无ID/无GNN的轻量推荐，内存降69%、冷启动提升26–38%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/beyond_isolated_clients_integrating_graph_based_embeddings_into_event_seque/">Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models</a></strong></dt>
<dd>
<p>将二部交互图结构以三种模型无关策略注入事件序列对比 SSL，揭示图密度决定最优融合策略，AUC 最高提升 2.3%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/diversity_augmented_negative_sampling_for_implicit_collaborative_filtering/">Diversity-Augmented Negative Sampling for Implicit Collaborative Filtering</a></strong></dt>
<dd>
<p>用用户级缓存 + 多样性增强 k-DPP + mixup 合成负样本，解决两阶段负采样同质化问题，Full-Ranking 下 NDCG@20 最高提升 8.76%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/efficient_content_based_recommendation_model_training_via_noise_aware_cores/">Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection</a></strong></dt>
<dd>
<p>用末层梯度子模优化选核集，配合渐进标签自校正与 MC-Dropout 去噪，1% 数据恢复 93-95% 全量性能。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/mclmr_a_model_agnostic_causal_learning_framework_for_multi_behavior_recomme/">MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用后门干预+MoE自适应聚合+偏差感知对比学习，作为即插即用框架增强多行为推荐，三数据集全库排序显著提升。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/social_group_rec/pulse_socially_aware_user_representation_modeling_toward_parameter_efficien/">PULSE: Socially-Aware User Representation Modeling Toward Parameter-Efficient Graph Collaborative Filtering</a></strong></dt>
<dd>
<p>用社区结构与社交邻居物品信号替代显式用户嵌入，参数降 50% 且 NDCG@20 提升 8-9%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/radar_relation_aware_diffusion_asymmetric_graph_contrastive_learning_for_re/">RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>以 VGAE 生成视图与关系感知去噪视图构成双视图，叠加扩散引导增广与非对称对比目标，在 Full-Ranking 下 Recall@20 相对 SOTA 提升 3.8%–4.7%。</p>
</dd>
</dl>

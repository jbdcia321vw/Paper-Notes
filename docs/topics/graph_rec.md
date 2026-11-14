# 🕸️ 图神经网络与知识图谱推荐 (Graph & Knowledge-Aware Recommendation)

> **研究范畴**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

当前方向共收录 **122** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/cross_domain_multi_scenario/behavior_importance_aware_graph_neural_architecture_search_for_cross_domain/">Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用一次性免重训超网为各域定制 GNN 架构，并以双层隐式梯度学习源域行为重要性权重，Amazon 与微信广告数据 AUC 最高提升 10.57%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/beyond_graph_convolution_multimodal_recommendation_with_topology_aware_mlps/">Beyond Graph Convolution: Multimodal Recommendation with Topology-aware MLPs</a></strong></dt>
<dd>
<p>用拓扑剪枝+邻域对齐损失替代GCN消息传递，以MLP建模多模态物品关系，全库排序下Recall@10最高提升7.53%</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/llm_rec/bridging_the_user_side_knowledge_gap_in_knowledge_aware_recommendations_wit/">Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 推断用户兴趣并结构化为 CIKG 异构图，配合兴趣重建与跨域对比学习去噪，在三个数据集全库排序下 R@100 最高提升 9.31%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/exploiting_fine_grained_skip_behaviors_for_micro_video_recommendation/">Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation</a></strong></dt>
<dd>
<p>将短视频跳过行为细分为高正/低正/负三级，构建双层正图与分层 BPR 损失，两数据集八项指标全面超越 FRAME 等基线。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/generative_rec/modeling_item_level_dynamic_variability_with_residual_diffusion_for_bundle_/">Modeling Item-Level Dynamic Variability with Residual Diffusion for Bundle Recommendation</a></strong></dt>
<dd>
<p>用残差扩散在潜空间重建捆绑包item级表征，模型无关地适配B-I归属动态变化，Recall/NDCG最高提升约23%，训练开销仅增4%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/graph_rec/task_aware_retrieval_augmentation_for_dynamic_recommendation/">Task-Aware Retrieval Augmentation for Dynamic Recommendation</a></strong></dt>
<dd>
<p>用任务感知评估机制自动标注历史子图收益，并以图 Transformer 打分检索融合，缓解动态推荐预训练-微调时序泛化鸿沟。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/kerl_knowledge_enhanced_personalized_recipe_recommendation_using_large_lang/">KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models</a></strong></dt>
<dd>
<p>用 FoodKG 子图作上下文 + 多 LoRA 适配器，统一完成约束食谱推荐、烹饪步骤生成与微量营养素估计，KGQA 基准 F1 达 0.973。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/knowledge_graph_retrieval_augmented_generation_for_llm_based_recommendation/">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用跳场知识子图索引+流行度选择性检索+GNN软提示，将KG结构知识注入冻结LLM，MovieLens-1M ACC 相对提升58.6%且推理仅慢0.14s</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/graph_rec/continual_graph_memory_for_adaptive_recommendation_under_intent_drift/">Continual Graph Memory for Adaptive Recommendation under Intent Drift</a></strong></dt>
<dd>
<p>将知识图谱视为可写双记忆（语义图记忆+情景教训记忆），冻结参数下仅靠质量门控图编辑实现意图漂移自适应重排。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/graphlora_structure_aware_low_rank_adaptation_for_large_language_model_reco/">GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation</a></strong></dt>
<dd>
<p>将可训练 GNN 消息传递嵌入 LoRA 瓶颈（A 与 B 之间），让协同拓扑在参数空间传播；在 ML-1M/Amazon-Book 上以 r=8 的极小参数预算超越 CoRA 等 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/memrec_collaborative_memory_augmented_agentic_recommender_system/">MemRec: Collaborative Memory-Augmented Agentic Recommender System</a></strong></dt>
<dd>
<p>将记忆管理从推理中架构解耦，用轻量 LM_Mem 在协同记忆图上做零样本规则剪枝与异步传播，仅向重型 LLM_Rec 注入蒸馏后的高信号协同上下文，四基准全库排序 H@1 最高 +28.98%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/dual_phase_playtime_guided_recommendation_interest_intensity_exploration_an/">Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks</a></strong></dt>
<dd>
<p>用双 Beta 分布解耦游玩时长强弱兴趣，并以多模态随机游走做跨类目探索，在 Steam 数据上同时提升准确率与覆盖度。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/one_graph_multiple_gains_single_high_quality_item_item_graph_for_multimodal/">One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation</a></strong></dt>
<dd>
<p>构建一次高质量物品-物品图，经三元闭包重加权后复用于表征、交互图与优化三阶段，四数据集全指标SOTA且训练更快更省显存。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/collaborative_filtering/a_node_aware_dynamic_quantization_approach_for_graph_collaborative_filterin/">A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering</a></strong></dt>
<dd>
<p>提出节点感知动态量化 GNAQ，用参数共享学习每节点量化步长并以图消息传递更新，2-bit 下 Recall@10 平均提升 27.8%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/benefit_from_rich_tackling_search_interaction_sparsity_in_search_enhanced_r/">Benefit from Rich: Tackling Search Interaction Sparsity in Search Enhanced Recommendation</a></strong></dt>
<dd>
<p>用 LLM+RQ-VAE 将用户搜索/推荐偏好离散为码本，构建 User-Code 图做消息传递，把搜索富用户信号传给稀疏用户，Qilin 上 HR@1 相对 UniSAR 提升 12.1%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/causality_aware_graph_aggregation_weight_estimator_for_popularity_debiasing/">Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation</a></strong></dt>
<dd>
<p>将图聚合证明为后门调整，用 VAE 式编码器-解码器估计无偏聚合权重，配合动量更新，小众物品 Recall@20 提升约 20%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/social_group_rec/community_aware_social_community_recommendation/">Community-Aware Social Community Recommendation</a></strong></dt>
<dd>
<p>提出 CASO，用模块度最大化、社交紧密度聚合与协同编码三路解耦用户表征，并以 HSIC 互斥与 KL 社区检测损失优化，在 6 个真实社交网络上 Full-Ranking 全面超越 9 个基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/sequential_rec/cpmr_context_aware_incremental_sequential_recommendation_with_pseudo_multi_/">CPMR: Context-Aware Incremental Sequential Recommendation with Pseudo-Multi-Task Learning</a></strong></dt>
<dd>
<p>提出伪多任务学习范式，用历史图与上下文图双通道演化用户/物品时序状态，在四个数据集上 MRR 平均提升 30.98%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/graph_rec/enhancing_graph_collaborative_filtering_with_fourierkan_feature_transformat/">Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation</a></strong></dt>
<dd>
<p>用 Fourier KAN 替代 NGCF 中被误删的交互特征变换 W2，在 7:1:2 时间切分全库排序下 R@20 最高提升 7.8%，且可作即插即用骨干。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/generative_data_augmentation_in_graph_contrastive_learning_for_recommendati/">Generative Data Augmentation in Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用 VAE 式生成器自适应生成噪声替代固定随机扰动构造对比视图，并引入物品互补矩阵，在三个稀疏数据集上 Full-Ranking 提升约 3–6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/hypercomplex_prompt_aware_multimodal_recommendation/">Hypercomplex Prompt-aware Multimodal Recommendation</a></strong></dt>
<dd>
<p>用 Cayley-Dickson 超复数多分量嵌入 + 超复数乘法非线性融合 + Prompt 动态补偿，在四个 Amazon 数据集 Full-Ranking 下 Recall@20 最高提升约 13%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/improving_recommendation_fairness_via_graph_structure_and_representation_au/">Improving Recommendation Fairness via Graph Structure and Representation Augmentation</a></strong></dt>
<dd>
<p>用双先验假设定位敏感边与敏感特征，做图剪枝+特征掩码双增广，HSIC去偏+对比学习保效用，ML-1M 上 DP@10 降 3.4%、EO@10 降 7.6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/graph_rec/spark_adaptive_low_rank_knowledge_graph_modeling_in_hybrid_geometric_spaces/">SPARK: Adaptive Low-Rank Knowledge Graph Modeling in Hybrid Geometric Spaces for Recommendation</a></strong></dt>
<dd>
<p>用 Tucker 低秩去噪 KG、SVD 初始化双曲/欧氏并行 GNN，并按物品流行度自适应融合多源信号，长尾推荐 R@10 最高提升 10.8%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/graph_rec/ultra_fast_warm_start_solution_for_graph_recommendations/">Ultra Fast Warm Start Solution for Graph Recommendations</a></strong></dt>
<dd>
<p>将 SVD Folding-In 的闭式解迁移到 UltraGCN，用一次矩阵-向量乘精确更新热用户嵌入，比 SGD 微调快 30 倍且质量更优、覆盖率更高。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/graph_rec/beyond_observed_auxiliary_relations_environment_conditioned_modeling_for_mu/">Beyond Observed Auxiliary Relations: Environment-Conditioned Modeling for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>以辅助行为可观测性为环境条件，用去偏稠密化与目标引导剪枝双模块重连多行为图，隐藏目标物品 HR@10 最高提升 44.2%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/graph_rec/do_all_nodes_benefit_equally_from_knowledge_graphs_adaptive_node_aware_kg_f/">Do All Nodes Benefit Equally from Knowledge Graphs? Adaptive Node-Aware KG Fusion for Recommendation</a></strong></dt>
<dd>
<p>以对抗扰动估计节点级 CF 不稳定性，自适应门控 KG 注入强度并配合自适应对比对齐，四数据集 Full-Ranking 平均排名第一。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/enriching_sequential_recommendation_with_graph_laplacian_positional_embeddi/">Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings</a></strong></dt>
<dd>
<p>用物品共现图归一化拉普拉斯特征向量冻结替换 SASRec 可学习位置嵌入，四基准多数指标提升且在线开销近零。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/graph_rec/neural_tree_collaborative_filtering_rethinking_graph_collaborative_filterin/">Neural Tree Collaborative Filtering: Rethinking Graph Collaborative Filtering as Tree Collaborative Filtering with Curvature-Aware Propagation Depth</a></strong></dt>
<dd>
<p>以局部度不平衡曲率代理为每节点分配自适应传播深度，将 GCF 重释为节点根树集成，Full-Ranking 下 R@20 稳定超越 LightGCN 与 SSL 骨干。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/social_group_rec/scaling_graph_neural_networks_for_friend_recommendation_multi_hash_user_emb/">Scaling Graph Neural Networks for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling</a></strong></dt>
<dd>
<p>用多哈希ID嵌入(表压缩&gt;98%)与时间戳排序CSR二分采样(O(log d+K))支撑194M用户/28B边GNN好友排序，线上好友添加+16%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/sprig_semantic_id_enhanced_paths_for_knowledge_graph_based_generative_recom/">SPRIG: Semantic-ID-enhanced Paths for Knowledge Graph-based Generative Recommendation</a></strong></dt>
<dd>
<p>将内容派生的分层语义 ID 注入知识图谱路径语言建模，以 768 个共享码替代逐物品词表，nDCG@10 较 KGGLM 提升 44%/58% 且参数与算力更低。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/cross_domain_multi_scenario/a_unified_framework_for_adaptive_representation_enhancement_and_inversed_le/">A Unified Framework for Adaptive Representation Enhancement and Inversed Learning in Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>解耦共享/专属表征，用 LightGCN 域内增强 + 自注意力跨域自适应融合 + GRL 逆向约束，全库排序下 Recall@20 最高提升 8.31%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/graph_rec/genet_unleashing_the_power_of_side_information_for_recommendation_via_hyper/">GENET: Unleashing the Power of Side Information for Recommendation via Hypergraph Pre-training</a></strong></dt>
<dd>
<p>将异构边信息统一构建为超图，通过超边预测与双层自监督对比预训练，再以 LightGCN/GRU 微调，Top-N 与序列推荐最高提升 38%，冷启动提升 175%+。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/okra_an_explainable_heterogeneous_multi_stakeholder_job_recommender_system/">OKRA: an Explainable, Heterogeneous, Multi-Stakeholder Job Recommender System</a></strong></dt>
<dd>
<p>异构知识图谱+多头GATv2构建求职者/企业双侧嵌入，谐波均值融合打分，nDCG@10较六基线最高提升约18%，并揭示城乡地域偏差。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2025/cold_start_rec/all_in_one_heterogeneous_interaction_modeling_for_cold_start_rating_predict/">All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction</a></strong></dt>
<dd>
<p>HIRE 用纯 MHSA 在采样的预测上下文上端到端学习用户/物品/属性三层异构交互，三种冷启动场景下 Precision/NDCG/MAP 平均超 12 个基线 0.21/0.29/0.22。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2025/llm_rec/making_large_language_models_better_knowledge_miners_for_online_marketing_w/">Making Large Language Models Better Knowledge Miners for Online Marketing with Progressive Prompting Augmentation</a></strong></dt>
<dd>
<p>用渐进式提示增强把 LLM 变成可控的营销知识图谱挖掘器，并以教师蒸馏出可部署的 LightPAIR，受众定向覆盖提升 15%–122%。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2025/graph_rec/path_based_summary_explanations_for_graph_recommenders_extended_version/">Path-based summary explanations for graph recommenders (extended version)</a></strong></dt>
<dd>
<p>首次提出路径解释的摘要化：用 Steiner Tree 与 Prize-Collecting Steiner Tree 将多条用户-物品解释路径压缩为连通子图，在可理解性、多样性、隐私与可扩展性上全面优于逐条路径基线。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/graph_rec/contextgnn_beyond_two_tower_recommendation_systems/">ContextGNN: Beyond Two-Tower Recommendation Systems</a></strong></dt>
<dd>
<p>提出局部成对表征+全局双塔回退的混合GNN架构，在RelBench十个任务上平均MAP较最强成对基线提升20%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/spot_trip_dual_preference_driven_out_of_town_trip_recommendation/">SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation</a></strong></dt>
<dd>
<p>首次形式化异地行程推荐，用知识图谱静态偏好+Neural ODE动态偏好解耦建模，F1最高提升17.01%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multimodal_rec/can_complementary_signals_bridge_similarity_islands_manifold_augmented_grap/">Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 MAGE，用离线 Soft Conceptor 残差注入 + 在线非对称双流 GNN 桥接互补品“相似性孤岛”，四个 Amazon 数据集 Full-Ranking 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/gcib_graph_contrastive_information_bottleneck_for_multi_behavior_recommenda/">GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用图信息瓶颈在结构层剪枝辅助行为噪声边、用跨行为 InfoNCE 在特征层对齐目标语义，Leave-One-Out 全库排序下 NDCG@10 最高提升 37.2%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/harp_training_free_dual_profile_agentic_communication_for_llm_based_recomme/">HARP: Training-Free Dual-Profile Agentic Communication for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 LLM 推荐重构为四阶段智能体通信，用双画像、同质性门控检索与单次反思三个免训练算子补齐缺失环节，Hit@1 相对提升 6.9%–24.0%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/low_pass_personalized_subgraph_federated_recommendation/">Low-pass Personalized Subgraph Federated Recommendation</a></strong></dt>
<dd>
<p>用低通谱滤波提取跨子图稳定的结构信号做个性化联邦聚合，并以局部流行度偏差感知 margin 抑制长尾偏置，五数据集 NDCG@20 最高提升 24.2%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitatio/">Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy</a></strong></dt>
<dd>
<p>展开 GCF 预测为邻居对加权聚合，揭示 SSM 仅按物品侧相似度更新且忽略邻居对类型，提出类型感知 NT-SSM，NDCG@20 最高提升 21.45%。</p>
</dd>
<dt><strong><a href="../../venues/IJCAI/2025/debias_fairness_industrial/single_node_trigger_backdoor_attacks_in_graph_based_recommendation_systems/">Single-Node Trigger Backdoor Attacks in Graph-Based Recommendation Systems</a></strong></dt>
<dd>
<p>提出单节点触发器后门攻击 SNT-BA，仅注入 1 个虚假用户即可让目标物品在 99% 目标用户 Top-10 中曝光，同时 HR@20 仅降约 5%。</p>
</dd>
<dt><strong><a href="../../venues/IJCAI/2026/rl_bandit_rec/pace_a_personalized_adaptive_curriculum_engine_for_9_1_1_call_taker_trainin/">PACE: A Personalized Adaptive Curriculum Engine for 9-1-1 Call-taker Training</a></strong></dt>
<dd>
<p>用技能图信念传播+个体学习/遗忘动力学+Thompson Sampling 上下文老虎机为 9-1-1 调度员推荐训练场景，Z2H 缩短 19.50%、终局掌握度提升 10.95%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/graph_rec/an_automatic_graph_construction_framework_based_on_large_language_models_fo/">An Automatic Graph Construction Framework based on Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 逐点生成语义向量 + 残差量化抽取隐因子作额外节点，O(N) 构图并设计元路径聚合，Full-Ranking 下 NDCG@10 最高提升 43.57%，华为广告 A/B 提升 RPM 2.69%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/graph_rec/collaborative_filtering_meets_spectrum_shift_connecting_user_item_interacti/">Collaborative Filtering Meets Spectrum Shift: Connecting User-Item Interaction with Graph-Structured Side Information</a></strong></dt>
<dd>
<p>揭示增广图频谱右移现象，提出仅含平移/缩放两因子的 SSC 校正，即插即用提升 LightGCN/JGCF 最高 23%，零额外开销。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/combinatorial_optimization_perspective_based_framework_for_multi_behavior_r/">Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation</a></strong></dt>
<dd>
<p>将多行为融合视为组合优化问题施加三阶段约束，并用分布拟合多专家网络抑制负迁移，三数据集 HR@10 最高提升 49.91%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/self_supervised_denoising/enhancing_graph_contrastive_learning_with_reliable_and_informative_augmenta/">Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation</a></strong></dt>
<dd>
<p>用端到端多级向量量化学习离散码，将码作为虚拟邻居与语义正样本构建三重对比视图，Full-Ranking 下 NDCG@10 最高提升 17.5%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/gigl_large_scale_graph_neural_networks_at_snapchat/">GiGL: Large-Scale Graph Neural Networks at Snapchat</a></strong></dt>
<dd>
<p>Snapchat 开源十亿级图学习库 GiGL，以表格化子图采样+PyG 解耦实现 900M 节点/16.8B 边图 12 小时内端到端训练推理，支撑 35+ 线上发布。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/graph_rec/lightkg_efficient_knowledge_aware_recommendations_with_simplified_gnn_archi/">LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture</a></strong></dt>
<dd>
<p>用标量对编码关系+线性聚合+免子图对比层，在稀疏场景超越12个KGRS基线5.8%并省84.3%训练时间</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/mopi_hfrs_a_multi_objective_personalized_health_aware_food_recommendation_s/">MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation</a></strong></dt>
<dd>
<p>构建首个融合 NHANES 医疗数据的健康感知食物推荐基准，用符号图结构学习 + Pareto 多目标优化平衡偏好/个性化健康/营养多样性，并以知识注入 LLM 生成可解释理由。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/sequential_rec/multi_grained_preference_enhanced_transformer_for_multi_behavior_sequential/">Multi-Grained Preference Enhanced Transformer for Multi-Behavior Sequential Recommendation</a></strong></dt>
<dd>
<p>M-GPT 用交互级图卷积建模跨类型行为依赖，并以多粒度多尺度注意力抽取行为感知偏好，Leave-One-Out 全库排序下 HR@5 相对提升 12.2%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/self_supervised_denoising/omnisage_large_scale_multi_entity_heterogeneous_graph_representation_learni/">OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning</a></strong></dt>
<dd>
<p>统一异构图 GNN、内容特征与用户序列的多任务对比学习框架，支撑 Pinterest 十亿级实体通用表征，线上全站 Repin 提升约 2.5%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/sequential_rec/time_matters_enhancing_sequential_recommendations_with_time_guided_graph_ne/">Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs</a></strong></dt>
<dd>
<p>用时间引导扩散补全稀疏用户时间图、图神经ODE对齐用户兴趣与物品演化，五数据集Recall@20最高提升46%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/copersona_collaborative_persona_graphs_for_robust_llm_personalization/">CoPersona: Collaborative Persona Graphs for Robust LLM Personalization</a></strong></dt>
<dd>
<p>用方面级多重人物图+双分支推理补全稀疏用户画像，Amazon六域生成式评测全面超越DEP/DPL。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/self_supervised_denoising/dynamic_spectral_denoising_with_global_context_attention_for_multi_behavior/">Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>沿特征维做 RFFT 学习视图自适应谱调制净化表征，再以净化全局锚做可靠性感知注意力融合，全库排序下 Taobao HR@10 相对提升 26.17%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/enriching_semantic_profiles_into_knowledge_graph_for_recommender_systems_us/">Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 为 KG 全实体生成语义画像，以可逆加性注入+成对匹配对齐，Full-Ranking 下 Recall@40 最高提升 2.6%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/graph_rec/how_do_graph_signals_affect_recommendation_unveiling_the_mystery_of_low_and/">How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals</a></strong></dt>
<dd>
<p>理论证明同绝对值低频/高频图信号在推荐中等价，提出频率信号缩放器与空间翻转插件，在四个数据集全库排序下 Recall@20 最高提升约 3.0%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/mixture_of_experts_knowledge_graph_retrieval_augmented_generation_for_multi/">Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation</a></strong></dt>
<dd>
<p>用 MoE 多粒度 KG 检索智能体按查询复杂度路由检索粒度，配合知识对齐与对比偏好优化，在 20 候选池协议下 Accuracy 相对提升 5.8%–20.4%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/hierarchical_graph_information_bottleneck_for_multi_behavior_recommendation/">Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用信息瓶颈原理重构多行为层次推荐，以 InfoNCE 保真、HSIC 压缩、可学习边丢弃去噪，全库排序下 HR@10 最高提升 15.08%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/nlgcl_naturally_existing_neighbor_layers_graph_contrastive_learning_for_rec/">NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用 GNN 相邻层天然异构节点作正样本对替代图增广，免视图构建，全库排序下 NDCG@10 最高提升 12.41% 且训练最快。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/non_parametric_graph_convolution_for_re_ranking_in_recommendation_systems/">Non-parametric Graph Convolution for Re-ranking in Recommendation Systems</a></strong></dt>
<dd>
<p>将图卷积从训练期搬到测试期，用相似用户/物品构造增强对并加权聚合打分，重排平均提升 8.1% 而额外开销仅 0.5%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/rankgraph_unified_heterogeneous_graph_learning_for_cross_domain_recommendat/">RankGraph: Unified Heterogeneous Graph Learning for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>构建跨产品异构异质图，用 GPU 加速 RGCN + 对比学习产出图 token 注入推荐基础模型，线上 CTR +0.92%、CVR +2.82%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/debias_fairness_industrial/ricciflowrec_a_geometric_root_cause_recommender_using_ricci_curvature_on_fi/">RicciFlowRec: A Geometric Root Cause Recommender Using Ricci Curvature on Financial Graphs</a></strong></dt>
<dd>
<p>用 Ollivier/Forman-Ricci 曲率与 Ricci 流在动态金融图上做根因归因，以曲率风险惩罚重排资产，NDCG@10 提升 4.9%、Top-10 波动降 28%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/sgcl_unifying_self_supervised_and_supervised_learning_for_graph_recommendat/">SGCL: Unifying Self-Supervised and Supervised Learning for Graph Recommendation</a></strong></dt>
<dd>
<p>将 BPR 与 InfoNCE 合并为单一监督图对比损失，免增广免负采样，精度与训练效率双超 SGL/SimGCL。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/unbiased_recommender_systems_with_implicit_feedback/">Unbiased Recommender Systems with Implicit Feedback</a></strong></dt>
<dd>
<p>博士论文愿景：以控制函数法纠偏排序位置偏差、以事后投影与消息传递加权纠偏 GNN 协同过滤的流行度偏差。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/adaptive_graph_integration_for_cross_domain_recommendation_via_heterogeneou/">Adaptive Graph Integration for Cross-Domain Recommendation via Heterogeneous Graph Coordinators</a></strong></dt>
<dd>
<p>用异质自适应图协调器把多域图拼成统一结构做多域预训练，再以图提示迁移到目标域，Full-Ranking 下全面超越 9 个基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/graph_rec/benchmarking_recommendation_classification_and_tracing_based_on_hugging_fac/">Benchmarking Recommendation, Classification, and Tracing Based on Hugging Face Knowledge Graph</a></strong></dt>
<dd>
<p>构建 260 万节点 HuggingKG 与三任务 HuggingBench，揭示稀疏领域 KG 上 KGRec 类方法失效、TransE 反超复杂模型的关键现象。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/cohesion_composite_graph_convolutional_network_with_dual_stage_fusion_for_m/">COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用行为模态早期精炼+晚期注意力融合的双阶段融合，配合异质/同质复合图与自适应BPR，在三个 Amazon 数据集上 R@20 相对提升 4.4%–6.1%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/graph_rec/comprehending_knowledge_graphs_with_large_language_models_for_recommender_s/">Comprehending Knowledge Graphs with Large Language Models for Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 理解 KG 局部子图并构建语义物品图做全局检索增强，离线预计算语义嵌入后接入 LightGCN，全库排序下全面超越 12 个基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/llm_rec/corona_a_coarse_to_fine_framework_for_graph_based_recommendation_with_large/">CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 做偏好/意图推理在交互图上逐级缩小候选子图，再由 GNN 精排，全库排序下 Recall 平均相对提升 18.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/graph_spectral_filtering_with_chebyshev_interpolation_for_recommendation/">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a></strong></dt>
<dd>
<p>用切比雪夫插值逼近非线性图滤波器，直接以原始交互为信号，突破 LightGCN 的截断线性低通瓶颈，Full-Ranking 下全面 SOTA 且推理快 8.1 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/llm_rec/large_language_models_enhanced_hyperbolic_space_recommender_systems/">Large Language Models Enhanced Hyperbolic Space Recommender Systems</a></strong></dt>
<dd>
<p>HyperLLM 用 LLM 生成层级标签与偏好摘要，经元优化两阶段训练将结构/语义层级信息注入双曲推荐，Recall@20 最高提升 41.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/pre_training_for_recommendation_unlearning/">Pre-training for Recommendation Unlearning</a></strong></dt>
<dd>
<p>提出预训练式影响编码器 UnlearnRec，将遗忘请求映射为嵌入漂移，免重训实现高效图推荐遗忘，10x 加速且保持精度。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/poi_spatiotemporal_rec/sep_gcn_leveraging_similar_edge_pairs_with_temporal_and_spatial_contexts_fo/">SEP-GCN: Leveraging Similar Edge Pairs with Temporal and Spatial Contexts for Location-Based Recommender Systems</a></strong></dt>
<dd>
<p>以时空相似交互边对构建边级相似图并注入LightGCN传播，在NYC/Gowalla/Brightkite全库排序上Recall@20最高提升14.89%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/self_supervised_denoising/squeeze_and_excitation_a_weighted_graph_contrastive_learning_for_collaborat/">Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>仅扰动末层GCN视图并用SENet动态加权特征，在三个数据集上实现最高24.45%的NDCG@20提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/llm_rec/you_are_what_you_bought_generating_customer_personas_for_e_commerce_applica/">You Are What You Bought: Generating Customer Personas for E-commerce Applications</a></strong></dt>
<dd>
<p>用 LLM 标注少量原型用户、随机游走传播生成可读人物画像，并以 RevAff 近似加速，画像注入 GCN 推荐最高提升 12%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/aspect_aware_content_based_recommendations_for_mathematical_research_papers/">Aspect-Aware Content-Based Recommendations for Mathematical Research Papers</a></strong></dt>
<dd>
<p>面向数学论文的 aspect 条件异构图推荐，构建 GoldRiM/SilverRiM 数据集，AchGNN 在 Full-Ranking 下 MRR 相对 GraphCL 提升 26%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/dcgl_dual_channel_graph_learning_with_large_language_models_for_knowledge_a/">DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation</a></strong></dt>
<dd>
<p>双通道解耦 LLM 语义与 ID 行为信号，多级对比对齐 + 频率感知门控融合，四数据集全库排序全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/from_top_1_to_top_k_a_reproducibility_study_and_benchmarking_of_counterfact/">From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems</a></strong></dt>
<dd>
<p>统一复现11种反事实解释器，首次提出列表级Top-K评估协议与PN-R位置敏感指标，揭示显式格式下有效性-稀疏性权衡与图解释器可扩展性瓶颈。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/qagcf_graph_collaborative_filtering_for_q_a_recommendation/">QAGCF: Graph Collaborative Filtering for Q&amp;A Recommendation</a></strong></dt>
<dd>
<p>将问答对解耦为用户-问题/用户-答案双协同图与问题-答案/问题-问题双语义图，融合为全局异质图后用 Jacobi 多项式图滤波器分离高低频信号，配合噪声对比学习，在知乎与商业数据集 Full-Ranking 下 Recall@10 提升 5.16%–5.34%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/robust_multimodal_recommendation_via_graph_retrieval_enhanced_modality_comp/">Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion</a></strong></dt>
<dd>
<p>用模态感知子图检索替代 K 跳邻居聚合，配合图 Transformer 与稀疏路由码本补全缺失模态，40% 缺失率下 Recall@20 相对提升 4.2%–5.1%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/timemm_time_as_operator_spectral_filtering_for_dynamic_multimodal_recommend/">TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation</a></strong></dt>
<dd>
<p>将交互新近度映射为时间核加权图算子族，隐式实现免特征分解的多尺度谱滤波，配合自适应算子混合与模态路由，全库排序下 R@20 相对提升 3%–10%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/wpgrec_wavelet_packet_guided_graph_enhanced_sequential_recommendation/">WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用平稳小波包变换生成等长平移不变子带，在子带内做 Chebyshev 图传播并以能量-谱平坦度门控融合，Full-Ranking 下稀疏数据集 HR@20 最高 +15.4%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/graph_rec/dual_channel_multiplex_graph_neural_networks_for_recommendation/">Dual-Channel Multiplex Graph Neural Networks for Recommendation</a></strong></dt>
<dd>
<p>双通道多路图网络显式建模行为模式、隐式挖掘关系链依赖，三数据集全指标平均提升 10%+</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/sequential_rec/explainable_session_based_recommendation_via_path_reasoning/">Explainable Session-based Recommendation via Path Reasoning</a></strong></dt>
<dd>
<p>提出分层强化学习框架 PR4SR，用会话级智能体选起点、路径级智能体做知识图谱推理，配合多目标与路径中点奖励，为 5 种 SR 模型提供可解释路径并提升精度。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/graph_rec/graph_cross_correlated_network_for_recommendation/">Graph Cross-Correlated Network for Recommendation</a></strong></dt>
<dd>
<p>用非递归 PGR 保留各跳邻居纯语义，再以 HCC/ECC 显式建模用户-物品子图交叉相关项，全库排序下 NDCG@20 平均提升约 1.8%–2.9%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/graph_rec/intent_guided_heterogeneous_graph_contrastive_learning_for_recommendation/">Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用元路径意图构建双对比学习视图，并以信息瓶颈掩码自编码器去噪，六数据集全库排序 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/collaborative_filtering/memory_is_no_longer_a_bottleneck_memory_efficient_graph_filtering_for_scala/">Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering</a></strong></dt>
<dd>
<p>用逐用户 Krylov 子空间替代全量物品相似度图，实现免训练多项式图滤波，内存降 5.74×、推理加速 26× 且精度反超 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/multimodal_rec/tackling_the_modality_inhibitance_effect_with_wavelet_based_modality_integr/">Tackling the Modality Inhibitance Effect with Wavelet-based Modality Integration in Multi-modal Recommendation</a></strong></dt>
<dd>
<p>提出模态抑制效应(MIE)概念，用Haar小波多分辨率分解提取各模态本质特征、结构感知多模态图学习与交叉加权自适应调节器，在4数据集Full-Ranking下平均提升6.48%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/multimodal_rec/training_free_graph_based_imputation_of_missing_modalities_in_multimodal_re/">Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation</a></strong></dt>
<dd>
<p>将缺失模态重铸为物品-物品共购图上的图特征插值，提出四种免训练传播式填补算子，可即插即用并保持甚至扩大多模态与传统推荐的性能差距。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/graph_rec/kg4receval_does_knowledge_graph_really_matter_for_recommender_systems/">KG4RecEval: Does Knowledge Graph Really Matter for Recommender Systems?</a></strong></dt>
<dd>
<p>提出 KGER 指标与 KG4RecEval 框架，系统扰动/删除 KG 知识，发现 KG 对推荐精度贡献常为负或近零，挑战 KG 必有益的主流认知。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/collaborative_filtering/lightweight_embeddings_with_graph_rewiring_for_collaborative_filtering/">Lightweight Embeddings with Graph Rewiring for Collaborative Filtering</a></strong></dt>
<dd>
<p>量化组合码本+图重连剪枝，在严格存储预算下兼顾 GNN 推荐精度与边缘端推理效率，iFashion 存储降 97.7%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/graph_rec/position_aware_graph_transformer_for_recommendation/">Position-aware Graph Transformer for Recommendation</a></strong></dt>
<dd>
<p>用四种为交互图定制的节点位置编码把 Transformer 全局建模能力注入 GCN 推荐骨干，在四个数据集上以 Full-Ranking 协议取得最高 27.69% 的 Recall@20 相对提升。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/self_supervised_denoising/symmetric_graph_contrastive_learning_against_noisy_views_for_recommendation/">Symmetric Graph Contrastive Learning against Noisy Views for Recommendation</a></strong></dt>
<dd>
<p>定义图增广产生的噪声视图，用对称指数对比损失替代 InfoNCE，使 GCL 对噪声视图鲁棒，全库排序下 NDCG@20 最高提升 12.25%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/graph_rec/cpgrec_a_balance_oriented_framework_for_personalized_video_game_recommendat/">CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations</a></strong></dt>
<dd>
<p>以 Fisher 分布符号边重加权缓解过平滑、LLM 生成偏好描述精炼表征，在 Steam 双数据集上兼顾精度与多样性。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/graph_rec/hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/recommendation_unlearning_via_influence_function/">Recommendation Unlearning via Influence Function</a></strong></dt>
<dd>
<p>用影响函数一步更新模型实现推荐遗忘，联合直接损失影响与图卷积溢出影响，剪枝加速后较重训提速 250 倍以上。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/graph_rec/an_aspect_performance_aware_hypergraph_neural_network_for_review_based_reco/">An Aspect Performance-aware Hypergraph Neural Network for Review-based Recommendation</a></strong></dt>
<dd>
<p>用四元组超图建模用户-物品-方面-情感极性，以用户偏好加权聚合冲突情感推断物品方面表现，六数据集 MSE 平均降 2.30%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/graph_rec/lightgnn_simple_graph_neural_network_for_recommendation/">LightGNN: Simple Graph Neural Network for Recommendation</a></strong></dt>
<dd>
<p>用可学习边/嵌入剪枝+分层知识蒸馏压缩 GNN 推荐模型，80% 边与 90% 嵌入裁剪下仍超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/graph_rec/mixrec_heterogeneous_graph_collaborative_filtering/">MixRec: Heterogeneous Graph Collaborative Filtering</a></strong></dt>
<dd>
<p>参数化异质超图解耦多行为意图，配合节点/图级自适应对比学习，Leave-One-Out 全库排序下 HR@10 最高提升 9.8%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/self_supervised_denoising/scone_a_novel_stochastic_sampling_to_generate_contrastive_views_and_hard_ne/">SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation</a></strong></dt>
<dd>
<p>用分数生成模型统一生成用户-物品特定对比视图与难负样本，6 数据集全库排序 Recall@20 最高提升 1.70%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/multimodal_rec/spectrum_based_modality_representation_fusion_graph_convolutional_network_f/">Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation</a></strong></dt>
<dd>
<p>将多模态特征投影至频域做滤波去噪与点积融合，配合模态专属/融合双图学习与模态感知偏好模块，在 Amazon 三数据集全库排序下 Recall@20 相对最优基线提升约 5%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/cold_start_rec/epiccbr_item_relation_enhanced_dual_scenario_contrastive_learning_for_cold_/">EpicCBR: Item-Relation-Enhanced Dual-Scenario Contrastive Learning for Cold-Start Bundle Recommendation</a></strong></dt>
<dd>
<p>通过物品对关系增强与流行度感知双场景对比学习，在冷启动捆绑推荐上实现最高 387% 的 Recall@20 提升。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/criteria_aware_graph_filtering_extremely_fast_yet_accurate_multi_criteria_r/">Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation</a></strong></dt>
<dd>
<p>免训练免矩阵分解的多准则图滤波推荐，按准则定制多项式低通滤波器并融合用户准则偏好，BA 数据集 0.2 秒推理、NDCG@5 最高提升 24%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/distributionally_robust_graph_out_of_distribution_recommendation_via_diffus/">Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model</a></strong></dt>
<dd>
<p>揭示 DRO 推荐对噪声分布过度加权之弊，用 VGAE+扩散去噪、Sinkhorn 距离与熵正则重构最坏情形优化，OOD 场景最高提升 75.5%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/frequency_aware_graph_signal_processing_for_collaborative_filtering/">Frequency-aware Graph Signal Processing for Collaborative Filtering</a></strong></dt>
<dd>
<p>用级联高通-低通滤波捕获独有/共性特征、并行低通滤波捕获高阶邻域，免训练线性组合在六个数据集上 Full-Ranking 提升 2-5%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/g_refer_graph_retrieval_augmented_large_language_model_for_explainable_reco/">G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation</a></strong></dt>
<dd>
<p>用路径级+节点级混合图检索把协同过滤信号显式翻译为文本，经知识剪枝与 RAFT 微调驱动 LLM 生成可解释推荐，BERT-F1 最高提升 2.81%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/graph_neural_controlled_differential_equations_for_collaborative_filtering/">Graph Neural Controlled Differential Equations For Collaborative Filtering</a></strong></dt>
<dd>
<p>用 MLP 控制器生成连续时变权重矩阵替代 GODE-CF 的离散权重，在四个 Amazon 5-core 数据集 Full-Ranking 下 Recall@20 全面超越 LightGCN/GODE-CF。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/graph_representation_learning_via_causal_diffusion_for_out_of_distribution_/">Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation</a></strong></dt>
<dd>
<p>用结构因果模型定位环境混杂因子，以变分推断+扩散模型学习环境不变图表征，OOD 场景最高提升 36.73%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/graph_rec/graph_sequential_alignment_and_uniformity_toward_enhanced_recommendation_sy/">Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems</a></strong></dt>
<dd>
<p>以共享嵌入层联合训练 LightGCN 与 SASRec，用对齐-均匀性损失实现跨子模块知识迁移，Leave-One-Out 全库排序下 R@10 最高提升 13.4%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/graphhash_graph_clustering_enables_parameter_efficiency_in_recommender_syst/">GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems</a></strong></dt>
<dd>
<p>用模块度二部图聚类替代 ID 哈希，将图平滑前置到预处理阶段，嵌入表缩减 75%+ 时 Recall@20 平均提升 101.52%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/generative_rec/hyperbolic_diffusion_recommender_model/">Hyperbolic Diffusion Recommender Model</a></strong></dt>
<dd>
<p>将扩散推荐从欧氏空间迁移至双曲流形，用聚类锚定的方向性噪声替代各向同性高斯噪声，Full-Ranking 下 R@10 最高提升 5.6%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/social_group_rec/leveraging_member_group_relations_via_multi_view_graph_filtering_for_effect/">Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation</a></strong></dt>
<dd>
<p>提出免训练多视图图滤波 Group-GF，用三张物品相似图与多项式滤波器聚合成员-群组关系，HR@10 最高提升 16.4% 且运行时间仅 1.55s。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/self_supervised_denoising/mixrec_individual_and_collective_mixing_empowers_data_augmentation_for_reco/">MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems</a></strong></dt>
<dd>
<p>用 Beta/Dirichlet 混合在嵌入空间零成本造正负视图，配合双混合对比损失，全库排序下 Recall@20 最高提升 5.05%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/rankformer_a_graph_transformer_for_recommendation_based_on_ranking_objectiv/">Rankformer: A Graph Transformer for Recommendation based on Ranking Objective</a></strong></dt>
<dd>
<p>从 BPR 梯度下降推导出图 Transformer 层，正负样本分离聚合+基准项+偏移项，线性加速后 4 数据集平均提升 4.48%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/beyond_isolated_clients_integrating_graph_based_embeddings_into_event_seque/">Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models</a></strong></dt>
<dd>
<p>将二部交互图结构以三种模型无关策略注入事件序列对比 SSL，揭示图密度决定最优融合策略，AUC 最高提升 2.3%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/iterative_semantic_reasoning_from_individual_to_group_interests_for_generat/">Iterative Semantic Reasoning from Individual to Group Interests for Generative Recommendation with LLMs</a></strong></dt>
<dd>
<p>用 LLM 双向 CoT 推理物品/用户语义，构建个体显式兴趣图与相似用户群隐式兴趣图，并以对比蒸馏迭代对齐，直接推荐 H@10 最高提升 24.57%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/modeling_endogenous_logic_causal_neuro_symbolic_reasoning_model_for_explain/">Modeling Endogenous Logic: Causal Neuro-Symbolic Reasoning Model for Explainable Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用前门调整把多行为链内源逻辑做成可解释因果中介，三路神经逻辑推理在三个数据集 Full-Ranking 上 HR@10 提升 20%+。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/modeling_stage_wise_evolution_of_user_interests_for_news_recommendation/">Modeling Stage-wise Evolution of User Interests for News Recommendation</a></strong></dt>
<dd>
<p>用全局LightGCN+阶段时序子图LSTM/自注意力双分支建模新闻兴趣演化，Adressa AUC 0.7993 大幅超越SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/radar_relation_aware_diffusion_asymmetric_graph_contrastive_learning_for_re/">RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>以 VGAE 生成视图与关系感知去噪视图构成双视图，叠加扩散引导增广与非对称对比目标，在 Full-Ranking 下 Recall@20 相对 SOTA 提升 3.8%–4.7%。</p>
</dd>
</dl>

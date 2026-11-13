# ⏱️ 序列与会话推荐 (Sequential & Session-Based Recommendation)

> **研究范畴**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

当前方向共收录 **140** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2026/sequential_rec/bid_farewell_to_seesaw_towards_accurate_long_tail_session_based_recommendat/">Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents</a></strong></dt>
<dd>
<p>提出混合意图学习与意图约束损失的双约束即插即用框架 HID，在三个数据集上同时提升会话推荐准确率与长尾覆盖，打破 see-saw 困境。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/sequential_rec/driving_engagement_in_daily_fantasy_sports_with_a_scalable_and_urgency_awar/">Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine</a></strong></dt>
<dd>
<p>在 DIN 上注入实时紧迫度特征与时间位置编码，配合 neuralNDCG 列表损失，在 6.5 亿用户级工业数据上 nDCG@1 相对 LightGBM 提升 9%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/sequential_rec/exploiting_inter_session_information_with_frequency_enhanced_dual_path_netw/">Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation</a></strong></dt>
<dd>
<p>FreqRec 用可学习复数频域 MLP 双路建模跨会话群体节奏与用户局部高频，并加频域一致性损失，全库排序下 NDCG@10 最高提升 7.38%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/sequential_rec/hymoerec_hybrid_mixture_of_experts_for_sequential_recommendation/">HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation</a></strong></dt>
<dd>
<p>用共享+稀疏双分支混合专家替换均匀 PFFN，配合自适应融合与负载均衡，在 ML-1M/Beauty 全库排序上稳定超越 Mamba4Rec。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/ctr_feature_interaction/length_adaptive_interest_network_for_balancing_long_and_short_sequence_mode/">Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction</a></strong></dt>
<dd>
<p>将序列长度作为条件信号注入提示与注意力温度，缓解短序列用户注意力极化，五骨干平均AUC最高+1.15%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/graph_rec/task_aware_retrieval_augmentation_for_dynamic_recommendation/">Task-Aware Retrieval Augmentation for Dynamic Recommendation</a></strong></dt>
<dd>
<p>用任务感知评估机制自动标注历史子图收益，并以图 Transformer 打分检索融合，缓解动态推荐预训练-微调时序泛化鸿沟。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/sequential_rec/wavelet_enhanced_adaptive_frequency_filter_for_sequential_recommendation/">Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation</a></strong></dt>
<dd>
<p>用上下文动态调制频域滤波器并叠加 Haar 小波高频增强，在 4 个基准 Full-Ranking 下 HR@20 最高提升 11.4%，长序列下训练开销低于 BSARec/SLIME4Rec。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/beyond_retrieval_learning_compact_user_representations_for_scalable_llm_per/">Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization</a></strong></dt>
<dd>
<p>用双轨前缀（用户态+查询感知时序记录）替代提示序列化与逐用户适配器，LaMP 六任务全面领先且每用户参数降 130×</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/generative_rec/from_past_to_path_masked_history_learning_for_next_item_prediction_in_gener/">From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation</a></strong></dt>
<dd>
<p>在自回归生成式推荐上叠加掩码历史重建辅助任务，用熵引导掩码与课程调度弥合训练-推理鸿沟，Amazon 三域 Full-Ranking 全面 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/cross_domain_multi_scenario/horizon_a_benchmark_for_in_the_wild_user_behaviour_modeling/">HORIZON: A Benchmark for In-the-wild User Behaviour Modeling</a></strong></dt>
<dd>
<p>重构 Amazon Reviews 为 54M 用户跨域长程基准，提出时间-用户双轴四象限评估与 LLM 查询改写/长程建模任务，揭示 ID 模型在时间外推下崩塌。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/tatk_triple_aware_top_k_learning_with_knowledge_grounded_verification_for_l/">TATK: Triple-Aware Top-K Learning with Knowledge-Grounded Verification for LLM-based Sequential Recommendation</a></strong></dt>
<dd>
<p>用元数据物品图贯穿提示接地、Top-K 位置奖励与推理期 top-M 重排，全库排序下 NDCG@10 最高提升 27.03%</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/sequential_rec/two_sided_state_space_models_for_sequential_recommendation_with_non_random_/">Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback</a></strong></dt>
<dd>
<p>将评论视为双侧状态空间中的内生事件，用 MNAR 观测模式、非对称评论结转与局部图传播联合演化用户/物品状态，Full-Ranking 下 R@20 较 BSARec 提升 14.8%–18.8%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/what_makes_llms_effective_sequential_recommenders_a_study_on_preference_int/">What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context</a></strong></dt>
<dd>
<p>揭示二元偏好对齐丢失偏好强度与时序上下文，提出自适应奖励边际 RecPO，五数据集 HR@1 全面领先 S-DPO。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_noisy_signals_dual_level_denoising_for_multi_modal_sequential_recomm/">Beyond Noisy Signals: Dual-Level Denoising for Multi-modal Sequential Recommendation</a></strong></dt>
<dd>
<p>图拉普拉斯低通滤波去特征噪声 + FFT 可学习频域滤波去序列噪声，四数据集 Full-Ranking 全面 SOTA，R@20 最高 +19.33%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/closing_the_long_short_view_gap_in_sequential_recommendation_without_cached/">Closing the Long-Short View Gap in Sequential Recommendation without Cached History</a></strong></dt>
<dd>
<p>用角度相似度打分+softmax1 修正长视训练几何缺陷，再仅微调 Bias/LayerNorm 实现无缓存短视推理，短窗下反超全序列基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/cross_domain_sequential_recommendation_with_representation_level_user_trans/">Cross-Domain Sequential Recommendation with Representation-Level User Transfer</a></strong></dt>
<dd>
<p>仅通过用户表征级迁移实现跨域序列推荐，融合层注入源域用户向量，兼容纵向联邦与部分重叠用户，全库排序下 NDCG@10 提升 7.4–24.1%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/decoupled_temporal_encoding_for_generative_recommendation/">Decoupled Temporal Encoding for Generative Recommendation</a></strong></dt>
<dd>
<p>将宏观时间动态与微观序列顺序解耦的双模块位置编码，工业数据集 GAUC +1.16pt，线上 CTR +1.8%、RPM +3.0%，延迟仅 +0.3%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/enriching_sequential_recommendation_with_graph_laplacian_positional_embeddi/">Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings</a></strong></dt>
<dd>
<p>用物品共现图归一化拉普拉斯特征向量冻结替换 SASRec 可学习位置嵌入，四基准多数指标提升且在线开销近零。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/rl_bandit_rec/evol_simulator_guided_evolutionary_expert_synthesis_for_deployment_free_lea/">EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation</a></strong></dt>
<dd>
<p>用知识追踪模拟器内进化搜索合成专家路径，蒸馏为仅凭初始掌握度盲规划的非对称 Actor-Critic，实现免模拟器部署的学习路径推荐。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/from_overlooked_to_explored_recovering_item_relations_via_mixture_of_perspe/">From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation</a></strong></dt>
<dd>
<p>揭示自注意力相似性偏置导致异构关系被系统性忽略，提出 K 个视角透镜（亲和/对比双视图）校准注意力，7 个基准全库排序平均提升 3–11%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/god_enhancing_generalization_via_deep_grafting_for_sequential_recommendatio/">GOD: Enhancing Generalization via Deep Grafting for Sequential Recommendation</a></strong></dt>
<dd>
<p>用「嫁接」构造教师条件混合源模型，对学生嵌入表/编码器做组件级蒸馏，Full-Ranking 下最高相对提升 13.92%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/intsr_an_integrated_generative_framework_for_search_and_recommendation/">IntSR: An Integrated Generative Framework for Search and Recommendation</a></strong></dt>
<dd>
<p>以 query 模态统一搜索/推荐与召回/排序四任务，提出 query 驱动解码器与时间对齐负采样，Amap 线上 GMV+9.34%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/lsf_sr_latent_semantic_fusion_for_sequential_recommendation_via_flow_based_/">LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders</a></strong></dt>
<dd>
<p>用归一化流增强的 CVAE 将 LLM 语义作为条件概率融合进 ID 协同嵌入，Full-Ranking 下 Recall@20 最高提升 12.98%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/palate_pretrained_autoregressive_latent_embeddings/">PALATE: Pretrained Autoregressive LATent Embeddings</a></strong></dt>
<dd>
<p>用GPT式自回归模型对消费者门店互动序列建模，导出末词元隐状态为冻结256维用户嵌入，供DCNv2广告排序器跨场景复用，离线AUC+0.59%/+0.57%，线上收入+0.79%/+0.94%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/rethinking_item_tokenization_in_generative_recommenders_from_fixed_atoms_to/">Rethinking Item Tokenization in Generative Recommenders: From Fixed Atoms to Semantic Subwords</a></strong></dt>
<dd>
<p>指出固定长度SID复用于历史侧引发“物品内注意力过载”，提出历史侧语义子词重分词(IST)+行为共现增强(BCA)的非对称分词，三数据集三骨干全库排序NDCG@10最高+13.8%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/routerec_behavior_guided_sparse_routing_for_sequential_recommendation/">RouteRec: Behavior-Guided Sparse Routing for Sequential Recommendation</a></strong></dt>
<dd>
<p>用会话行为线索（节奏/焦点/记忆/流行度）驱动分层稀疏 MoE 路由，六数据集 18 指标平均排名 1.61 显著优于次优 4.11。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/self_evolving_memory_for_generative_recommendation/">Self-Evolving Memory for Generative Recommendation</a></strong></dt>
<dd>
<p>揭示生成式推荐持续演化中的“演化冲突”问题，提出稀疏 KV 记忆层 LION，以隔离记忆+巩固损失实现异构行为模式的持续演化。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/sequential_modality_dropout_for_robust_multi_modal_sequential_recommendatio/">Sequential Modality Dropout for Robust Multi-Modal Sequential Recommendation</a></strong></dt>
<dd>
<p>在融合点注入逐样本伯努利模态掩码，四行代码即插即用，使四种多模态序列骨干在模态缺失下 HR@10 留存率提升 1.0–3.2×。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/towards_comprehensible_recommendation_with_large_language_model_fine_tuning/">Towards Comprehensible Recommendation with Large Language Model Fine-tuning</a></strong></dt>
<dd>
<p>用 GRPO 两阶段对齐 LLM 生成协同视角的用户模式与推荐理由，并以推荐模型打分为奖励精炼，Full-Ranking 下 Recall@5 相对提升 14–18%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/tracer_balancing_stability_plasticity_cognitivity_trilemma_for_llm_enhanced/">TRACER: Balancing Stability-Plasticity-Cognitivity Trilemma for LLM Enhanced Continual Recommendation</a></strong></dt>
<dd>
<p>揭示 LLM 增强持续推荐中的 SPC 三难困境，用 Procrustes 投影、置信门控凝聚与语义同步三模块协同破局，Full-Ranking 下 ACC 最高提升 14.38%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2026/llm_rec/dynamicpo_dynamic_preference_optimization_for_recommendation/">DynamicPO: Dynamic Preference Optimization for Recommendation</a></strong></dt>
<dd>
<p>揭示多负样本 DPO 的偏好优化坍缩现象，提出动态边界负样本选择与双边际动态 β 调整，即插即用提升 LLM 推荐 HitRatio@1。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/llm_rec/pre_trained_llms_meet_sequential_recommenders_efficient_user_centric_knowle/">Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation</a></strong></dt>
<dd>
<p>用预训练 LLM 生成文本用户画像并蒸馏进 SASRec/BERT4Rec 隐层，两阶段训练实现零推理开销的语义增强，Recall@10 最高提升 23.5%。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2026/multimodal_rec/cammsr_category_guided_attentive_mixture_of_experts_for_multimodal_sequenti/">CAMMSR: Category-Guided Attentive Mixture of Experts for Multimodal Sequential Recommendation</a></strong></dt>
<dd>
<p>用类别预测任务引导注意力 MoE 动态分配模态权重，并以模态交换对比学习对齐跨模态序列表征，Full-Ranking 下 NDCG@5 最高提升 15.3%。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2026/cold_start_rec/user_adaptive_meta_learning_for_cold_start_medication_recommendation_with_u/">User-Adaptive Meta-Learning for Cold-Start Medication Recommendation with Uncertainty Filtering</a></strong></dt>
<dd>
<p>提出两级元自适应（自自适应+同伴自适应）与不确定性过滤的 MetaDrug，在 MIMIC-III/AKI 冷启动患者上 PRAUC 提升约 2.5%-6.2%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/llm_rec/empowering_compact_llms_with_fusion_of_layer_wise_exits_for_recommendation/">Empowering Compact LLMs with Fusion of Layer-wise Exits for Recommendation</a></strong></dt>
<dd>
<p>在紧凑 LLM 每层插入预测出口并用连续 ReLU 路由器自适应融合多层打分分布，target-k hinge 损失约束激活出口数，全库排序下精度超越大模型基线且延迟仅增 &lt;1ms。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/sequential_rec/hierarchical_exponential_gaussian_mixtures_for_watch_time_distribution_pred/">Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction</a></strong></dt>
<dd>
<p>用分层 skip-watch 分解与结构化初始化修复 EGMN 的方差坍缩与组件冗余，在固定骨干换头协议下提升 XAUC 与阈值事件预测，线上会话深度 +9.26%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/llm_rec/memory_has_geometry_non_uniform_geometric_memory_for_long_horizon_personali/">Memory Has Geometry: Non-Uniform Geometric Memory for Long-Horizon Personalized AI</a></strong></dt>
<dd>
<p>将长程个性化记忆重构为用户特定非均匀几何状态空间上的轨迹条件状态估计，替代扁平记录检索范式。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/sequential_rec/revisiting_self_attentive_sequential_recommendation_beyond_the_llm_paradigm/">Revisiting Self-Attentive Sequential Recommendation Beyond the LLM Paradigm</a></strong></dt>
<dd>
<p>以 SASRec 为锚点对比推荐与语言建模的熵结构差异，诊断四条继承假设并给出数据优先议程，含同模型双视图实证。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/sequential_rec/user_representation_via_cross_multi_source_behavior_pre_training_for_mobile/">User Representation via Cross Multi-source Behavior Pre-training for Mobile Games</a></strong></dt>
<dd>
<p>面向设备级多源多粒度行为日志，提出跨源级联掩码预训练 CM-PTM，以源→类目→App→操作逐级预测统一建模跨源依赖，离线 AUC 提升 1.05%–2.85%，线上 A/B 提升 1.28%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/can_recommender_systems_teach_themselves_a_recursive_self_improving_framewo/">Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control</a></strong></dt>
<dd>
<p>提出 RSIR 递归自我改进框架，用模型自生成序列+保真度质量控制稠密化稀疏交互，理论证明其等价于流形切向隐式正则，多骨干多数据集 Full-Ranking 下 NDCG/Recall 稳定提升 3%–14%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential/">CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation</a></strong></dt>
<dd>
<p>用可学习全局 K/V 池跨用户共享协同信息，将 KV cache 压缩至 0.8% 且精度不降，解码延迟随 batch 增长近乎平坦。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/cramer_control_via_request_aware_masking_for_editing_recommenders/">CRAMER: Control via Request-Aware Masking for Editing Recommenders</a></strong></dt>
<dd>
<p>将自然语言请求编码为稀疏行列门控掩码，对冻结序列推荐骨干做后验行为编辑，采样100负例排序下全面超越四类请求感知基线且推理仅增0.018s。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multi_task_reranking/goal_conditioned_supervised_learning_for_multi_objective_recommendation/">Goal-Conditioned Supervised Learning for Multi-Objective Recommendation</a></strong></dt>
<dd>
<p>将 GCSL 的标量目标扩展为多维向量目标，用 Transformer 编码器 + 交叉熵直接做多目标下一物品预测，无需复杂架构与权重调参，在 RetailRocket/Challenge15 上购买 HR@10 相对提升约 4-17%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/learning_evidence_highlighting_for_frozen_llms/">Learning Evidence Highlighting for Frozen LLMs</a></strong></dt>
<dd>
<p>用轻量 Emphasis Actor 以 RL 在冻结 LLM 的原始上下文中插入高亮标签，解耦证据选择与推理，序列推荐 HR@10 相对最优基线提升 5.38%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/lime_link_based_user_item_interaction_modeling_with_decoupled_xor_attention/">LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling</a></strong></dt>
<dd>
<p>用低秩 link embedding 解耦用户-候选交互并预计算 QK 权重，配合 XOR 线性注意力将长历史建模降至 O(N)，实现 HSTU 级精度与 10× 推理加速。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/massive_memorization_with_hundreds_of_trillions_of_parameters_for_sequentia/">Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders</a></strong></dt>
<dd>
<p>VISTA 将超长用户历史两阶段解耦：离线用准线性注意力压缩为数百种子嵌入并缓存，下游仅做候选-摘要注意力，工业上线推理 GPU 降 94%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/not_all_is_lost_repairing_lossy_user_preference_states_of_personalization_e/">Not All Is Lost: Repairing Lossy User Preference States of Personalization Encoders</a></strong></dt>
<dd>
<p>冻结宿主编码器与任务头，仅训练轻量 REPAIR 模块，用缓存逐步表征对压缩偏好状态做时序结构化残差校正，12 个推荐宿主 MRR/nDCG@10 全面提升。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/generative_rec/rethinking_semantic_id_construction_for_generative_recommendation_simhash_w/">Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment</a></strong></dt>
<dd>
<p>揭示LSH分词劣势源于与自回归解码的结构错配，提出免训练SimHash+并行解码+语义对齐的FLASH，全库排序下NDCG@5最高提升14.46%。</p>
</dd>
<dt><strong><a href="../../venues/IJCAI/2026/sequential_rec/rethinking_convolutional_networks_for_attribute_aware_sequential_recommenda/">Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation</a></strong></dt>
<dd>
<p>用金字塔式降尺度卷积替代自注意力聚合属性感知序列，线性复杂度下在四个 Amazon 数据集上 HR@10 最高提升 4.08%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/apao_bridging_the_training_inference_gap_in_generative_recommendation_via_a/">APAO: Bridging the Training-Inference Gap in Generative Recommendation via Adaptive Prefix-Aware Optimization</a></strong></dt>
<dd>
<p>揭示 Beam Search 剪枝与 teacher-forcing 训练的前缀级不一致，用自适应最差前缀损失对齐训练与推理，全库排序下 Recall@20 最高提升 13.3%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/fuxi_efficient_sequential_recommendation_with_exponential_power_temporal_en/">FuXi-$γ$: Efficient Sequential Recommendation with Exponential-Power Temporal Encoder and Diagonal-Sparse Positional Mechanism</a></strong></dt>
<dd>
<p>用艾宾浩斯遗忘曲线启发的指数幂时间编码器替代分桶查表，配合 Toeplitz 对角稀疏位置剪枝，在 Full-Ranking 下精度 SOTA 且训练/推理加速最高 4.74×/6.18×。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/hi_sam_a_hierarchical_structure_aware_multi_modal_framework_for_large_scale/">Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>解耦语义分词器+分层记忆锚点Transformer，修复语义ID推荐中分词冗余与扁平架构失配，工业A/B响应率+6.55%</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/journeyformer_encoding_airbnb_guest_journey_with_sequence_modeling/">JourneyFormer: Encoding Airbnb Guest Journey with Sequence Modeling</a></strong></dt>
<dd>
<p>Airbnb 搜索排序的工业级序列建模方案，以频次分层长短期序列+统一ID嵌入+因果Transformer编码访客旅程，离线NDCG+1.48%、线上Bookers+0.55%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/mixformer_co_scaling_up_dense_and_sequence_in_industrial_recommenders/">MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders</a></strong></dt>
<dd>
<p>用单一参数空间统一序列建模与稠密特征交叉，配合用户-物品解耦实现请求级复用，AUC 提升 1.28% 且 FLOPs 降 36%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/one_sequential_recommendation_model_pretrained_from_synthetic_priors_predic/">One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets</a></strong></dt>
<dd>
<p>用 hDCSBM 合成先验预训练 PFN，以支持集条件化实现免更新推理的序列推荐，五基准平均超最优基线 7.53%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/s_2_gr_stepwise_semantic_guided_reasoning_in_latent_space_for_generative_re/">S$^2$GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation</a></strong></dt>
<dd>
<p>在每级 SID 码生成前插入受码本聚类语义对比监督的思考 token，实现逐步可验证潜在推理，工业短视频 A/B 显著提升时长。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/scaling_recommender_transformers_to_one_billion_parameters/">Scaling Recommender Transformers to One Billion Parameters</a></strong></dt>
<dd>
<p>ARGUS 将自回归预训练解耦为反馈预测与下一物品预测双任务，把推荐 Transformer 编码器扩到 10 亿参数，线上听歌时长 +2.26%、点赞率 +6.37%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/the_pitfall_of_scaling_up_uncovering_and_mitigating_popularity_bias_amplifi/">The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders</a></strong></dt>
<dd>
<p>揭示 Transformer 序列推荐随深度加深发生谱坍缩并放大流行度偏差，提出逐层注意力列和与 FFN 谱范数正则 SPRINT，在 6 数据集上同时提升准确率与长尾公平性。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/understanding_generative_recommendation_with_semantic_ids_from_a_model_scal/">Understanding Generative Recommendation with Semantic IDs from a Model-scaling View</a></strong></dt>
<dd>
<p>揭示 SID 式生成推荐在编码器/量化器/RS 三轴扩展均快速饱和，归因于 SID 语义容量瓶颈，并证明 Text-based GR 具备更优扩展性且 CF 能力随规模提升。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/unleashing_the_potential_of_neighbors_diffusion_based_latent_neighbor_gener/">Unleashing the Potential of Neighbors: Diffusion-based Latent Neighbor Generation for Session-based Recommendation</a></strong></dt>
<dd>
<p>用检索先验与多模态信号双路引导扩散模型生成“潜在邻居”，突破可观测邻居边界，Full-Ranking 下 P@20 最高提升 18.45%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/addressing_cross_stage_decoupling_of_semantic_and_collaborative_signals_in_/">Addressing Cross-Stage Decoupling of Semantic and Collaborative Signals in Generative Recommendation</a></strong></dt>
<dd>
<p>提出 SCRec，通过协同增强分词、语义引导生成与双曲流形对齐三模块双向补全语义与协同信号，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 相对提升 18.7%–26.6%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/ctr_feature_interaction/building_a_user_foundation_model_for_the_open_web/">Building a User Foundation Model for the Open Web</a></strong></dt>
<dd>
<p>在身份碎片化的开放网络 RTB 场景下，用 MLM+时序切分对比预训练 Transformer 用户序列编码器，并以 LLM-in-the-loop NAS 优化训练管线，线上 A/B 取得 +2.13% CTR、-1.13% eCPC。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/cofirec_coarse_to_fine_tokenization_for_generative_recommendation/">CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation</a></strong></dt>
<dd>
<p>将物品元数据按类目→标题→描述→CF信号分层独立量化，自回归由粗到细解码，Leave-One-Out全库排序下平均超越SOTA约6.4%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/deepaffinity_long_term_aspect_preference_prediction_in_ecommerce_using_smal/">DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models</a></strong></dt>
<dd>
<p>用 135M 小模型+结构化字符串化+多任务预测头，从时序行为预测品牌/尺码/颜色等商品属性偏好，Fashion 微 F1 0.34 超 Gemini 零样本与 3B SFT。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/do_sequential_recommendation_benchmarks_really_require_higher_order_sequenc/">Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?</a></strong></dt>
<dd>
<p>用两个不学高阶序列表征的近期加权成对探针（SeqRules/PCTM）在 eSASRec 协议下反超 Transformer，质疑主流基准的容量测量效度。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/dp_rec_towards_dynamic_patching_for_efficient_long_sequence_recommendation/">DP-Rec: Towards Dynamic Patching for Efficient Long-Sequence Recommendation</a></strong></dt>
<dd>
<p>用对比熵惊讶度动态切分长交互序列为变长行为块，在 FLOPs 匹配下 NDCG@10 最高提升 21.1%，实现更优效率-精度前沿。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/longer_scaling_up_long_sequence_modeling_in_industrial_recommenders/">LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders</a></strong></dt>
<dd>
<p>用全局Token+Token Merge+混合因果注意力把工业超长行为序列端到端建模推到1万长度，离线AUC+1.57%、线上广告与电商双场景显著提升。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/recap_feedback_driven_streaming_semantic_user_profiles_for_short_video_reco/">RECAP: Feedback-Driven Streaming Semantic User Profiles for Short-Video Recommendation</a></strong></dt>
<dd>
<p>将流式结构化用户画像视为有界记忆状态，用 LLM 语义差分+确定性状态机维护，并以 LLM 判官清洗隐式反馈、双塔评估器打分作 GRPO 奖励闭环优化画像更新器。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/recrec_latent_interests_recursive_reasoning_for_sequential_recommendation/">RecRec: Latent Interests Recursive Reasoning for Sequential Recommendation</a></strong></dt>
<dd>
<p>解耦推理态与预测态、以 R 个潜在兴趣做递归推理的免 RL 序列推荐框架，Full-Ranking 下 NDCG@10 最高提升 14.8%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/residual_dominance_as_a_structural_account_of_last_item_reliance_in_causal_/">Residual Dominance as a Structural Account of Last-Item Reliance in Causal Self-Attention Recommenders</a></strong></dt>
<dd>
<p>用范数分解揭示残差连接主导同位置信息保留，并以推理期残差缩放诊断因果自注意力推荐器的末项依赖机制。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/sequenceo1_end_to_end_ultra_long_100k_sequence_modeling_in_recommendation_w/">SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching</a></strong></dt>
<dd>
<p>用可学习原型压缩 100K 历史为定长 sketch 并缓存复用，端到端精排训练/推理 FLOPs 降 49.9×/63.9×，线上 Finish +2.33%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/structure_preserving_projection_for_mitigating_modality_bias_in_llm_based_s/">Structure-Preserving Projection for Mitigating Modality Bias in LLM-Based Sequential Recommendation</a></strong></dt>
<dd>
<p>用余弦相似度保持与序列对比保持双损失约束协同嵌入投影，缓解 LLM 推荐中的模态偏置，20 候选协议下 HR@1 最高提升约 12pt。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/topology_aware_tokenization_for_generative_recommendation/">Topology-Aware Tokenization for Generative Recommendation</a></strong></dt>
<dd>
<p>揭示 RQ-VAE 语义 ID 量化中的拓扑失真问题，提出由粗到细的三级拓扑蒸馏 TopoTok，在 Amazon 三数据集 Full-Ranking 下 Recall@5 最高提升 9.42%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/watchlens_a_configurable_platform_for_online_video_recommendation_experimen/">WatchLens: A Configurable Platform for Online Video Recommendation Experiments</a></strong></dt>
<dd>
<p>开源视频推荐在线实验平台，解耦界面/内容池/双表面策略并原生记录曝光上下文，30人被试验证观看页策略对会话留存的影响。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/zero_observation_user_reactivation_with_gap_driven_dimensional_gating/">Zero-Observation User Reactivation with Gap-Driven Dimensional Gating</a></strong></dt>
<dd>
<p>定义零观测用户唤醒问题，提出冻结骨干的逐维门控插件 DeltaGate，以 Δt 与历史表征联合路由个性化历史与全局先验，&gt;365d 桶 Hit@10 提升 33%–84%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在保持语义层级的同时缓解几何失衡，Full-Ranking 下 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 学习用户自适应增强，四数据集全排序下 HR/NDCG 平均提升约 7-8%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/beyond_item_ids_scaling_short_form_video_recommendation_via_semantic_native/">Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling</a></strong></dt>
<dd>
<p>用深度截断 Bi-gram 语义 ID 替代正交 Video ID，并以无参时间折叠压缩 Transformer 将序列扩至 2000，线上满意观看时长 +1.42%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/bridging_behavior_and_semantics_for_time_aware_cross_domain_sequential_reco/">Bridging Behavior and Semantics for Time-aware Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用神经 ODE 解耦长短期行为、LLM 离散时间 token 反事实增强语义、门控自适应域迁移，1:999 采样排序下 MRR 最高提升 30.5%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong></dt>
<dd>
<p>以日历时间二值信号+共享多尺度时序卷积建模物品复购节奏，再用 ISAB 建模跨物品依赖，在四个杂货数据集与千万级生产系统上取得显著提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong></dt>
<dd>
<p>提出可学习多序列构造与线性注意力，缓解单序列上下文污染，Meta 四场景 NE 降 0.3%–0.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong></dt>
<dd>
<p>借鉴 ZPD 理论构建 Teacher-Peer-Self 多源负采样，用分歧重排提升多样性、共识蒸馏复用打分开销，六数据集全库排序下 Recall@20 最高提升 27.3%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/erase_a_real_world_aligned_benchmark_for_unlearning_in_recommender_systems/">ERASE -- A Real-World Aligned Benchmark for Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>面向真实场景的推荐系统机器遗忘基准，覆盖 CF/SBR/NBR 三任务、七种遗忘算法、九数据集九模型，揭示近似遗忘的鲁棒性与效率短板。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/every_preference_has_its_strength_injecting_ordinal_semantics_into_llm_base/">Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders</a></strong></dt>
<dd>
<p>将显式评分建模为LLM词表中的序数语义锚点，以强度感知对齐交互表征，在序列推荐Hit@1与成对偏好准确率上显著超越CF-LLM混合基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/fave_flow_based_average_velocity_establishment_for_sequential_recommendatio/">FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义锚先验替代高斯噪声、以平均速度场实现单步流匹配生成推荐，全库排序下 N@20 相对提升 9.9%，推理提速 20 倍以上。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>以目标条件化频谱熵观察为动机，用目标感知频谱滤波+时域双分支做CTR预估，三数据集GAUC/AUC全面SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结+可学习双网络的子模块组合模拟指数级集成多样性，推理仅用单网络，NDCG@20 最高提升 9.70% 且收敛快 7.69 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/from_time_and_place_to_preference_llm_driven_geo_temporal_context_in_recomm/">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></strong></dt>
<dd>
<p>用 LLM 从时间戳+粗粒度位置生成地理-时间上下文嵌入，以特征融合或辅助对齐损失注入 SASRec，并提出轻量信息量诊断测试。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习消除空间结构冲突，全库排序下尾部长尾 H@10 最高提升 427%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong></dt>
<dd>
<p>用 Conv1D 局部混合、序列级 Key 门控软遗忘与门控 RMSNorm 将核化线性注意力扩展为双向线性时间算子，1:2 混合编码器在 Yandex Music 上追平双向自注意力，32768 长度单层提速 8.2×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/hyformer_revisiting_the_roles_of_sequence_modeling_and_feature_interaction_/">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></strong></dt>
<dd>
<p>用 Global Tokens 交替执行 Query Decoding 与 Query Boosting，统一长序列建模与特征交叉，AUC 提升 0.17% 且 FLOPs 仅 3.9T。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/kuailive_a_real_time_interactive_dataset_for_live_streaming_recommendation/">KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation</a></strong></dt>
<dd>
<p>首个含直播间生命周期与多行为实时交互的快手直播推荐数据集，构建 Top-K/CTR 基准并揭示主播长尾公平性挑战。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/l2rec_towards_dual_view_understanding_of_llms_for_personalized_recommendati/">L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation</a></strong></dt>
<dd>
<p>以参数级双视图 LoRA-MoE 替代输入/输出级融合，个性化路由对齐行为与语义视图，四数据集 N@10 提升 3.87%–8.02%，线上 CTR +9.24%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/learning_decomposed_contextual_token_representations_from_pretrained_and_co/">Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation</a></strong></dt>
<dd>
<p>DECOR 用冻结码本+可学协同嵌入分解融合、上下文 token 组合与可学 BOS 查询，缓解静态分词歧义与预训练语义丢失，Full-Ranking 下 NDCG@10 最高提升 14%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱食门控+TTNP自监督学习商品复购周期，PPRR降超60%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong></dt>
<dd>
<p>用可迁移物品增广+双阶段训练+域感知画像，缓解CDSR域不平衡与域迁移难题，全库排序下H@10最高提升26.96%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/mitigating_collaborative_semantic_id_staleness_in_generative_retrieval/">Mitigating Collaborative Semantic ID Staleness in Generative Retrieval</a></strong></dt>
<dd>
<p>提出模型无关的 SID 对齐更新，将刷新后的协同语义 ID 双部图匹配映射回旧词表，实现热启动微调，Recall@500 提升且训练算力降 8-9 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/mviger_multi_view_variational_integration_of_complementary_knowledge_for_ge/">MVIGER: Multi-View Variational Integration of Complementary Knowledge for Generative Recommender</a></strong></dt>
<dd>
<p>将模板-索引组合视为隐变量视图，用可学习先验做变分集成，Full-Ranking 下 H@10 最高提升约 26%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致模型评估失真，系统分析SSS生效条件与数据分布重塑机理。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_gener/">RCLRec: Reverse Curriculum Learning for Modeling Sparse Conversions in Generative Recommendation</a></strong></dt>
<dd>
<p>以逆向课程前缀为稀疏转化目标注入实例级中间监督，配合课程质量感知损失，离线全库排序全面超越 SOTA，线上广告收入 +2.09%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/recgpt_mobile_on_device_large_language_models_for_user_intent_understanding/">RecGPT-Mobile: On-Device Large Language Models for User Intent Understanding in Taobao Feed Recommendation</a></strong></dt>
<dd>
<p>将 Qwen3-0.6B 量化后部署于手机端，用意图漂移触发机制按需推理生成下一搜索意图，线上 GMV +2.5%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/recnexteval_a_reference_implementation_for_temporal_next_batch_recommendati/">RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation</a></strong></dt>
<dd>
<p>提出滑动时间窗 next-batch 评估框架，强制全局时间线防泄漏并支持增量学习，含 Python 库与 Web UI。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/recpfn_prior_fitted_networks_for_in_context_based_recommendations/">RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations</a></strong></dt>
<dd>
<p>用结构因果先验合成点击流预训练 PFN，单次前向即可零样本适配新域序列推荐，8 个基准 Full-Ranking 上零样本 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/rote_coarse_to_fine_multi_level_rotary_time_embedding_for_sequential_recomm/">RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation</a></strong></dt>
<dd>
<p>将时间戳分解为年/月/日多粒度并借 RoPE 旋转注入 Q/K，即插即用提升 SASRec/RPG，NDCG@5 最高 +20.11%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/spectran_spectral_aware_transformer_based_adapter_for_llm_enhanced_sequenti/">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用全谱注意力+可学习谱位置编码替代静态SVD截断，缓解LLM语义嵌入投影的维度坍缩，四数据集三骨干平均提升9.17%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/sprint_scalable_and_predictive_intent_refinement_for_llm_enhanced_session_b/">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></strong></dt>
<dd>
<p>用全局意图池约束 LLM 生成、以预测-纠错回路校验意图，并蒸馏为轻量意图预测器，实现训练选择性调用 LLM、推理零 LLM 依赖的会话推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong></dt>
<dd>
<p>TIDE 用 Hawkes 增强傅里叶时间编码刻画物品级非单调补货周期，并以双专家解耦习惯与探索意图，Leave-One-Out 全库排序下 R@10 最高提升 18.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/wpgrec_wavelet_packet_guided_graph_enhanced_sequential_recommendation/">WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用平稳小波包变换生成等长平移不变子带，在子带内做 Chebyshev 图传播并以能量-谱平坦度门控融合，Full-Ranking 下稀疏数据集 HR@20 最高 +15.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR-AP/2026/collaborative_filtering/lsnr_learned_sparse_news_recommendations_with_attentive_user_modeling/">LSNR: Learned Sparse News Recommendations with Attentive User Modeling</a></strong></dt>
<dd>
<p>用独立编码+注意力聚合替代 STAR 的历史拼接，构建双通路稀疏用户向量，MIND NDCG@10 较 STAR 提升 11%，短历史场景反超稠密模型。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/sequential_rec/an_adaptive_context_aware_and_stacked_attention_network_based_recommendatio/">An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users&#x27; Temporal Preference</a></strong></dt>
<dd>
<p>用上下文物品注意力+堆叠多头用户注意力捕捉用户时变偏好，Leave-One-Out+100负采样排序，三数据集HR/NDCG/MRR@10全面超越SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/generative_rec/accelerating_generative_recommendation_via_simple_categorical_user_sequence/">Accelerating Generative Recommendation via Simple Categorical User Sequence Compression</a></strong></dt>
<dd>
<p>用物品类别桶压缩长历史序列，在同等算力下较 HSTU 最高提升 39% 精度、最高降低 6 倍计算开销。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/sequential_rec/capturing_user_interests_from_data_streams_for_continual_sequential_recomme/">Capturing User Interests from Data Streams for Continual Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性注意力参数化记忆替代回放/正则，配合 Cauchy-Schwarz 归一化与共享兴趣池，在流式数据块上兼顾历史保留与新兴趣获取。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/sequential_rec/foresight_prediction_enhanced_live_streaming_recommendation/">Foresight Prediction Enhanced Live-Streaming Recommendation</a></strong></dt>
<dd>
<p>将直播切片经 LLM 语义量化成 Sid 序列，用 Transformer 编解码预测下一段内容，作为前瞻特征注入多任务精排，线上曝光 +0.46%、送礼 +2.48%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/collaborative_filtering/have_we_really_understood_collaborative_information_an_empirical_investigat/">Have We Really Understood Collaborative Information? An Empirical Investigation</a></strong></dt>
<dd>
<p>以物品共现路径长度量化协作信息为0/1/2-hop协作关系，系统实证其在六个基准上的分布、对12种推荐算法的影响与预测依赖，揭示模型强于简单CR、弱于复杂CR的核心规律。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/llm_rec/on_device_large_language_models_for_sequential_recommendation/">On-Device Large Language Models for Sequential Recommendation</a></strong></dt>
<dd>
<p>面向端侧序列推荐的 LLM 任务自适应压缩框架 OD-LLM，融合 Token 协方差归一化、SVD 低秩分解与逐层渐进对齐，50% 压缩率下推荐效果无损且推理提速 3.4 倍。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/self_supervised_denoising/simdiffrec_semantic_similarity_guided_diffusion_for_contrastive_sequential_/">SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义相似物品均值替代高斯随机噪声做扩散增强，按去噪置信度选增强位置并配难负采样，五数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/sequential_rec/time_aware_adaptive_side_information_fusion_for_sequential_recommendation/">Time-Aware Adaptive Side Information Fusion for Sequential Recommendation</a></strong></dt>
<dd>
<p>提出 TSP 时间跨度划分 + AFF 自适应频域滤波 + ASIF「引导不混合」融合层，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 10.28%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/generative_rec/beyond_the_flat_sequence_hierarchical_and_preference_aware_generative_recom/">Beyond the Flat Sequence: Hierarchical and Preference-Aware Generative Recommendations</a></strong></dt>
<dd>
<p>用会话层次预训练+偏好引导稀疏注意力打破 HSTU 扁平序列假设，APPGallery 全库 AUC 0.8929、线上 eCPM +1.99%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/blossomrec_block_level_fused_sparse_attention_mechanism_for_sequential_reco/">BlossomRec: Block-level Fused Sparse Attention Mechanism for Sequential Recommendations</a></strong></dt>
<dd>
<p>双路块级稀疏注意力（LTIS 块选择 + STIS 幂律掩码）经门控融合，长序列训练提速 3.2×、显存降至 1/11，精度持平或超越 SASRec。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/generative_rec/diffgrm_diffusion_based_generative_recommendation_model/">DiffGRM: Diffusion-based Generative Recommendation Model</a></strong></dt>
<dd>
<p>用掩码离散扩散替代自回归解码 SID，配合 OPQ 解耦分词、难度感知相干加噪与置信度并行去噪，NDCG@10 提升 6.9%–15.5%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/generative_rec/dualgr_generative_retrieval_with_long_and_short_term_interests_modeling/">DualGR: Generative Retrieval with Long and Short-Term Interests Modeling</a></strong></dt>
<dd>
<p>双分支长短期路由+桶内搜索式SID解码+曝光感知NTP损失，快手双列场景在线HR@1000达19.53%，A/B视频播放+0.527%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/generative_archetype_grounded_item_representations_for_sequential_recommend/">Generative Archetype-Grounded Item Representations for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 生成物品“目标人群原型”文本并单次前向抽取 prefill/decoding 双嵌入，再以行为关联分数调制排斥力做校准，Full-Ranking 下 HR@10 最高 +6.95%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/generative_data_transformation_from_mixed_to_unified_data/">Generative Data Transformation: From Mixed to Unified Data</a></strong></dt>
<dd>
<p>提出数据侧范式 Taesar，用三模型对比解码把跨域混合序列重生成纯目标域序列，免改架构即插即用，Books HR@10 提升 29.4%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/hyena_operator_for_fast_sequential_recommendation/">Hyena Operator for Fast Sequential Recommendation</a></strong></dt>
<dd>
<p>用 Legendre 正交多项式参数化 Hyena 长卷积核，线性复杂度建模超长用户序列，XLong 上 NDCG@10 提升 17.65% 且训练最高 6× 加速。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/is_more_context_always_better_examining_llm_reasoning_capability_for_time_i/">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></strong></dt>
<dd>
<p>系统评测 GPT-4o/Claude/Gemini 在复购时间间隔预测上的零样本能力，发现 LLM 仅胜统计基线、全面落后 XGBoost，且中等上下文最优、高上下文反降。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/learning_evolving_preferences_a_federated_continual_framework_for_user_cent/">Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation</a></strong></dt>
<dd>
<p>联邦持续推荐框架 FCUCR：时间感知自蒸馏抗遗忘 + 跨用户原型迁移促协同，四数据集全库排序 HR@10 最高提升 44.0%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/make_it_long_keep_it_fast_end_to_end_10k_long_user_behavior_sequence_modeli/">Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation</a></strong></dt>
<dd>
<p>以单查询目标-历史交叉注意力(STCA)将长序列建模降至 O(L)，配合请求级批处理(RLB)与稀疏训练稠密推理，在抖音十亿级流量实现 10K 序列端到端排序。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/mixture_of_sequence_theme_aware_mixture_of_experts_for_long_sequence_recomm/">Mixture of Sequence: Theme-Aware Mixture-of-Experts for Long-Sequence Recommendation</a></strong></dt>
<dd>
<p>揭示长序列“会话跳跃”现象，用主题码本路由+多尺度专家融合做子序列抽取，AUC 平均提升 0.68% 且 FLOPs 更低</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/modeling_stage_wise_evolution_of_user_interests_for_news_recommendation/">Modeling Stage-wise Evolution of User Interests for News Recommendation</a></strong></dt>
<dd>
<p>用全局LightGCN+阶段时序子图LSTM/自注意力双分支建模新闻兴趣演化，Adressa AUC 0.7993 大幅超越SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/onetrans_unified_feature_interaction_and_sequence_modeling_with_one_transfo/">OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</a></strong></dt>
<dd>
<p>用单一因果 Transformer 统一序列建模与特征交叉，配合金字塔剪枝与跨请求 KV 缓存，线上 GMV/u 提升 5.68%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multimodal_rec/personalized_parameter_efficient_fine_tuning_of_foundation_models_for_multi/">Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation</a></strong></dt>
<dd>
<p>按兴趣分群并为每群挂载独立 PEFT 模块，使 CLIP 编码出用户条件化的物品表征，配合群内难负采样，NDCG@20 最高提升 15.3%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/pi2i_a_personalized_item_based_collaborative_filtering_retrieval_framework/">PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework</a></strong></dt>
<dd>
<p>放宽 I2I 截断保召回、Trigger-Target 负采样对齐离线在线、目标注意力精排打分，淘宝线上成交率 +1.05%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multimodal_rec/prism_personalized_recommendation_via_information_synergy_module/">PRISM: Personalized Recommendation via Information Synergy Module</a></strong></dt>
<dd>
<p>用 MoE 显式分解模态唯一/冗余/协同信息并以用户偏好动态重加权，即插即用提升三类序列骨干，Yelp N@10 超最强基线 10.91%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/raie_region_aware_incremental_preference_editing_with_lora_for_llm_based_re/">RAIE: Region-Aware Incremental Preference Editing with LoRA for LLM-based Recommendation</a></strong></dt>
<dd>
<p>冻结 LLM 主干，用球面 k-means 构建偏好区域并为每区配专属 LoRA，通过 Update/Expand/Add 三种区域级编辑与置信路由实现抗遗忘的增量偏好适配。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cold_start_rec/tail_aware_data_augmentation_for_long_tail_sequential_recommendation/">Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性物品相似度构建候选集，设计 T-Substitute/T-Insert 两个尾部感知算子并在表征层 Mixup，实现头部与尾部双赢的长尾序列推荐数据增强。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/thinkrec_thinking_based_recommendation_via_llm/">ThinkRec: Thinking-based Recommendation via LLM</a></strong></dt>
<dd>
<p>以合成推理链激活 LLM 的 System 2 思维，并用实例级 LoRA 专家融合适配异构用户，AUC 平均提升 7.96%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/unleashing_the_potential_of_sparse_attention_on_long_term_behaviors_for_ctr/">Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction</a></strong></dt>
<dd>
<p>面向长行为序列 CTR 的个性化时间分块 + 三分支稀疏注意力 SparseCTR，AUC 相对提升 8.49%，线上 CTR +1.72%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/vk_lsvd_a_large_scale_industrial_dataset_for_short_video_recommendation/">VK-LSVD: A Large-Scale Industrial Dataset for Short-Video Recommendation</a></strong></dt>
<dd>
<p>发布 407 亿交互、千万用户、两千万短视频的工业级开放数据集，配全局时间切分与冷启动竞赛基准。</p>
</dd>
</dl>

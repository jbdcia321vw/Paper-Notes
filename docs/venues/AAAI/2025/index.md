---
hide:
  - toc
---

# 🏛️ AAAI (RecSys Track) 2025 推荐系统论文解读

31 篇 AAAI (RecSys Track) 2025 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/from_pairwise_to_ranking_climbing_the_ladder_to_ideal_collaborative_filteri/">From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking</a></strong></dt>
<dd>
<p>用噪声注入构造伪排序并以多序分类排序损失替代成对损失，在四个数据集 Full-Ranking 下 NDCG@10 最高提升 38.9%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (6)</h2>
<dl>
<dt><strong><a href="sequential_rec/augmenting_sequential_recommendation_with_balanced_relevance_and_diversity/">Augmenting Sequential Recommendation with Balanced Relevance and Diversity</a></strong></dt>
<dd>
<p>提出即插即用平衡增强插件 BASRec，以单序列随机算子+表征级 mixup 与跨序列非线性 mixup 兼顾相关性与多样性，全库排序下 SASRec 平均提升 33.8%。</p>
</dd>
<dt><strong><a href="sequential_rec/future_sight_and_tough_fights_revolutionizing_sequential_recommendation_wit/">Future Sight and Tough Fights: Revolutionizing Sequential Recommendation with FENRec</a></strong></dt>
<dd>
<p>用时间相关软标签挖掘未来交互、用锚-负混合构造持久困难负样本，四数据集全库排序平均提升 6.16%。</p>
</dd>
<dt><strong><a href="sequential_rec/lightweight_yet_fine_grained_a_graph_capsule_convolutional_network_with_sub/">Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性注意力胶囊图+低秩子空间对齐，在共享账户混合序列上做细粒度潜在用户解耦，精度大幅领先且参数/耗时更低。</p>
</dd>
<dt><strong><a href="sequential_rec/multifaceted_user_modeling_in_recommendation_a_federated_foundation_models_/">Multifaceted User Modeling in Recommendation: A Federated Foundation Models Approach</a></strong></dt>
<dd>
<p>联邦场景下用组门控网络将用户分层路由到组级 FFN，配合用户私有 FFN 实现多面向个性化，CTR 任务 AUC 最高提升 1.16%。</p>
</dd>
<dt><strong><a href="sequential_rec/preliminary_evaluation_of_the_test_time_training_layers_in_recommendation_s/">Preliminary Evaluation of the Test-Time Training Layers in Recommendation System (Student Abstract)</a></strong></dt>
<dd>
<p>将 TTT-Linear 作为序列推荐特征提取层，逐 token 梯度更新隐状态，在 Beauty/Electronics/ML-1M 上以 1 epoch 追平 SASRec/ComiRec 数十轮训练效果。</p>
</dd>
<dt><strong><a href="sequential_rec/sigma_selective_gated_mamba_for_sequential_recommendation/">SIGMA: Selective Gated Mamba for Sequential Recommendation</a></strong></dt>
<dd>
<p>用部分翻转双向 Mamba + 输入敏感稠密门控 + FE-GRU 解决 Mamba 单向性与短序列状态不稳问题，五数据集 Full-Ranking 上 HR@10 最高提升 8.82%。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (6)</h2>
<dl>
<dt><strong><a href="llm_rec/active_large_language_model_based_knowledge_distillation_for_session_based_/">Active Large Language Model-based Knowledge Distillation for Session-based Recommendation</a></strong></dt>
<dd>
<p>用 max-min 主动学习从全量会话中挑出少量高增益实例喂给 GPT-4 教师蒸馏，500 条样本即超越全量 KD 与 10 倍参数教师。</p>
</dd>
<dt><strong><a href="llm_rec/bridging_the_user_side_knowledge_gap_in_knowledge_aware_recommendations_wit/">Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 推断用户兴趣并结构化为 CIKG 异构图，配合兴趣重建与跨域对比学习去噪，在三个数据集全库排序下 R@100 最高提升 9.31%。</p>
</dd>
<dt><strong><a href="llm_rec/coherency_improved_explainable_recommendation_via_large_language_model/">Coherency Improved Explainable Recommendation via Large Language Model</a></strong></dt>
<dd>
<p>用 LLaMA2 先预测评分再以软评分嵌入引导解释生成，配合课程学习与无标注一致性自动评测，解释一致性提升 7.3%。</p>
</dd>
<dt><strong><a href="llm_rec/harnessing_multimodal_large_language_models_for_multimodal_sequential_recom/">Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 将商品图转为文本摘要，再以循环式提示推断动态用户偏好，SFT 微调 LLaVA 做多模态序列推荐，AUC 最高提升约 4.3%。</p>
</dd>
<dt><strong><a href="llm_rec/predicting_user_behavior_in_smart_spaces_with_llm_enhanced_logs_and_persona/">Predicting User Behavior in Smart Spaces with LLM-Enhanced Logs and Personalized Prompts</a></strong></dt>
<dd>
<p>用 LLM 语义增强智能空间事件日志 + 个性化图软提示，在智能车/智能家居四数据集上显著提升下一行为预测，稀有事件 H@1 提升超 40%。</p>
</dd>
<dt><strong><a href="llm_rec/re2llm_reflective_reinforcement_large_language_model_for_session_based_reco/">Re2LLM: Reflective Reinforcement Large Language Model for Session-based Recommendation</a></strong></dt>
<dd>
<p>用 LLM 自反思生成纠错提示构建知识库，再以 PPO 训练轻量检索智能体按会话选提示，冻结主干即超越全量训练基线。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/llm_powered_user_simulator_for_recommender_system/">LLM-Powered User Simulator for Recommender System</a></strong></dt>
<dd>
<p>用 LLM 离线蒸馏物品优缺点关键词，构建关键词匹配+语义相似+序列统计三模型集成用户模拟器，为 RL 推荐提供可解释高保真环境。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (5)</h2>
<dl>
<dt><strong><a href="multimodal_rec/befa_a_general_behavior_driven_feature_adapter_for_multimedia_recommendatio/">BeFA: A General Behavior-driven Feature Adapter for Multimedia Recommendation</a></strong></dt>
<dd>
<p>用行为特征引导的轻量适配器解耦-过滤-重构预训练内容特征，修复信息漂移与遗漏，全库排序下平均提升约 9-11%。</p>
</dd>
<dt><strong><a href="multimodal_rec/beyond_graph_convolution_multimodal_recommendation_with_topology_aware_mlps/">Beyond Graph Convolution: Multimodal Recommendation with Topology-aware MLPs</a></strong></dt>
<dd>
<p>用拓扑剪枝+邻域对齐损失替代GCN消息传递，以MLP建模多模态物品关系，全库排序下Recall@10最高提升7.53%</p>
</dd>
<dt><strong><a href="multimodal_rec/exploiting_fine_grained_skip_behaviors_for_micro_video_recommendation/">Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation</a></strong></dt>
<dd>
<p>将短视频跳过行为细分为高正/低正/负三级，构建双层正图与分层 BPR 损失，两数据集八项指标全面超越 FRAME 等基线。</p>
</dd>
<dt><strong><a href="multimodal_rec/mentor_multi_level_self_supervised_learning_for_multimodal_recommendation/">MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation</a></strong></dt>
<dd>
<p>以 ID 嵌入为锚做四级跨模态分布对齐，叠加特征掩码与图扰动自监督，Amazon 三集 Recall@20 最高提升 74.73%</p>
</dd>
<dt><strong><a href="multimodal_rec/stair_manipulating_collaborative_and_multimodal_information_for_e_commerce_/">STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation</a></strong></dt>
<dd>
<p>揭示电商场景模态信息弱于协同信号，提出逐步图卷积沿维度分配层权重实现协同-模态共存，并以反向逐步卷积约束更新缓解模态遗忘，三数据集全排序提升2-6%且开销近LightGCN。</p>
</dd>
</dl>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl>
<dt><strong><a href="social_group_rec/disentangled_modeling_of_preferences_and_social_influence_for_group_recomme/">Disentangled Modeling of Preferences and Social Influence for Group Recommendation</a></strong></dt>
<dd>
<p>解耦用户偏好与社会影响的双通道(超)图传播，配合社交重要性引导的对比学习，缓解群组推荐的偏好偏差与数据稀疏。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/bayescns_a_unified_bayesian_approach_to_address_cold_start_and_non_stationa/">BayesCNS: A Unified Bayesian Approach to Address Cold Start and Non-Stationarity in Search Systems at Scale</a></strong></dt>
<dd>
<p>用经验贝叶斯先验网络+Thompson采样在线更新，统一解决物品冷启动与非平稳漂移，线上A/B新物品曝光+10.60%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/behavior_importance_aware_graph_neural_architecture_search_for_cross_domain/">Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用一次性免重训超网为各域定制 GNN 架构，并以双层隐式梯度学习源域行为重要性权重，Amazon 与微信广告数据 AUC 最高提升 10.57%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/one_for_dozens_adaptive_recommendation_for_all_domains_with_counterfactual_/">One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation</a></strong></dt>
<dd>
<p>用分层专家+彩票假设掩码剪枝为数十域学专属子网，并以流行度反事实增强少数域，DomainAUC 提升 2‰~8.4‰。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/spatial_temporal_knowledge_distillation_for_takeaway_recommendation/">Spatial-Temporal Knowledge Distillation for Takeaway Recommendation</a></strong></dt>
<dd>
<p>用 STKG 教师模型蒸馏时空知识给 ST-Transformer 学生模型，融合地理距离与功能区序列建模外卖动态偏好，三城 HR@10 提升约 1.5–2.8%。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/direct_routing_gradient_drgrad_a_personalized_information_surgery_for_multi/">Direct Routing Gradient (DRGrad): A Personalized Information Surgery for Multi-Task Learning (MTL) Recommendations</a></strong></dt>
<dd>
<p>用梯度余弦相似度做路由+Split-MMoE+PPNet个性化门控，在150亿样本工业数据上Click AUC提升0.0027且不增推理延迟。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/contrastive_representation_for_interactive_recommendation/">Contrastive Representation for Interactive Recommendation</a></strong></dt>
<dd>
<p>用偏好排序对比学习(PRCL)为 DRL 交互推荐提炼状态表征，冷启动下样本效率显著提升。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/neural_combinatorial_clustered_bandits_for_recommendation_systems/">Neural Combinatorial Clustered Bandits for Recommendation Systems</a></strong></dt>
<dd>
<p>用双神经网络+上下文聚类替代组合老虎机的精确优化 oracle，实现 $\tilde{O}(\tilde{d}\sqrt{T})$ 遗憾并显著降低推荐累积遗憾。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/an_xai_social_media_platform_for_teaching_k_12_students_ai_driven_profiling/">An XAI Social Media Platform for Teaching K-12 Students AI-Driven Profiling, Clustering, and Engagement-Based Recommending</a></strong></dt>
<dd>
<p>面向 K-12 的 XAI 社交媒体教学平台，实时可视化数据采集、画像聚类与推荐机制，209 名学生实测揭示三类使用画像。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/bites_of_tomorrow_personalized_recommendations_for_a_healthier_and_greener_/">Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate</a></strong></dt>
<dd>
<p>提出绿色食品推荐任务 GRAPE，用双注意力建模用户对可持续指标的动态偏好，并以两种 Green Loss 在保精度前提下提升推荐列表绿色度。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/multi_modal_recommendation_unlearning_for_legal_licensing_and_modality_cons/">Multi-Modal Recommendation Unlearning for Legal, Licensing, and Modality Constraints</a></strong></dt>
<dd>
<p>首个多模态推荐遗忘框架 MMRecUn，用反向 BPR 衰减遗忘集、正向 BPR 保真保留集，Recall 最高提升 49.85% 且比从头重训快 1.3 倍。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/personalized_federated_collaborative_filtering_a_variational_autoencoder_ap/">Personalized Federated Collaborative Filtering: A Variational AutoEncoder Approach</a></strong></dt>
<dd>
<p>将 FedCF 重构为 VAE 任务，用门控双编码器分离共享/个性化知识，HR@20 较 FedRAP 提升约 97%。</p>
</dd>
</dl>
</div>

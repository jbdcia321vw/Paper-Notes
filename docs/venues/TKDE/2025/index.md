---
hide:
  - toc
---

# 📖 IEEE TKDE 2025 推荐系统论文解读

25 篇 IEEE TKDE 2025 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/intent_propagation_contrastive_collaborative_filtering/">Intent Propagation Contrastive Collaborative Filtering</a></strong></dt>
<dd>
<p>双螺旋消息传播融合高阶关系与意图解耦，对比学习对齐结构/意图视图，Full-Ranking 下 NDCG 平均提升 9.36%</p>
</dd>
<dt><strong><a href="collaborative_filtering/latent_factor_modeling_with_expert_network_for_multi_behavior_recommendatio/">Latent Factor Modeling with Expert Network for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用门控专家网络显式建模共享潜在因子，配合对比学习保证因子独立与一致，在三个多行为数据集 Full-Ranking 下 HR@10 最高提升 3.95%。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl>
<dt><strong><a href="graph_rec/dual_channel_multiplex_graph_neural_networks_for_recommendation/">Dual-Channel Multiplex Graph Neural Networks for Recommendation</a></strong></dt>
<dd>
<p>双通道多路图网络显式建模行为模式、隐式挖掘关系链依赖，三数据集全指标平均提升 10%+</p>
</dd>
<dt><strong><a href="graph_rec/graph_cross_correlated_network_for_recommendation/">Graph Cross-Correlated Network for Recommendation</a></strong></dt>
<dd>
<p>用非递归 PGR 保留各跳邻居纯语义，再以 HCC/ECC 显式建模用户-物品子图交叉相关项，全库排序下 NDCG@20 平均提升约 1.8%–2.9%。</p>
</dd>
<dt><strong><a href="graph_rec/intent_guided_heterogeneous_graph_contrastive_learning_for_recommendation/">Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用元路径意图构建双对比学习视图，并以信息瓶颈掩码自编码器去噪，六数据集全库排序 Recall@20 最高提升 12.4%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl>
<dt><strong><a href="sequential_rec/explainable_session_based_recommendation_via_path_reasoning/">Explainable Session-based Recommendation via Path Reasoning</a></strong></dt>
<dd>
<p>提出分层强化学习框架 PR4SR，用会话级智能体选起点、路径级智能体做知识图谱推理，配合多目标与路径中点奖励，为 5 种 SR 模型提供可解释路径并提升精度。</p>
</dd>
<dt><strong><a href="sequential_rec/improving_sequential_recommendations_via_bidirectional_temporal_data_augmen/">Improving Sequential Recommendations via Bidirectional Temporal Data Augmentation with Pre-training</a></strong></dt>
<dd>
<p>BARec 用双向时序增强预训练生成保偏好伪先验物品，并以 KL 知识增强微调对齐原/增广序列分布，全库排序下 Beauty Recall@5 提升 16.6%。</p>
</dd>
<dt><strong><a href="sequential_rec/pattern_wise_transparent_sequential_recommendation/">Pattern-wise Transparent Sequential Recommendation</a></strong></dt>
<dd>
<p>用多尺度滑窗切出点级/联合级模式，以概率嵌入+逻辑合取建模模式，KL散度+距离权重实现可解释且超越黑盒的序列推荐。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl>
<dt><strong><a href="generative_rec/tokenrec_learning_to_tokenize_id_for_llm_based_generative_recommendation/">TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation</a></strong></dt>
<dd>
<p>用掩码K路向量量化把GNN协同表征压成LLM兼容离散token，并以生成式检索替代自回归解码，HR@20平均提升约19%、推理加速超10倍。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl>
<dt><strong><a href="llm_rec/collm_integrating_collaborative_embeddings_into_large_language_models_for_r/">CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>将传统协同过滤模型学到的用户/物品表征经 MLP 映射进 LLM 词嵌入空间，以两阶段调优兼顾暖启动与冷启动，AUC 相对 TALLRec 提升约 8%。</p>
</dd>
<dt><strong><a href="llm_rec/enhancing_collaborative_semantics_of_language_model_driven_recommendations_/">Enhancing Collaborative Semantics of Language Model-Driven Recommendations via Graph-Aware Learning</a></strong></dt>
<dd>
<p>用多跳提示+图感知对比学习把 GNN 聚合意图注入 LLM，在三个 Amazon/Yelp 数据集 Full-Ranking 下 HR@10 最高提升约 15%。</p>
</dd>
<dt><strong><a href="llm_rec/exact_and_efficient_unlearning_for_large_language_model_based_recommendatio/">Exact and Efficient Unlearning for Large Language Model-based Recommendation</a></strong></dt>
<dd>
<p>首个面向 LLMRec 的精确遗忘框架 APA：语义分片训练子 LoRA、仅重训受影响分片，并以样本自适应参数级聚合实现单次推理，遗忘提速约 4 倍且 AUC 反超全量重训。</p>
</dd>
<dt><strong><a href="llm_rec/preference_consistent_knowledge_distillation_for_recommender_system/">Preference-Consistent Knowledge Distillation for Recommender System</a></strong></dt>
<dd>
<p>揭示特征蒸馏中投影器导致的偏好不一致现象，提出成对/列表两种正则项约束投影器，全库排序下 NDCG@20 最高提升 20%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/efficient_and_effective_adaptation_of_multimodal_foundation_models_in_seque/">Efficient and Effective Adaptation of Multimodal Foundation Models in Sequential Recommendation</a></strong></dt>
<dd>
<p>IISAN-Versa 用解耦 PEFT + 组 LayerDrop + 维度变换层，统一适配对称/非对称多模态基础模型，缓存隐藏态后 GPU 显存降 12–15 倍、训练提速 16–20 倍，MicroLens 达 SOTA。</p>
</dd>
</dl>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl>
<dt><strong><a href="social_group_rec/score_based_generative_diffusion_models_for_social_recommendations/">Score-based Generative Diffusion Models for Social Recommendations</a></strong></dt>
<dd>
<p>用 SDE 扩散模型在隐空间直接生成协同一致的社交表征，配合课程训练与跨域对比学习，Full-Ranking 下 Recall@10 相对提升 1.85%–3.07%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/automated_prompting_for_non_overlapping_cross_domain_sequential_recommendat/">Automated Prompting for Non-overlapping Cross-domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用可学习连续 Prompt（域无关+域特定+冻结标签）替代域对齐约束，实现无重叠实体、仅隐式反馈的双目标跨域序列推荐，Movie-Book 上 HR@10 相对最优基线提升约 17.8%。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/a_survey_on_point_of_interest_recommendation_models_architectures_and_secur/">A Survey on Point-of-Interest Recommendation: Models, Architectures, and Security</a></strong></dt>
<dd>
<p>首个贯通模型演进、架构部署与安全隐私三维度的POI推荐综述，覆盖LDA到LLM/DM、集中式到联邦式及投毒/同态加密防护。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/aefs_adaptive_early_feature_selection_for_deep_recommender_systems/">AEFS: Adaptive Early Feature Selection for Deep Recommender Systems</a></strong></dt>
<dd>
<p>用 mini-AdaFS 辅助模型在嵌入层前做实例级自适应特征选择，配合双对齐损失，精度持平 AdaFS 且激活嵌入参数降 37.5%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/uncertainty_calibration_for_counterfactual_propensity_estimation_in_recomme/">Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation</a></strong></dt>
<dd>
<p>揭示倾向分估计严重失准问题，提出以ECE为质量度量的模型无关校准框架，理论证明校准倾向分降低IPS偏差，在Coat/Yahoo/KuaiRand上显著提升CVR去偏效果。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/reval_a_semantic_evaluation_framework_for_hashtag_recommendation/">#REVAL: a semantic evaluation framework for hashtag recommendation</a></strong></dt>
<dd>
<p>提出语义评估框架 #REval，用 BERTag 话题标签嵌入构建同义词库，以 #REval-hit-ratio 替代精确匹配，修正话题标签推荐评估的系统性低估。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/dual_test_time_training_for_out_of_distribution_recommender_system/">Dual Test-time Training for Out-of-distribution Recommender System</a></strong></dt>
<dd>
<p>首个将测试时训练引入 OOD 推荐，用自蒸馏+对比双任务在测试阶段适配用户/物品特征漂移，全库排序下 R@50 最高提升 11.56%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/fairdgcl_fairness_aware_recommendation_with_dynamic_graph_contrastive_learn/">FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning</a></strong></dt>
<dd>
<p>用对抗式视图生成器+判别器动态生成公平增强视图，在四个数据集上同时提升准确率与群体公平性。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/fairsort_learning_to_fair_rank_for_personalized_recommendations_in_two_side/">FairSort: Learning to Fair Rank for Personalized Recommendations in Two-Sided Platforms</a></strong></dt>
<dd>
<p>将重排视为“跑道”而非背包，用公平提升速度+二分搜索保证用户最低效用，实现双边公平与个性化推荐的三方权衡。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/personalized_recommendation_models_in_federated_settings_a_survey/">Personalized Recommendation Models in Federated Settings: A Survey</a></strong></dt>
<dd>
<p>首篇以“个性化模型”为核心的 FedRecSys 综述，给出双层优化定义并系统梳理四类挑战与解法。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/towards_robust_recommendation_a_review_and_an_adversarial_robustness_evalua/">Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library</a></strong></dt>
<dd>
<p>系统综述推荐系统对抗/非对抗鲁棒性，提出 ShillingREC 评测库统一攻击与防御模型评估。</p>
</dd>
</dl>
</div>

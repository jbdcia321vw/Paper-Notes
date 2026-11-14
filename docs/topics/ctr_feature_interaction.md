# 🎯 CTR/CVR 预估与特征交叉 (CTR Prediction & Feature Interaction)

> **研究范畴**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

当前方向共收录 **155** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/cross_domain_multi_scenario/one_for_dozens_adaptive_recommendation_for_all_domains_with_counterfactual_/">One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation</a></strong></dt>
<dd>
<p>用分层专家+彩票假设掩码剪枝为数十域学专属子网，并以流行度反事实增强少数域，DomainAUC 提升 2‰~8.4‰。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/ctr_feature_interaction/length_adaptive_interest_network_for_balancing_long_and_short_sequence_mode/">Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction</a></strong></dt>
<dd>
<p>将序列长度作为条件信号注入提示与注意力温度，缓解短序列用户注意力极化，五骨干平均AUC最高+1.15%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/the_efficiency_vs_accuracy_trade_off_optimizing_rag_enhanced_llm_recommende/">The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit</a></strong></dt>
<dd>
<p>用 GCN 检索器替代 LLM embedder 加速 RAG，并以多头早退动态终止 LLM 推理，在 5 个 CTR 数据集上兼顾 AUC 与吞吐。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/a_soft_partitioned_semi_supervised_collaborative_transfer_learning_approach/">A Soft-partitioned Semi-supervised Collaborative Transfer Learning Approach for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用软分区动态参数缓解主导域压制、用主导域伪标签加权迁移缓解稀疏域过拟合，离线全库AUC与美团线上A/B双验证。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/generative_rec/asymmetric_diffusion_recommendation_model/">Asymmetric Diffusion Recommendation Model</a></strong></dt>
<dd>
<p>用特征丢弃替代高斯噪声做离散前向、在隐空间做非对称反向去噪，抖音音乐全量A/B活跃天数+0.131%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/sequential_rec/climber_toward_efficient_scaling_laws_for_large_recommendation_models/">Climber: Toward Efficient Scaling Laws for Large Recommendation Models</a></strong></dt>
<dd>
<p>面向推荐场景改造 Transformer：多尺度子序列抽取降复杂度、自适应温度调制注意力，配合单用户多物品批处理与 KV 缓存，线上指标 +12.19%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/distribution_guided_auto_encoder_for_user_multimodal_interest_cross_fusion/">Distribution-Guided Auto-Encoder for User Multimodal Interest Cross Fusion</a></strong></dt>
<dd>
<p>以行为粒度做模态内/模态间交叉融合，并用兴趣分布解码器作自监督约束，工业 A/B 上 CTR +12.9%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/global_distribution_aware_scenario_specific_variational_representation_lear/">Global-Distribution Aware Scenario-Specific Variational Representation Learning Framework</a></strong></dt>
<dd>
<p>用变分推断为每个场景下的用户/物品学习分布表征，并以全局行为模式与物品属性构造先验，缓解稀疏场景表征欠学习，即插即用提升 PEPNet/MARIA。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cold_start_rec/harnessing_light_for_cold_start_recommendations_leveraging_epistemic_uncert/">Harnessing Light for Cold-Start Recommendations: Leveraging Epistemic Uncertainty to Enhance Performance in User-Item Interactions</a></strong></dt>
<dd>
<p>用集成 CVAE 生成暖化嵌入，并以 Sinkhorn 散度版 PaiDE 估计认知不确定性做优化，在冷启动 CTR 上稳定超越 CVAR 等基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/hit_model_a_hierarchical_interaction_enhanced_two_tower_model_for_pre_ranki/">HIT Model: A Hierarchical Interaction-Enhanced Two-Tower Model for Pre-Ranking Systems</a></strong></dt>
<dd>
<p>用双生成器预生成跨塔粗粒度向量+多头子空间细粒度匹配，在保持双塔预缓存效率下逼近全交互模型精度，腾讯线上GMV+1.66%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/interformer_effective_heterogeneous_interaction_learning_for_click_through_/">InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>提出交错式双向信息流的 InterFormer，用独立 Cross Arch 做选择性聚合，在三个公开集与 Meta 工业集上取得 SOTA，线上 NE 提升 0.15% 且 QPS 提升 24%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/generative_rec/mtgr_industrial_scale_generative_recommendation_framework_in_meituan/">MTGR: Industrial-Scale Generative Recommendation Framework in Meituan</a></strong></dt>
<dd>
<p>以用户级样本聚合复用 HSTU 编码器并保留 DLRM 交叉特征，实现 65× FLOPs 下线上 CTCVR +1.02%、推理成本降 12%</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multi_task_reranking/orca_mitigating_over_reliance_for_multi_task_dwell_time_prediction_with_cau/">ORCA: Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling</a></strong></dt>
<dd>
<p>用因果解耦显式建模并减去CTR对停留时间预测的负迁移，恢复中时长分桶语义，DT指标平均提升10.6%且不损CTR。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/personalized_tree_based_progressive_regression_model_for_watch_time_predict/">Personalized Tree-Based Progressive Regression Model for Watch-Time Prediction in Short Video Recommendation</a></strong></dt>
<dd>
<p>将TPM固定满二叉树改为样本级可学习剪枝树，并用IPS校正条件建模选择偏差，离线MAE/XAUC全面领先，线上观看时长+0.428%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/llm_rec/prompt_tuning_as_user_inherent_profile_inference_machine/">Prompt Tuning as User Inherent Profile Inference Machine</a></strong></dt>
<dd>
<p>将用户潜在画像建模为可训练软提示，用因果掩码+EM 推断、码本量化成协同 ID，AUC 最高相对提升 4.77%，已上线华为应用市场。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/rankmixer_scaling_up_ranking_models_in_industrial_recommenders/">RankMixer: Scaling Up Ranking Models in Industrial Recommenders</a></strong></dt>
<dd>
<p>用无参多头Token Mixing替代自注意力、Per-token FFN隔离特征子空间，配合ReLU路由稀疏MoE，在抖音万亿级数据上将MFU从4.5%提至45%，参数扩100倍而延迟不增。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/scenario_wise_rec_a_multi_scenario_recommendation_benchmark/">Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark</a></strong></dt>
<dd>
<p>首个多场景推荐统一基准，整合 6 个公开数据集与 12 个 MSR 模型，统一数据处理/训练/评测管线并附工业广告数据验证。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/see_beyond_a_single_view_multi_attribution_learning_leads_to_better_convers/">See Beyond a Single View: Multi-Attribution Learning Leads to Better Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将多归因标签作为辅助任务联合学习，用笛卡尔积辅助训练增强监督信号，离线GAUC+0.51%、线上ROI+2.6%</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/llm_rec/self_surrogate_light_feature_selection_with_large_language_models_in_deep_r/">SELF: Surrogate-light Feature Selection with Large Language Models in Deep Recommender Systems</a></strong></dt>
<dd>
<p>用多 LLM 迭代提示生成语义特征排序，再以可学习桥接向量对齐任务空间，实现轻量特征选择，AUC 全面领先且线上 CTR +3.01%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/llm_rec/trawl_external_knowledge_enhanced_recommendation_with_llm_assistance/">TRAWL: External Knowledge-Enhanced Recommendation with LLM Assistance</a></strong></dt>
<dd>
<p>用 LLM 从原始外部文本抽取推荐知识，再以对比学习适配器对齐语义与行为空间，ML-1M 上 AUC 稳定提升且微信线上 A/B 三项指标正向。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/bidding_aware_retrieval_for_multi_stage_consistency_in_online_advertising/">Bidding-Aware Retrieval for Multi-Stage Consistency in Online Advertising</a></strong></dt>
<dd>
<p>将竞价信号注入召回打分，用单调性约束+多任务蒸馏+任务注意力精炼实现召回-精排 eCPM 一致性，线上收入 +4.32%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/can_large_language_models_identify_meaningful_touchpoints_in_conversion_att/">Can Large Language Models Identify Meaningful Touchpoints in Conversion Attribution?</a></strong></dt>
<dd>
<p>构建SILVA触点选择基准，揭示隐式触点语义鸿沟，系统评测8个LLM的成对/列表提示能力，并以LLM归因标签增强工业CVR训练获+0.35pp GAUC。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/closing_the_long_short_view_gap_in_sequential_recommendation_without_cached/">Closing the Long-Short View Gap in Sequential Recommendation without Cached History</a></strong></dt>
<dd>
<p>用角度相似度打分+softmax1 修正长视训练几何缺陷，再仅微调 Bias/LayerNorm 实现无缓存短视推理，短窗下反超全序列基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/funnelcausalnet_funnel_aware_joint_conversion_revenue_uplift_for_multi_tier/">FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation</a></strong></dt>
<dd>
<p>以漏斗恒等式 μ_gmv=μ_conv·μ_val 耦合双头 uplift 估计，配 RCT 锚定拉格朗日预算分配与 Bonferroni 联合保形审计带，工业多档券 RCT 上 7/7 锚点 ΔROI 最优。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/jointly_optimizing_debiased_ctr_and_uplift_for_coupons_marketing_a_unified_/">Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing: A Unified Causal Framework</a></strong></dt>
<dd>
<p>UniMVT 用 MoE 解耦混淆表征 + logit 空间单调线性假设，统一估计去偏 Base CTR 与多值券面额 Uplift，线上收入 +18.14%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/latent_residual_reasoning_for_ranking_in_generative_recommendation/">Latent Residual Reasoning for Ranking in Generative Recommendation</a></strong></dt>
<dd>
<p>在冻结的生成式排序骨干上叠加 LoRA 与残差推理头，用多步隐空间加性修正精化候选打分，公开集 GAUC 提升 2.7%–4.3%，线上 PV-CTR +0.75%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/lfm4ads_large_foundation_model_for_ads_recommendation/">LFM4Ads: Large Foundation Model for Ads Recommendation</a></strong></dt>
<dd>
<p>腾讯广告提出预训练-迁移范式，联合迁移用户/物品/交叉表征，以时间感知EMA聚合样本级CR解耦上游推理，三级粒度下游复用，离线AUC+0.0189、线上GMV+2.45%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/native_multimodal_representation_learning_for_click_through_rate_prediction/">Native Multimodal Representation Learning for Click-Through Rate Prediction in E-Commerce Scenarios</a></strong></dt>
<dd>
<p>揭示端到端联合训练在强预训练编码器上失效的根因（CTR 监督多模态不可解释+梯度不一致），提出 Mine-Then-Train 挖掘三元组微调编码器，线上 CTR +1.5%、RPM +0.5%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/personalized_task_dependency_graphs_for_mitigating_signal_erosion_in_multi_/">Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation</a></strong></dt>
<dd>
<p>用低秩个性化任务依赖图+硬因果掩码GCN替代刚性漏斗，缓解深层稀疏任务信号侵蚀，离线AUC最高+1.45%，线上CVR+1.2%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/profiling_what_matters_context_aware_item_profiles_from_large_scale_metadat/">Profiling What Matters: Context-Aware Item Profiles from Large-Scale Metadata for LLM Recommenders</a></strong></dt>
<dd>
<p>将海量异构物品元数据结构化并训练轻量选择器，为每个用户-物品对动态生成上下文相关画像，显著提升 LLM 重排效果。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/recsoup_efficient_model_merging_approach_for_controllable_multi_objective_r/">RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation</a></strong></dt>
<dd>
<p>用剪枝+可学习门控做任务向量合并，实现推理时可控多目标推荐，HV 提升 8.5% 且延迟快 165–228 倍。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/routerec_behavior_guided_sparse_routing_for_sequential_recommendation/">RouteRec: Behavior-Guided Sparse Routing for Sequential Recommendation</a></strong></dt>
<dd>
<p>用会话行为线索（节奏/焦点/记忆/流行度）驱动分层稀疏 MoE 路由，六数据集 18 指标平均排名 1.61 显著优于次优 4.11。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/sort_a_systematically_optimized_ranking_transformer_for_industrial_scale_re/">SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders</a></strong></dt>
<dd>
<p>以请求为中心组织样本、局部注意力+查询剪枝、生成式预训练与稀疏MoE重构工业精排Transformer，线上GMV+8.65%且延迟降62%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/task_blind_no_more_multi_task_information_flow_in_unified_ranking_backbones/">Task-Blind No MORE: Multi-Task Information Flow in Unified Ranking Backbones</a></strong></dt>
<dd>
<p>将多任务信息流下沉至统一排序主干，以共享/私有 Anchor Token 逐层共演化，离线七任务全胜、线上时长+3.0%、P99 延迟降约30%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/uniscale_synergistic_entire_space_data_and_model_scaling_for_search_ranking/">UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking</a></strong></dt>
<dd>
<p>提出数据-架构协同缩放框架：ES³全空间样本系统扩展训练信号，HHSFT异构分层Transformer吸收跨域分布，离线AUC+1.14%、线上GMV+2.04%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/collaborative_filtering/a_situation_aware_enhancer_for_personalized_recommendation/">A Situation-aware Enhancer for Personalized Recommendation</a></strong></dt>
<dd>
<p>将情境视为交互前置条件，用可插拔双塔增强器 SARE 建模个性化情境感知与偏好影响，在 MIND/KuaiRand 上显著超越骨干与 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/debias_fairness_industrial/modeling_behavioral_patterns_in_news_recommendations_using_fuzzy_neural_net/">Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks</a></strong></dt>
<dd>
<p>用多层模糊神经网络端到端学习可读逻辑规则预测新闻点击，阈值可调控制规则复杂度，EB-NeRD 上 nDCG@10 相对 NRMS 提升约 21.9%。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2025/ctr_feature_interaction/graphex_a_graph_based_extraction_method_for_advertiser_keyphrase_recommenda/">GraphEx: A Graph-based Extraction Method for Advertiser Keyphrase Recommendation</a></strong></dt>
<dd>
<p>用二分图从商品标题抽取关键词排列，绕开点击数据偏差，CPU 毫秒级推理，线上广告收入提升 8.3%。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2026/llm_rec/relevance_matters_a_multi_task_and_multi_stage_large_language_model_approac/">Relevance Matters: A Multi-Task and Multi-Stage Large Language Model Approach for E-commerce Query Rewriting</a></strong></dt>
<dd>
<p>将相关性打标任务注入电商查询改写，多任务 SFT + 查询级归一化 GRPO 融合奖励对齐，离线召回 67.94%、线上 UCVR +0.181%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/cross_domain_multi_scenario/adversarial_alignment_and_disentanglement_for_cross_domain_ctr_prediction_w/">Adversarial Alignment and Disentanglement for Cross-Domain CTR Prediction with Domain-Encompassing Features</a></strong></dt>
<dd>
<p>提出 A²DCDR，用 DC-MMD 对抗对齐 + CLUB 互信息解耦 + 特征重构，保留域不变与有价值非对齐特征，四组 Amazon 场景 HR@10 最高 +6.96，线上 CTR +7.18%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/ctr_feature_interaction/on_predicting_post_click_conversion_rate_via_counterfactual_inference/">On Predicting Post-Click Conversion Rate via Counterfactual Inference</a></strong></dt>
<dd>
<p>用 SCM 对未点击样本做 do(C=1) 反事实推断生成转化伪标签，全空间训练 CVR 模型，离线 AUC 平均 +1.01%，线上 CVR +17.35%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/rl_bandit_rec/learning_personalized_ad_impact_via_contextual_reinforcement_learning_under/">Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards</a></strong></dt>
<dd>
<p>将自动出价建模为带延迟泊松奖励的CMDP，提出两阶段MLE与数据分割，实现个性化出价近最优遗憾界。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/sequential_rec/long_sequence_recommendation_models_need_decoupled_embeddings/">Long-Sequence Recommendation Models Need Decoupled Embeddings</a></strong></dt>
<dd>
<p>揭示长序列推荐中注意力与表征共享嵌入导致的梯度支配与冲突，提出双嵌入表 DARE，AUC 最高提升 9‰ 并加速检索 50%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/optimal_auction_design_in_the_joint_advertising/">Optimal Auction Design in the Joint Advertising</a></strong></dt>
<dd>
<p>联合广告单槽位Myerson最优机制+多槽位BundleNet束级IC约束神经网络，逼近理论最优并显著提升平台收益</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/causal_representation_learning_for_generalisable_recommendation/">Causal Representation Learning for Generalisable Recommendation</a></strong></dt>
<dd>
<p>提出信息论解耦准则 J=I(g(T);Y|Z)−λI(g(T);Z)，用 InfoNCE+梯度反转实现，零推理开销，Spotify 千万级 A/B 提升 +0.75% 播放。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/lime_link_based_user_item_interaction_modeling_with_decoupled_xor_attention/">LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling</a></strong></dt>
<dd>
<p>用低秩 link embedding 解耦用户-候选交互并预计算 QK 权重，配合 XOR 线性注意力将长历史建模降至 O(N)，实现 HSTU 级精度与 10× 推理加速。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/model_monotonicity_in_autobidding_auctions_when_do_better_predictions_lead_/">Model Monotonicity in Autobidding Auctions: When Do Better Predictions Lead to Better Outcomes?</a></strong></dt>
<dd>
<p>用划分细化刻画模型改进，系统证明自动出价拍卖中仅三种设定保证平台指标单调，其余均可反例击穿。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/on_the_coordination_of_value_maximizing_bidders/">On the Coordination of Value-Maximizing Bidders</a></strong></dt>
<dd>
<p>证明价值最大化自动出价方在“最高价值者出价”协调机制下，RoS 合规与联盟总价值均优于独立出价，并给出充要条件与渐近最优性。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/llm_rec/agent0_leveraging_llm_agents_to_discover_multi_value_features_from_text_for/">Agent0: Leveraging LLM Agents to Discover Multi-value Features from Text for Enhanced Recommendations</a></strong></dt>
<dd>
<p>用 Architect-Sentinel-Oracle 三角色 LLM 智能体闭环自动挖掘文本多值类别特征，以 DCNv2+RIG 为 Oracle 在真实广告生产数据上取得 0.005+ RIG 提升并提速 2-3 倍。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/rl_bandit_rec/an_adaptable_budget_planner_for_enhancing_budget_constrained_auto_bidding_i/">An Adaptable Budget Planner for Enhancing Budget-Constrained Auto-Bidding in Online Advertising</a></strong></dt>
<dd>
<p>将竞价周期切分为多阶段，用 in-context RL 逐 episode 调整预算分配，few-shot 适配新广告主，仿真与线上 A/B 均提升累计价值。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/an_audio_centric_multi_task_learning_framework_for_streaming_ads_targeting_/">An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify</a></strong></dt>
<dd>
<p>面向音频主导多模态广告场景的 CAMoE 框架，以模态感知任务分组、自适应损失掩码与 DCN 专家实现近帕累托最优 CTR 预估，线上音频 CTR +14.5%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/beyond_advertising_mechanism_design_for_platform_wide_marketing_service_qua/">Beyond Advertising: Mechanism Design for Platform-Wide Marketing Service &#x27;QuanZhanTui&#x27;</a></strong></dt>
<dd>
<p>提出全站推场景下的库存约束卖家模型与流式支付拍卖 LPA，在合成与真实数据上同时提升流式福利、用户体验与平台收入。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/bid2x_revealing_dynamics_of_bidding_environment_in_online_advertising_from_/">Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens</a></strong></dt>
<dd>
<p>将自动出价环境建模重构为跨场景基础模型的下一token预测，用变量/时间双注意力+零膨胀投影，8数据集MAE全面领先，线上GMV+4.65%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/breaker_removing_shortcut_cues_with_user_clustering_for_single_slot_recomme/">Breaker: Removing Shortcut Cues with User Clustering for Single-slot Recommendation System</a></strong></dt>
<dd>
<p>用用户表征聚类辅助任务+多塔簇专属偏好建模，破除单槽推荐中用户固有倾向的捷径偏置，离线Recall@1提升7%并上线美团支付。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/contextual_generative_auction_with_permutation_level_externalities_for_onli/">Contextual Generative Auction with Permutation-level Externalities for Online Advertising</a></strong></dt>
<dd>
<p>用自回归生成器直接生成广告位序列并建模排列级外部性，配合可微事后遗憾学习支付规则，淘宝线上 RPM +3.2%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/effective_two_stage_knowledge_transfer_for_multi_entity_cross_domain_recomm/">Effective Two-Stage Knowledge Transfer for Multi-Entity Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>提出 MKT 两阶段多实体知识迁移框架，用异构特征对齐与极化分布损失解决异质特征 schema 下的负迁移，线上 CTR +4.13%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/lightweight_auto_bidding_based_on_traffic_prediction_in_live_advertising/">Lightweight Auto-bidding based on Traffic Prediction in Live Advertising</a></strong></dt>
<dd>
<p>直播广告场景下，用LightGBM预测全时段累计成本/点击，结合LP对偶最优出价公式与投影梯度下降求解对偶变量，实现秒级轻量自动出价，线上GMV+4.28%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/scaling_transformers_for_discriminative_recommendation_via_generative_pretr/">Scaling Transformers for Discriminative Recommendation via Generative Pretraining</a></strong></dt>
<dd>
<p>用生成式预训练初始化判别式排序模型并冻结稀疏Embedding，破解CTR/CVR过拟合，使稠密参数从13K扩到0.3B仍遵循幂律提升。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/swat_statistical_modeling_of_video_watch_time_through_user_behavior_analysi/">SWaT: Statistical Modeling of Video Watch Time through User Behavior Analysis</a></strong></dt>
<dd>
<p>将用户观看行为假设翻译为分桶二项/几何分布白盒统计模型，以软标签 BCE 训练，离线与线上 A/B 均稳定超越 WLR/TPM/CREAD。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/tayfcs_towards_light_feature_combination_selection_for_deep_recommender_sys/">TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems</a></strong></dt>
<dd>
<p>用泰勒展开+IME近似一次性梯度回传估计高阶组合重要性，再用LR滑窗剔除冗余，三数据集AUC显著提升且线上CVR+13.9%</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/canniuplift_a_holistic_framework_for_mitigating_seller_and_incentive_cannib/">CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling</a></strong></dt>
<dd>
<p>用平台级GMV一致性约束与核销路径分解，缓解多卖家电商uplift建模中的卖家级与激励级蚕食，线上ΔGMV相对提升4.08%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/ctr_sink_attention_sink_for_language_models_in_click_through_rate_predictio/">CTR-Sink: Attention Sink for Language Models in Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>在文本化行为序列中插入携带时间距离的 [SINK] 锚点并配合两阶段训练与 sink 间注意力增强，缓解语义碎片化，AUC 稳定提升 0.2–0.6%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/expand_more_shrink_less_shaping_effective_rank_dynamics_for_dense_scaling_i/">Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation</a></strong></dt>
<dd>
<p>从有效秩视角诊断 RankMixer 的阻尼振荡塌缩，提出参数化全混合与 GLU-PFFN 实现“多扩张少收缩”，Criteo/Avazu 上 AUC 提升超 0.001。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/flowtime_towards_continuous_generative_watch_time_prediction_via_flow_based/">FlowTime: Towards Continuous Generative Watch Time Prediction via Flow-based Personalized Priors</a></strong></dt>
<dd>
<p>提出连续生成式回归新范式，用一步VAE+归一化流个性化先验建模多峰观看时长分布，KuaiRec MAE降0.676%、线上播放时长+1.044%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/from_scaling_to_structured_expressivity_rethinking_transformers_for_ctr_pre/">From Scaling to Structured Expressivity: Rethinking Transformers for CTR Prediction</a></strong></dt>
<dd>
<p>用域感知参数重构 Transformer 块，将复杂度从词表规模 n 转向域数 F，配合基组合超网络实现可预测幂律缩放，离线 AUC 最高 +4.38%，线上 CTR +2.33%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/hmaf_a_hierarchical_multi_slot_gd_rtb_allocation_framework/">HMAF: A Hierarchical Multi-Slot GD-RTB Allocation Framework</a></strong></dt>
<dd>
<p>以 Plan-Calibrate-Execute 范式将 PV 约束对偶优化与在线列表级生成-评估排序耦合，美团线上 GD 交付率 +3.72%、广告总收入 +1.59%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/mac_a_conversion_rate_prediction_benchmark_featuring_labels_under_multiple_/">MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms</a></strong></dt>
<dd>
<p>首个多归因机制CVR公开基准MAC与PyMAL库，提出MoAE模型，GAUC最高提升0.39pt。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/mixformer_co_scaling_up_dense_and_sequence_in_industrial_recommenders/">MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders</a></strong></dt>
<dd>
<p>用单一参数空间统一序列建模与稠密特征交叉，配合用户-物品解耦实现请求级复用，AUC 提升 1.28% 且 FLOPs 降 36%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/onerank_unified_transformer_native_ranking_architecture_for_multi_task_reco/">OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation</a></strong></dt>
<dd>
<p>将多任务推理内化进 Transformer 栈，用任务私有 token 通道 + 梯度截断 + 匹配式打分替代编码器-预测器分离，Shopee 离线与线上 A/B 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/blending_sequential_embeddings_graphs_and_engineered_features_4th_place_sol/">Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025</a></strong></dt>
<dd>
<p>异构因果Transformer序列编码+TwHIN图+300维手工特征经DCN-v2融合，构建跨6任务通用用户画像，获挑战赛第4名。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/counterfactual_inference_under_thompson_sampling/">Counterfactual Inference under Thompson Sampling</a></strong></dt>
<dd>
<p>推导 Thompson Sampling 下动作倾向性的精确可计算闭式解，使 IPS 类离线评估在 TS 策略下首次可行，OBP 仿真验证无偏收敛。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/decoupled_entity_representation_learning_for_pinterest_ads_ranking/">Decoupled Entity Representation Learning for Pinterest Ads Ranking</a></strong></dt>
<dd>
<p>提出解耦式上游-下游实体表征框架 DERM，三塔+对比自监督+滑动平均聚合，离线 AUC 提升 0.19%、线上 CTR +1.38%、CPA -1.61%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/exploring_scaling_laws_of_ctr_model_for_online_performance_improvement/">Exploring Scaling Laws of CTR Model for Online Performance Improvement</a></strong></dt>
<dd>
<p>提出 SUAN 统一注意力块实现 CTR 模型三级量级缩放律，再经稀疏注意力+并行推理+在线蒸馏落地线上，CTR +2.81%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/identifying_offline_metrics_that_predict_online_impact_a_pragmatic_strategy/">Identifying Offline Metrics that Predict Online Impact: A Pragmatic Strategy for Real-World Recommender Systems</a></strong></dt>
<dd>
<p>用帕累托前沿近似单模型多偏好分流，在 OTTO 线上实验验证 Recall@20→CTR、OD@20→CVR、乘积→销量 的离线-在线对齐关系。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/industry_insights_from_comparing_deep_learning_and_gbdt_models_for_e_commer/">Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank</a></strong></dt>
<dd>
<p>在 OTTO 电商 LTR 场景系统对比 DNN 与生产级 LambdaMART，Two-Tower+Softmax CE 离线 NDCG_c +4.32%，8 周线上点击 +1.86%、营收 +0.56%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/modeling_long_term_user_behaviors_with_diffusion_driven_multi_interest_netw/">Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction</a></strong></dt>
<dd>
<p>用目标正交分解构造兴趣通道做多兴趣抽取，再用条件扩散生成增强兴趣，工业 A/B 上 CTR +1.52%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/practical_multi_task_learning_for_rare_conversions_in_ad_tech/">Practical Multi-Task Learning for Rare Conversions in Ad Tech</a></strong></dt>
<dd>
<p>以历史CVR统计自动划分软/硬转化任务，共享底层+双塔MTL提升稀有转化预估，离线AUC+0.69%、线上CPA降2%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/r4ec_a_reasoning_reflection_and_refinement_framework_for_recommendation_sys/">R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems</a></strong></dt>
<dd>
<p>用 actor/reflection 双小模型迭代反思精炼 LLM 知识，注入 CTR 主干，AUC 相对提升 2.5%+，线上收入 +2.2%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/ctr_feature_interaction/building_a_user_foundation_model_for_the_open_web/">Building a User Foundation Model for the Open Web</a></strong></dt>
<dd>
<p>在身份碎片化的开放网络 RTB 场景下，用 MLM+时序切分对比预训练 Transformer 用户序列编码器，并以 LLM-in-the-loop NAS 优化训练管线，线上 A/B 取得 +2.13% CTR、-1.13% eCPC。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/ctr_feature_interaction/hierarchical_gtv_estimation_bridging_the_gap_between_container_level_rankin/">Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion</a></strong></dt>
<dd>
<p>面向容器式电商的层级 GTV 预估框架 HGE，用集合感知编码器 + 代理标签分解 + 残差融合解决粒度错配，离线 XAUC +0.63pp、线上 RPS +1.42%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/ctr_feature_interaction/lo_far_a_cost_aware_local_filter_for_sparse_feature_ranking_in_industrial_a/">LO-FAR: A Cost-Aware Local Filter for Sparse Feature Ranking in Industrial Ad Recommendation</a></strong></dt>
<dd>
<p>用CPU级逐特征独立局部估计器替代GPU重训式置换/门控重要性，在475个稀疏ID-list特征上以约2 CPU小时完成排序并保持下游NE增益。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/longer_scaling_up_long_sequence_modeling_in_industrial_recommenders/">LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders</a></strong></dt>
<dd>
<p>用全局Token+Token Merge+混合因果注意力把工业超长行为序列端到端建模推到1万长度，离线AUC+1.57%、线上广告与电商双场景显著提升。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/multi_objective_ranking_for_live_streaming_balancing_fresh_and_delayed_sign/">Multi-Objective Ranking for Live-Streaming: Balancing Fresh and Delayed Signals with Segment-Aware Targeting</a></strong></dt>
<dd>
<p>Twitch 直播精排用延迟窗口+新鲜/延迟双模型分离+分段推理加权+MMoE，DAV +0.09%、D 段 ARPU +0.56%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/personalizing_incremental_video_search_with_hybrid_text_and_id_embeddings/">Personalizing Incremental Video Search with Hybrid Text and ID Embeddings</a></strong></dt>
<dd>
<p>面向 Apple TV 逐键增量搜索，用文本语义与 ID 协同双嵌入注入 XGBoost 排序器，短前缀 NDCG@10 提升 8.63%，线上 CTR +1.14%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/pricorec_a_privacy_aware_cloud_device_collaborative_framework_for_ad_recomm/">PriCoRec: A Privacy-Aware Cloud-Device Collaborative Framework for Ad Recommendation under Feature Constraints</a></strong></dt>
<dd>
<p>将敏感特征留在端侧，云侧预排加 DPP 多样性正则、端侧轻量排序加云端 logit 蒸馏，三数据集 gAUC/R@10 全面领先且时延可控。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/unirec_bridging_the_expressive_gap_between_generative_and_discriminative_re/">UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute</a></strong></dt>
<dd>
<p>用贝叶斯等价性证明生成式推荐差距源于特征覆盖，以属性链 CoA 前缀解码 + 容量约束 SID + 条件解码上下文补齐，Shopee 线上 GMV +5.60%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/a_learnable_fully_interacted_two_tower_model_for_pre_ranking_system/">A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System</a></strong></dt>
<dd>
<p>用可学习 Item Meta Matrix 实现显式早期交互、行/列 FC 相似度矩阵实现晚期交互，在保持双塔解耦推理效率下 AUC 相对提升 4.6%–14.3%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/llm_rec/adore_autonomous_domain_oriented_relevance_engine_for_e_commerce/">ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce</a></strong></dt>
<dd>
<p>用 CoT-LLM 自动挖掘难样本、KTO 对齐用户行为、错误类型感知合成对抗样本，再以关键属性显式蒸馏到轻量 BERT，离线 F1 85.45%、线上 CTR +1.24%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/deep_multiple_quantization_network_on_long_behavior_sequence_for_click_thro/">Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>用多码本量化将超长行为序列端到端压缩为约百个兴趣簇码字，配合 HSTU 簇间交互与候选感知注意力，消除两阶段检索的相关性分布偏差，线上 CTR +3.5%、RPM +2.0%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/diffusion_based_multi_modal_synergy_interest_network_for_click_through_rate/">Diffusion-based Multi-modal Synergy Interest Network for Click-through Rate Prediction</a></strong></dt>
<dd>
<p>用 PLE 式共享/独有专家解耦模态知识、以扩散式多步跨模态去噪捕捉协同信号，FDAF 非侵入融合，四数据集 AUC 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/dlf_enhancing_explicit_implicit_interaction_via_dynamic_low_order_aware_fus/">DLF: Enhancing Explicit-Implicit Interaction via Dynamic Low-Order-Aware Fusion for CTR Prediction</a></strong></dt>
<dd>
<p>以残差感知低阶交互网络与逐层注意力融合，动态平衡显隐式特征交叉，四数据集AUC全面SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/elec_efficient_large_language_model_empowered_click_through_rate_prediction/">ELEC: Efficient Large Language Model-Empowered Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>伪孪生增益/朴素双网络+分数级CLID与表示级MSE双蒸馏，在线仅跑表格DCNv2，AUC较FLIP提升0.0020/0.0034。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/hybrid_advertising_in_the_sponsored_search/">Hybrid Advertising in the Sponsored Search</a></strong></dt>
<dd>
<p>提出混合广告拍卖模型与 HRegNet 神经网络机制，在近 DSIC/IR 约束下联合优化独立店铺与店铺-品牌捆绑包的广告位分配与定价，真实数据上收益显著超越 VCG 与 JRegNet。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/interest_changes_considering_user_interest_life_cycle_in_recommendation_sys/">Interest Changes: Considering User Interest Life Cycle in Recommendation System</a></strong></dt>
<dd>
<p>用曝光/点击/交互直方图+VQ聚类刻画兴趣生命周期，经特征重校准与专家网络融合注入MMOE精排，Lofter线上CTR+0.38%、CVR+1.04%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/lrea_low_rank_efficient_attention_on_modeling_long_term_user_behaviors_for_/">LREA: Low-Rank Efficient Attention on Modeling Long-Term User Behaviors for CTR Prediction</a></strong></dt>
<dd>
<p>用低秩分解+矩阵吸收把万级长行为序列压到秩 r 维做目标注意力，配非负正则保非线性，线上仅增 3ms 延迟、CTR +5.88%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/multi_objective_aligned_bidword_generation_model_for_e_commerce_search_adve/">Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising</a></strong></dt>
<dd>
<p>用判别器提供相关性/真实性/CPM三维奖励，扩展MODPO做多目标偏好对齐，驱动LLM生成高商业价值竞价词，线上广告收入+2.13%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/nam_a_normalization_attention_model_for_personalized_product_search_in_flig/">NAM: A Normalization Attention Model for Personalized Product Search In Fliggy</a></strong></dt>
<dd>
<p>用逆物品频率(IIF)对注意力做全局归一化并门控个性化强度，解决热门物品高估/长尾低估，离线CTCVR AUC+0.001、线上CTCVR+0.8%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/sequential_rec/pyramid_mixer_multi_dimensional_multi_period_interest_modeling_for_sequenti/">Pyramid Mixer: Multi-dimensional Multi-period Interest Modeling for Sequential Recommendation</a></strong></dt>
<dd>
<p>用纯 MLP 金字塔 Mixer 做跨行为/跨特征/跨周期三维兴趣建模，离线全库排序超越 SASRec/MLP4Rec，抖音线上停留时长 +0.106%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/sequential_rec/star_rec_making_peace_with_length_variance_and_pattern_diversity_in_sequent/">STAR-Rec: Making Peace with Length Variance and Pattern Diversity in Sequential Recommendation</a></strong></dt>
<dd>
<p>用偏好感知注意力与 SSM 双路径自适应融合 + 序列级 MoE，统一建模长度方差与行为模式多样性，Leave-One-Out 全库排序下 Recall@10 最高提升 2.15%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>揭示稠密MLP隐式权重稀疏现象，提出 filter-then-fuse 多视图显式稀疏框架 SSR，在十亿级工业数据与三个公开集上以更低算力超越 RankMixer。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/cheaper_is_better_a_discount_aware_network_for_conversion_rate_prediction_i/">Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System</a></strong></dt>
<dd>
<p>用 FFT 时频分解 + 用户/上下文双重去偏 + 折扣回归辅助任务建模商品折扣率时序，离线 AUC +1.61%、线上 pCVR +3.63%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong></dt>
<dd>
<p>提出可学习多序列构造与线性注意力，缓解单序列上下文污染，Meta 四场景 NE 降 0.3%–0.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/counterfactual_multi_task_learning_for_delayed_conversion_modeling_in_e_com/">Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion</a></strong></dt>
<dd>
<p>面向大促预热期“集中式延迟转化”的因果多任务 CVR 模型，联合直接/延迟转化建模、个性化门控迁移与 ATC 反事实正则，离线 AUC 显著超越延迟反馈基线并落地线上 A/B。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/exploring_test_time_scaling_via_prediction_merging_on_large_scale_recommend/">Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出预测合并式测试时缩放，用异构架构与同构随机种子生成多样 logit 并平均融合，三数据集八骨干上稳定超越参数缩放。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>以目标条件化频谱熵观察为动机，用目标感知频谱滤波+时域双分支做CTR预估，三数据集GAUC/AUC全面SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/follow_the_trace_exploiting_post_click_trajectories_for_online_delayed_conv/">Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将点击后行为演化建模为反馈轨迹，用贝叶斯分解+可靠性门控回溯补全动态修正未转化样本后验，Criteo AUC 0.8382 超越 DEFUSE。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/generative_bid_shading_in_real_time_bidding_advertising/">Generative Bid Shading in Real-Time Bidding Advertising</a></strong></dt>
<dd>
<p>用自回归残差生成+CHNet奖励模型+GRPO偏好对齐替代两阶段出价调价，离线SR达41.74%、线上ROI+3.4%且推理耗时降37.6%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/hyformer_revisiting_the_roles_of_sequence_modeling_and_feature_interaction_/">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></strong></dt>
<dd>
<p>用 Global Tokens 交替执行 Query Decoding 与 Query Boosting，统一长序列建模与特征交叉，AUC 提升 0.17% 且 FLOPs 仅 3.9T。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱食门控+TTNP自监督学习商品复购周期，PPRR降超60%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/modular_representation_compression_adapting_llms_for_efficient_and_effectiv/">Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 中间层表征优势 (MRA) 并用模块化解耦压缩修复，128 维表征在 CTR/重排/召回全面超越基线，线上 eCPM +2.82%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/next_interest_flow_a_generative_pre_training_paradigm_for_recommender_syste/">Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines</a></strong></dt>
<dd>
<p>将用户意图建模为隐流形上的连续演化轨迹，用切空间正交与测地平滑约束生成式预训练，并以双向对齐+TSP桥接判别式精排，工业级AUC+0.87pt。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/r_f_inventory_a_large_scale_dataset_for_monotonic_inventory_estimation_in_r/">R&amp;F-Inventory: A Large-Scale Dataset for Monotonic Inventory Estimation in Reach and Frequency Advertising</a></strong></dt>
<dd>
<p>首个公开的 R&amp;F 合约库存预估数据集，以「定向-排期-频控」为上下文提供完整预算-效果曲线，定义单点预测与多维单调曲线重建两大基准任务。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/solaris_speculative_offloading_of_latent_based_representation_for_inference/">SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling</a></strong></dt>
<dd>
<p>借鉴投机解码思想，异步预计算基础模型用户-物品隐表征并分层补全覆盖，实现推理时知识蒸馏，Meta 广告线上收入 +0.67%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/understanding_dnns_in_feature_interaction_models_a_dimensional_collapse_per/">Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective</a></strong></dt>
<dd>
<p>从嵌入维度坍缩视角重新解释 DNN 在特征交叉模型中的作用：并行/堆叠 DNN 均能平坦化奇异值谱、提升 RankMe，并给出基于梯度的机理分析。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR-AP/2025/debias_fairness_industrial/synthetic_prefixes_to_mitigate_bias_in_real_time_neural_query_autocomplete/">Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomplete</a></strong></dt>
<dd>
<p>用无 QAC 会话的完整查询合成前缀训练数据，缓解展示偏差，并以 O(n) 成对近似实现列表级损失，线上 MRR 提升 0.92%–1.49%。</p>
</dd>
<dt><strong><a href="../../venues/TKDD/2025/debias_fairness_industrial/exploring_and_exploiting_data_heterogeneity_in_recommendation/">Exploring and Exploiting Data Heterogeneity in Recommendation</a></strong></dt>
<dd>
<p>提出双层聚类 BHE 分层挖掘推荐数据预测机制与协变量分布异质性，用于多子模型泛化与 IPS 去偏，Yelp/MovieLens 上 NDCG 提升 20%+。</p>
</dd>
<dt><strong><a href="../../venues/TKDD/2025/ctr_feature_interaction/jobformer_skill_aware_job_recommendation_with_semantic_enhanced_transformer/">JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer</a></strong></dt>
<dd>
<p>用局部-全局注意力 Transformer 聚合邻居 JD 补全语义，并以技能分布引导两阶段召回+CTR排序，Recall@20 较 NRMS 提升 2.31%。</p>
</dd>
<dt><strong><a href="../../venues/TKDD/2026/ctr_feature_interaction/dual_stream_mlp_is_all_you_need_for_ctr_prediction/">Dual-Stream MLP is All You Need for CTR Prediction</a></strong></dt>
<dd>
<p>用知识蒸馏把复杂显式交叉教师压进纯MLP双流学生，配合隐状态BN与预测对齐，三基准AUC全面SOTA且推理低延迟。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/ctr_feature_interaction/aefs_adaptive_early_feature_selection_for_deep_recommender_systems/">AEFS: Adaptive Early Feature Selection for Deep Recommender Systems</a></strong></dt>
<dd>
<p>用 mini-AdaFS 辅助模型在嵌入层前做实例级自适应特征选择，配合双对齐损失，精度持平 AdaFS 且激活嵌入参数降 37.5%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/graph_rec/graph_cross_correlated_network_for_recommendation/">Graph Cross-Correlated Network for Recommendation</a></strong></dt>
<dd>
<p>用非递归 PGR 保留各跳邻居纯语义，再以 HCC/ECC 显式建模用户-物品子图交叉相关项，全库排序下 NDCG@20 平均提升约 1.8%–2.9%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/llm_rec/preference_consistent_knowledge_distillation_for_recommender_system/">Preference-Consistent Knowledge Distillation for Recommender System</a></strong></dt>
<dd>
<p>揭示特征蒸馏中投影器导致的偏好不一致现象，提出成对/列表两种正则项约束投影器，全库排序下 NDCG@20 最高提升 20%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/ctr_feature_interaction/uncertainty_calibration_for_counterfactual_propensity_estimation_in_recomme/">Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation</a></strong></dt>
<dd>
<p>揭示倾向分估计严重失准问题，提出以ECE为质量度量的模型无关校准框架，理论证明校准倾向分降低IPS偏差，在Coat/Yahoo/KuaiRand上显著提升CVR去偏效果。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/sequential_rec/an_adaptive_context_aware_and_stacked_attention_network_based_recommendatio/">An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users&#x27; Temporal Preference</a></strong></dt>
<dd>
<p>用上下文物品注意力+堆叠多头用户注意力捕捉用户时变偏好，Leave-One-Out+100负采样排序，三数据集HR/NDCG/MRR@10全面超越SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/ctr_feature_interaction/a_unified_framework_for_multi_domain_ctr_prediction_via_large_language_mode/">A Unified Framework for Multi-Domain CTR Prediction via Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 分层语义表征作跨域共享底座、可插拔 DSN 建模域特性，掩码损失解耦梯度，AUC 相对提升 24%–63% 并支持零样本新域。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/multi_task_reranking/efficient_multi_task_prompt_tuning_for_recommendation/">Efficient Multi-task Prompt Tuning for Recommendation</a></strong></dt>
<dd>
<p>提出两阶段 MPT-Rec：GAN 解耦任务共享/特有信息后冻结主干，仅用任务感知 Prompt 微调新任务，参数降至 10% 且性能保持 95%+。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/impression_aware_recommender_systems/">Impression-Aware Recommender Systems</a></strong></dt>
<dd>
<p>首篇曝光感知推荐系统(IARS)系统性综述，提出统一理论框架与三维分类体系，厘清模型、数据集与评估协议。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/ctr_feature_interaction/towards_automated_model_design_on_recommender_systems/">Towards Automated Model Design on Recommender Systems</a></strong></dt>
<dd>
<p>用权重共享超网 NAS 在稠密连接+异构算子+维度+量化空间上自动搜索 CTR 模型，Criteo/KDD 降 0.001~0.003 LogLoss，FLOPs 降 8.5×。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/ctr_feature_interaction/explainable_ctr_prediction_via_llm_reasoning/">Explainable CTR Prediction via LLM Reasoning</a></strong></dt>
<dd>
<p>用 PPO 双奖励（LC 用户意图对齐 + IC 与 ID-CTR 模型对齐）把 LLM 解释生成嵌入 CTR 流程，免解释数据集，AUC 最高提升 18.2%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/ctr_feature_interaction/fusion_matters_learning_fusion_in_deep_click_through_rate_prediction_models/">Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models</a></strong></dt>
<dd>
<p>将CTR模型的融合连接与融合算子统一建模为可微搜索空间，用one-shot联合选择替代手工stacked/parallel，三数据集AUC稳定超越EDCN与NASRec。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/generative_rec/unlocking_scaling_law_in_industrial_recommendation_systems_with_a_three_ste/">Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model</a></strong></dt>
<dd>
<p>提出 LUM 三步范式（生成预训练→条件查询→判别式 DLRM 融合），以 next-condition-item 预测打通生成与判别，7B 参数下稳定扩展，线上 CTR +2.9%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/sequential_rec/a_contextual_aware_position_encoding_for_sequential_recommendation/">A Contextual-Aware Position Encoding for Sequential Recommendation</a></strong></dt>
<dd>
<p>首个为序列推荐定制的上下文感知位置编码 CAPE：用目标-上下文不相似度累加动态分配位置，配合门控融合异构特征，在 5 个骨干与工业 A/B 上稳定涨点。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/bat_benchmark_for_auto_bidding_task/">BAT: Benchmark for Auto-bidding Task</a></strong></dt>
<dd>
<p>发布 Avito 双拍卖格式（VCG/FP）真实自动出价数据集，配套 ALM/TA-PID 等 5 个基线，覆盖预算平滑与 CPC 约束两类 RTB 任务。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/ctr_driven_advertising_image_generation_with_multimodal_large_language_mode/">CTR-Driven Advertising Image Generation with Multimodal Large Language Models</a></strong></dt>
<dd>
<p>用 MLLM 双分支奖励模型把 CTR 相对比较转化为偏好信号，结合产品中心偏好优化 PCPO 驱动广告图生成，线上 CTR 提升 2%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/d2k_turning_historical_data_into_retrievable_knowledge_for_recommender_syst/">D2K: Turning Historical Data into Retrievable Knowledge for Recommender Systems</a></strong></dt>
<dd>
<p>将历史日志转为用户-物品-上下文三元知识库，Transformer 编码 + 个性化适配注入任意 CTR 模型，AUC 相对提升 1.35%–2.90%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/efficient_and_practical_approximation_algorithms_for_advertising_in_content/">Efficient and Practical Approximation Algorithms for Advertising in Content Feeds</a></strong></dt>
<dd>
<p>针对信息流原生广告位分配，提出反向贪心 2-近似算法，突破前人 4-近似瓶颈，并在合成与真实数据上首次系统评测。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/hcmrm_a_high_consistency_multimodal_relevance_model_for_search_ads/">HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads</a></strong></dt>
<dd>
<p>用视频关键词切分合成伪查询做三元组预训练，并以对称层次 Softmax 对齐四档相关性标签，快手搜索广告线上无关广告率降 6.1%、收入增 1.4%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/interactrank_personalized_web_scale_search_pre_ranking_with_cross_interacti/">InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features</a></strong></dt>
<dd>
<p>在双塔预排序点积之上叠加历史交互统计特征 IQP 与实时用户序列，以 1.1× FLOPs 实现线上 SIFR +6.5%、F1S +9.2%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/sequential_rec/long_term_interest_clock_fine_grained_time_perception_in_streaming_recommen/">Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System</a></strong></dt>
<dd>
<p>用候选物品+当前时刻双查询从长期行为中检索子序列，并以时距感知注意力聚合，实现流式推荐细粒度时间感知，线上活跃天数+0.122%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/middleman_bias_in_advertising_aligning_relevance_of_keyphrase_recommendatio/">Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search</a></strong></dt>
<dd>
<p>揭示广告关键词相关性过滤中的“中间人偏差”，用Search相关性判定重构训练标签，并以轻量Cross-Encoder在eBay线上取得CVR+11.6%的收益。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/on_the_practice_of_deep_hierarchical_ensemble_network_for_ad_conversion_rat/">On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction</a></strong></dt>
<dd>
<p>Pinterest 工业实践：两层 DHEN 多任务骨干 + 五类行为序列端到端建模 + 未来行为自监督辅助损失，线上 CPA 降 2.15%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/on_device_content_based_recommendation_with_single_shot_embedding_pruning_a/">On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspective</a></strong></dt>
<dd>
<p>用 Shapley 值一次性剪枝 CTR 嵌入表，配合场感知码本替代零填充，无需重训即可适配任意端侧存储预算。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multi_task_reranking/pcl_prompt_based_continual_learning_for_user_modeling_in_recommender_system/">PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems</a></strong></dt>
<dd>
<p>用位置级+上下文提示做持续用户建模，冻结骨干仅训提示，缓解灾难性遗忘，6/8 任务超越 MTL 上界。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/pre_train_and_fine_tune_recommenders_as_large_models/">Pre-train and Fine-tune: Recommenders as Large Models</a></strong></dt>
<dd>
<p>以信息瓶颈理论解释推荐模型微调，提出即插即用的信息感知自适应核 IAK，冻结十亿参数预训练推荐大模型，仅用一周域内数据微调即获 CTR-AUC 平均 +0.26%~+1.28% 与线上 GMV +0.74%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/prectr_a_synergistic_framework_for_integrating_personalized_search_relevanc/">PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction</a></strong></dt>
<dd>
<p>用条件概率融合把相关性分级与CTR统一进单模型，配两阶段训练、语义一致性正则与个性化相关性激励，线上CTR+0.4%、GMV+1.1%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/rtbagent_a_llm_based_agent_system_for_real_time_bidding/">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></strong></dt>
<dd>
<p>首个 LLM Agent 出价系统，用两步决策+多类记忆+专家策略工具在 iPinYou 上以调整因子方式超越规则/RL/扩散基线。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/delayed_feedback_modeling_for_post_click_gross_merchandise_volume_predictio/">Delayed Feedback Modeling for Post-Click Gross Merchandise Volume Prediction: Benchmark, Insights and Approaches</a></strong></dt>
<dd>
<p>首个后点击GMV延迟反馈基准TRACE，提出复购感知双分支READER，在线流式训练下ACC相对提升2.19%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/self_supervised_denoising/efficient_content_based_recommendation_model_training_via_noise_aware_cores/">Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection</a></strong></dt>
<dd>
<p>用末层梯度子模优化选核集，配合渐进标签自校正与 MC-Dropout 去噪，1% 数据恢复 93-95% 全量性能。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/fairfs_addressing_deep_feature_selection_biases_for_recommender_system/">FairFS: Addressing Deep Feature Selection Biases for Recommender System</a></strong></dt>
<dd>
<p>识别深度特征选择的三类偏差（层偏差/基线偏差/近似偏差），以平滑基线与聚合近似实现无偏特征重要性估计，AUC 提升最高 0.35%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/from_reasoning_llms_to_bert_a_two_stage_distillation_framework_for_search_r/">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></strong></dt>
<dd>
<p>美团搜索广告两阶段框架：领域自适应推理LLM做教师标注，CRSD自蒸馏将推理能力内化进6层BERT，线上AdCTR+0.91%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/genci_generative_modeling_of_user_interest_shift_via_cohort_based_intent_le/">GenCI: Generative Modeling of User Interest Shift via Cohort-based Intent Learning for CTR Prediction</a></strong></dt>
<dd>
<p>用RQ-VAE语义ID+NTP生成兴趣群组，再以分层候选感知注意力注入精排，弥合召回-排序意图鸿沟，三数据集AUC全面SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/mixture_of_sequence_theme_aware_mixture_of_experts_for_long_sequence_recomm/">Mixture of Sequence: Theme-Aware Mixture-of-Experts for Long-Sequence Recommendation</a></strong></dt>
<dd>
<p>揭示长序列“会话跳跃”现象，用主题码本路由+多尺度专家融合做子序列抽取，AUC 平均提升 0.68% 且 FLOPs 更低</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/modeling_cascaded_delay_feedback_for_online_net_conversion_rate_prediction_/">Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Insights and Solutions</a></strong></dt>
<dd>
<p>首个淘宝净转化率流式基准 CASCADE，提出 CVR-RFR 级联+分阶段去偏+延迟感知排序损失的 TESLA，NetCVR 的 RI-AUC 绝对提升 12.41%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multi_task_reranking/not_all_candidates_are_created_equal_a_heterogeneity_aware_approach_to_pre_/">Not All Candidates are Created Equal: A Heterogeneity-Aware Approach to Pre-ranking in Recommender Systems</a></strong></dt>
<dd>
<p>针对粗排异构候选的梯度冲突与算力错配，提出 GHCL 分组对比损失与难度感知双模型路由，AUC 全面超越 SOTA 且线上降本 6%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/onetrans_unified_feature_interaction_and_sequence_modeling_with_one_transfo/">OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</a></strong></dt>
<dd>
<p>用单一因果 Transformer 统一序列建模与特征交叉，配合金字塔剪枝与跨请求 KV 缓存，线上 GMV/u 提升 5.68%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/unleashing_the_potential_of_sparse_attention_on_long_term_behaviors_for_ctr/">Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction</a></strong></dt>
<dd>
<p>面向长行为序列 CTR 的个性化时间分块 + 三分支稀疏注意力 SparseCTR，AUC 相对提升 8.49%，线上 CTR +1.72%。</p>
</dd>
</dl>

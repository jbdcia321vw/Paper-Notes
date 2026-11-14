# ⚖️ 多任务学习与列表级重排 (Multi-Task Learning & Listwise Re-ranking)

> **研究范畴**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

当前方向共收录 **116** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/multi_task_reranking/direct_routing_gradient_drgrad_a_personalized_information_surgery_for_multi/">Direct Routing Gradient (DRGrad): A Personalized Information Surgery for Multi-Task Learning (MTL) Recommendations</a></strong></dt>
<dd>
<p>用梯度余弦相似度做路由+Split-MMoE+PPNet个性化门控，在150亿样本工业数据上Click AUC提升0.0027且不增推理延迟。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/sequential_rec/driving_engagement_in_daily_fantasy_sports_with_a_scalable_and_urgency_awar/">Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine</a></strong></dt>
<dd>
<p>在 DIN 上注入实时紧迫度特征与时间位置编码，配合 neuralNDCG 列表损失，在 6.5 亿用户级工业数据上 nDCG@1 相对 LightGBM 提升 9%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/generative_product_recommendations_for_implicit_superlative_queries/">Generative Product Recommendations for Implicit Superlative Queries</a></strong></dt>
<dd>
<p>提出 SUPERB 四级标注体系与深思式属性生成，用 LLM 列表级重排处理隐式最高级商品查询，nDCG@10 相对 BM25 提升约 18%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/multimodal_rec/premise_matching_based_prediction_for_accurate_review_recommendation/">PREMISE: Matching-based Prediction for Accurate Review Recommendation</a></strong></dt>
<dd>
<p>以多尺度多字段语义匹配分数替代跨模态融合表征，用于多模态评论有用性排序，MAP 较 GBDT 提升 5–17 点且训练提速 42%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/similarity_value_consultation_value_assessment_and_alignment_for_personaliz/">Similarity = Value? Consultation Value Assessment and Alignment for Personalized Search</a></strong></dt>
<dd>
<p>指出语义相似度无法代理咨询价值，提出场景范围/后验行为/时间衰减三维价值评估与 CAI 对齐的 VAPS，检索排序双任务全面领先。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/content_agnostic_moderation_for_stance_neutral_recommendation/">Content-Agnostic Moderation for Stance-Neutral Recommendation</a></strong></dt>
<dd>
<p>提出不依赖内容特征的立场中立化后处理审核，用双聚类离散化打散用户-物品共簇，仿真闭环中显著降低立场偏度且 CTR 仅降约 11%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multi_task_reranking/gref_a_unified_generative_framework_for_efficient_reranking_via_ordered_mul/">GReF: A Unified Generative Framework for Efficient Reranking via Ordered Multi-token Prediction</a></strong></dt>
<dd>
<p>用双向编码器+动态匹配自回归解码器统一生成式重排，Rerank-DPO 端到端对齐用户偏好，OMTP 多token并行解码将延迟压至 12.97ms，快手线上转发+2.98%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/hit_model_a_hierarchical_interaction_enhanced_two_tower_model_for_pre_ranki/">HIT Model: A Hierarchical Interaction-Enhanced Two-Tower Model for Pre-Ranking Systems</a></strong></dt>
<dd>
<p>用双生成器预生成跨塔粗粒度向量+多头子空间细粒度匹配，在保持双塔预缓存效率下逼近全交互模型精度，腾讯线上GMV+1.66%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/leveraging_watch_time_feedback_for_short_video_recommendations_a_causal_lab/">Leveraging Watch-time Feedback for Short-Video Recommendations: A Causal Labeling Framework</a></strong></dt>
<dd>
<p>用分位数把观看时长转成 WPR 与二值标签，并在标注阶段按视频时长分组做因果去偏，离线全指标最优、线上观看时长 +2.23%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multi_task_reranking/mpformer_adaptive_framework_for_industrial_multi_task_personalized_sequenti/">MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever</a></strong></dt>
<dd>
<p>用目标条件化 Transformer 统一多目标召回，并以排序分数监督动态配额分配，在快手 4 亿 DAU 上实现 +0.43% 总时长与 66.7% 推理资源节省。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multi_task_reranking/orca_mitigating_over_reliance_for_multi_task_dwell_time_prediction_with_cau/">ORCA: Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling</a></strong></dt>
<dd>
<p>用因果解耦显式建模并减去CTR对停留时间预测的负迁移，恢复中时长分桶语义，DT指标平均提升10.6%且不损CTR。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/personalized_tree_based_progressive_regression_model_for_watch_time_predict/">Personalized Tree-Based Progressive Regression Model for Watch-Time Prediction in Short Video Recommendation</a></strong></dt>
<dd>
<p>将TPM固定满二叉树改为样本级可学习剪枝树，并用IPS校正条件建模选择偏差，离线MAE/XAUC全面领先，线上观看时长+0.428%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/rankmixer_scaling_up_ranking_models_in_industrial_recommenders/">RankMixer: Scaling Up Ranking Models in Industrial Recommenders</a></strong></dt>
<dd>
<p>用无参多头Token Mixing替代自注意力、Per-token FFN隔离特征子空间，配合ReLU路由稀疏MoE，在抖音万亿级数据上将MFU从4.5%提至45%，参数扩100倍而延迟不增。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/see_beyond_a_single_view_multi_attribution_learning_leads_to_better_convers/">See Beyond a Single View: Multi-Attribution Learning Leads to Better Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将多归因标签作为辅助任务联合学习，用笛卡尔积辅助训练增强监督信号，离线GAUC+0.51%、线上ROI+2.6%</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/rl_bandit_rec/stratified_expert_cloning_for_retention_aware_recommendation_at_scale/">Stratified Expert Cloning for Retention-Aware Recommendation at Scale</a></strong></dt>
<dd>
<p>用分层专家行为克隆替代 RL 探索，配合动作熵正则与自适应专家选择，在快手双平台 A/B 中提升活跃天数 0.098%/0.122%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multi_task_reranking/you_only_evaluate_once_a_tree_based_rerank_method_at_meituan/">You Only Evaluate Once: A Tree-based Rerank Method at Meituan</a></strong></dt>
<dd>
<p>用树状多尺度上下文提取+上下文缓存实现单阶段全排列重排，离线GAUC显著提升并落地美团外卖。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/bridge_behavior_guided_residual_integration_with_dual_frequency_graph_evide/">BRIDGE: Behavior-Guided Residual Integration with Dual-Frequency Graph Evidence</a></strong></dt>
<dd>
<p>揭示跨视图对齐存在有效区间与频谱共享/私有分裂，提出双频图证据+行为符号残差仅在 top-K 候选内校准，Baby/Sports/Electronics 全库排序 R@20 最高提升 7.3%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/rl_bandit_rec/efficient_offline_learning_of_ranking_policies_via_top_k_policy_decompositi/">Efficient Offline Learning of Ranking Policies via Top-$k$ Policy Decomposition</a></strong></dt>
<dd>
<p>将排序策略分解为 top-$k$ 策略梯度与 bottom 回归两段，在 CPC 条件下无偏且大幅降方差，小样本与大规模排序空间下显著优于 IPS/DR/RIPS/IIPS。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/personalized_task_dependency_graphs_for_mitigating_signal_erosion_in_multi_/">Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation</a></strong></dt>
<dd>
<p>用低秩个性化任务依赖图+硬因果掩码GCN替代刚性漏斗，缓解深层稀疏任务信号侵蚀，离线AUC最高+1.45%，线上CVR+1.2%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/recsoup_efficient_model_merging_approach_for_controllable_multi_objective_r/">RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation</a></strong></dt>
<dd>
<p>用剪枝+可学习门控做任务向量合并，实现推理时可控多目标推荐，HV 提升 8.5% 且延迟快 165–228 倍。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/debias_fairness_industrial/revisiting_n2dcg_an_empirically_grounded_reformulation_of_carousel_recommen/">Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation Evaluation</a></strong></dt>
<dd>
<p>修正 N2DCG 的理想排序类别约束与镜像 F 型折扣，提出类别感知归一化与行-页折扣，在眼动数据与 2 万次布局仿真中显著提升评估保真度。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/selfdr_self_distillation_from_reasoning_for_llm_based_recommendation/">SelfDR: Self-Distillation from Reasoning for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用下游推荐奖励训练同源 Reasoner 生成理由，构建推理增强教师，再自蒸馏出免推理的直接推荐学生，重排 HR@1 提升约 15%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/sequential_multimodal_evidence_optimization_for_product_media_ranking_in_e_/">Sequential Multimodal Evidence Optimization for Product Media Ranking in E-Commerce</a></strong></dt>
<dd>
<p>将商品媒体排序建模为证据累积序列决策，两阶段解耦效用学习与生存加权自回归策略，离线 DR-CVR +5.4%、MSC −15.3%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/sort_a_systematically_optimized_ranking_transformer_for_industrial_scale_re/">SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders</a></strong></dt>
<dd>
<p>以请求为中心组织样本、局部注意力+查询剪枝、生成式预训练与稀疏MoE重构工业精排Transformer，线上GMV+8.65%且延迟降62%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/task_blind_no_more_multi_task_information_flow_in_unified_ranking_backbones/">Task-Blind No MORE: Multi-Task Information Flow in Unified Ranking Backbones</a></strong></dt>
<dd>
<p>将多任务信息流下沉至统一排序主干，以共享/私有 Anchor Token 逐层共演化，离线七任务全胜、线上时长+3.0%、P99 延迟降约30%。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2025/llm_rec/contextualizing_spotify_s_audiobook_list_recommendations_with_descriptive_s/">Contextualizing Spotify&#x27;s Audiobook List Recommendations with Descriptive Shelves</a></strong></dt>
<dd>
<p>用 LLM 抽取十类书目描述符并生成多样化描述性货架标题，两轮线上 A/B 测试验证发现性与互动指标提升。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/debias_fairness_industrial/explain_then_forget_recommendation_unlearning_under_asymmetric_access_const/">Explain, Then Forget: Recommendation Unlearning under Asymmetric Access Constraint</a></strong></dt>
<dd>
<p>面向 MLaaS 非对称访问约束的免参数推荐遗忘框架，用因果解释定位最小支持集并做输出级分数干预，遗忘效果逼近全量重训且延迟降低 9–55 倍。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/cold_start_rec/eating_for_a_sustainable_planet_personalized_sustainable_diet_recommendatio/">Eating for a Sustainable Planet: Personalized Sustainable Diet Recommendation via Constraint-Aware Decision-Making Modeling</a></strong></dt>
<dd>
<p>将可持续饮食推荐重构为约束感知决策问题，用可学习用户约束阈值替代偏好建模，在自建 SusDiet 上兼顾偏好精度与四维可持续性。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multi_task_reranking/goal_conditioned_supervised_learning_for_multi_objective_recommendation/">Goal-Conditioned Supervised Learning for Multi-Objective Recommendation</a></strong></dt>
<dd>
<p>将 GCSL 的标量目标扩展为多维向量目标，用 Transformer 编码器 + 交叉熵直接做多目标下一物品预测，无需复杂架构与权重调参，在 RetailRocket/Challenge15 上购买 HR@10 相对提升约 4-17%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multi_task_reranking/goalrank_group_relative_optimization_for_a_large_ranking_model/">GoalRank: Group-Relative Optimization for a Large Ranking Model</a></strong></dt>
<dd>
<p>证明单生成器排序器近似误差严格优于多生成器-评估器，提出组相对优化训练大排序模型，离线 H@6 最高 +25.4%，线上全量部署。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/rl_bandit_rec/optimal_and_efficient_contextual_combinatorial_semi_bandits_with_general_fu/">Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation</a></strong></dt>
<dd>
<p>提出 SquareCB.Comb，用回归 oracle + 对数障碍边际优化实现通用函数逼近下 CCSB 的极小极大最优 regret，并在网页推荐与最短路任务上验证。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/an_audio_centric_multi_task_learning_framework_for_streaming_ads_targeting_/">An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify</a></strong></dt>
<dd>
<p>面向音频主导多模态广告场景的 CAMoE 框架，以模态感知任务分组、自适应损失掩码与 DCN 专家实现近帕累托最优 CTR 预估，线上音频 CTR +14.5%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/beyond_item_dissimilarities_diversifying_by_intent_in_recommender_systems/">Beyond Item Dissimilarities: Diversifying by Intent in Recommender Systems</a></strong></dt>
<dd>
<p>用实时意图概率驱动整页贪心重排，以贝叶斯后验更新替代物品相似度，YouTube 线上 DAU +0.05%、用户享受度 +0.09%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/bid2x_revealing_dynamics_of_bidding_environment_in_online_advertising_from_/">Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens</a></strong></dt>
<dd>
<p>将自动出价环境建模重构为跨场景基础模型的下一token预测，用变量/时间双注意力+零膨胀投影，8数据集MAE全面领先，线上GMV+4.65%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/breaking_the_top_k_barrier_advancing_top_k_ranking_metrics_optimization_in_/">Breaking the Top-$K$Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems</a></strong></dt>
<dd>
<p>用量化分位数把 Top-K 截断转成阈值比较，推导 NDCG@K 的紧致平滑上界 SL@K，四数据集三骨干平均提升 6.03%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/combinatorial_optimization_perspective_based_framework_for_multi_behavior_r/">Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation</a></strong></dt>
<dd>
<p>将多行为融合视为组合优化问题施加三阶段约束，并用分布拟合多专家网络抑制负迁移，三数据集 HR@10 最高提升 49.91%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/contextual_generative_auction_with_permutation_level_externalities_for_onli/">Contextual Generative Auction with Permutation-level Externalities for Online Advertising</a></strong></dt>
<dd>
<p>用自回归生成器直接生成广告位序列并建模排列级外部性，配合可微事后遗憾学习支付规则，淘宝线上 RPM +3.2%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/embed_progressive_implicit_preference_in_unified_space_for_deep_collaborati/">Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering</a></strong></dt>
<dd>
<p>将多类隐式反馈按稀疏度映射为有序类别，用嵌套编码+嵌套OLR把渐进偏好压入统一空间，实现单次NNS一阶段排序。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/mopi_hfrs_a_multi_objective_personalized_health_aware_food_recommendation_s/">MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation</a></strong></dt>
<dd>
<p>构建首个融合 NHANES 医疗数据的健康感知食物推荐基准，用符号图结构学习 + Pareto 多目标优化平衡偏好/个性化健康/营养多样性，并以知识注入 LLM 生成可解释理由。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/rl_bandit_rec/multi_task_offline_reinforcement_learning_for_online_advertising_in_recomme/">Multi-task Offline Reinforcement Learning for Online Advertising in Recommender Systems</a></strong></dt>
<dd>
<p>用因果状态编码器+因果注意力+多任务解码器把广告渠道推荐与预算分配统一进离线RL，KuaiRand/Criteo 上 F1 提升约 7.5%/16.2%，线上 CTR +0.08。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/swat_statistical_modeling_of_video_watch_time_through_user_behavior_analysi/">SWaT: Statistical Modeling of Video Watch Time through User Behavior Analysis</a></strong></dt>
<dd>
<p>将用户观看行为假设翻译为分桶二项/几何分布白盒统计模型，以软标签 BCE 训练，离线与线上 A/B 均稳定超越 WLR/TPM/CREAD。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/bradley_terry_rankings_for_recommender_systems_across_dataset_taxonomies/">Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies</a></strong></dt>
<dd>
<p>用 Bradley-Terry 概率排序替代指标均值聚合，在 14 算法×89 数据集上给出数据集感知的推荐算法排名与免运行预测。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/breaking_the_likelihood_trap_consistent_generative_recommendation_with_grap/">Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model</a></strong></dt>
<dd>
<p>用有向无环图解码器扩展非自回归生成空间，并以 Gumbel-Softmax 打通评估器梯度，破解似然陷阱，快手线上 Views +0.78%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/degre_dense_supervised_generative_reranking_for_recommendation/">DeGRe: Dense-supervised Generative Reranking for Recommendation</a></strong></dt>
<dd>
<p>离线束搜索挖掘前瞻序列+累积回归稠密监督蒸馏至轻量生成器，单次贪心解码逼近全局最优，线上GMV+3.75%</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/heterogeneous_multi_treatment_uplift_modeling_for_trade_off_optimization_in/">Heterogeneous Multi-treatment Uplift Modeling for Trade-off Optimization in Short-Video Recommendation</a></strong></dt>
<dd>
<p>用异构多处理增益建模+在线动态权重，个性化权衡短视频消费与商业化指标，快手全量部署</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/hmaf_a_hierarchical_multi_slot_gd_rtb_allocation_framework/">HMAF: A Hierarchical Multi-Slot GD-RTB Allocation Framework</a></strong></dt>
<dd>
<p>以 Plan-Calibrate-Execute 范式将 PV 约束对偶优化与在线列表级生成-评估排序耦合，美团线上 GD 交付率 +3.72%、广告总收入 +1.59%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/mac_a_conversion_rate_prediction_benchmark_featuring_labels_under_multiple_/">MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms</a></strong></dt>
<dd>
<p>首个多归因机制CVR公开基准MAC与PyMAL库，提出MoAE模型，GAUC最高提升0.39pt。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/meta_lattice_model_space_redesign_for_cost_effective_industry_scale_ads_rec/">Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations</a></strong></dt>
<dd>
<p>以模型空间重构统一碎片化广告组合，融合归因窗口混合、帕累托特征选择与推理期知识迁移，线上Topline +10%、容量省20%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/onerank_unified_transformer_native_ranking_architecture_for_multi_task_reco/">OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation</a></strong></dt>
<dd>
<p>将多任务推理内化进 Transformer 栈，用任务私有 token 通道 + 梯度截断 + 匹配式打分替代编码器-预测器分离，Shopee 离线与线上 A/B 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/personalizing_marketplace_policies_with_competing_objectives_and_constraine/">Personalizing Marketplace Policies with Competing Objectives and Constrained Experiments: Evidence from a Job Marketplace</a></strong></dt>
<dd>
<p>双边市场免费额度阈值个性化：集成CATE+混合排序+线性外推，护栏风险降10%+，线上验证外推精度r=0.85</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/debias_fairness_industrial/a_reproducibility_study_of_product_side_fairness_in_bundle_recommendation/">A Reproducibility Study of Product-side Fairness in Bundle Recommendation</a></strong></dt>
<dd>
<p>首个捆绑推荐商品侧公平性复现研究，揭示捆绑级与物品级曝光分布不一致、指标间结论分歧，并发现用户捆绑偏好越强曝光越公平。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/d_rdw_diversity_driven_random_walks_for_news_recommender_systems/">D-RDW: Diversity-Driven Random Walks for News Recommender Systems</a></strong></dt>
<dd>
<p>将规范目标分布(NTD)作为线性约束嵌入随机游走采样，实现可解释、低能耗的新闻多样化重排，能耗降低一个数量级。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/non_parametric_graph_convolution_for_re_ranking_in_recommendation_systems/">Non-parametric Graph Convolution for Re-ranking in Recommendation Systems</a></strong></dt>
<dd>
<p>将图卷积从训练期搬到测试期，用相似用户/物品构造增强对并加权聚合打分，重排平均提升 8.1% 而额外开销仅 0.5%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/paragon_parameter_generation_for_controllable_multi_task_recommendation/">Paragon: Parameter Generation for Controllable Multi-Task Recommendation</a></strong></dt>
<dd>
<p>用扩散模型按偏好权重条件生成适配器参数，免重训实现测试时可控多任务推荐，响应时间降低 94.6%+。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/practical_multi_task_learning_for_rare_conversions_in_ad_tech/">Practical Multi-Task Learning for Rare Conversions in Ad Tech</a></strong></dt>
<dd>
<p>以历史CVR统计自动划分软/硬转化任务，共享底层+双塔MTL提升稀有转化预估，离线AUC+0.69%、线上CPA降2%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/generative_rec/prompt_to_slate_diffusion_models_for_prompt_conditioned_slate_generation/">Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation</a></strong></dt>
<dd>
<p>用潜空间扩散 Transformer 从纯文本 Prompt 直接生成整组 Slate，离线 NDCG 提升 17%、线上重复曝光降 13.4%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/radar_recall_augmentation_through_deferred_asynchronous_retrieval/">RADAR: Recall Augmentation through Deferred Asynchronous Retrieval</a></strong></dt>
<dd>
<p>用离线异步全复杂度精排模型对 50 倍候选池预打分并缓存 Top-200，线上绕过召回与粗排直接注入精排，Recall@200 翻倍、线上 +0.8%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/decoupled_learning_and_selection_in_slate_recommendation_for_privacy_and_st/">Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores</a></strong></dt>
<dd>
<p>将列表推荐拆为随机打分器+确定性选择器，给出隐私后处理契约与可审计的 margin 稳定性证书，锚点加权使 Flip@K 降 53–81%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/embedding_subspace_partitioning_for_dynamic_multi_objective_retrieval/">Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval</a></strong></dt>
<dd>
<p>将双塔嵌入切分为任务感知正交子空间，用服务时可调权重的加权内积替代单一点积，单模型即可在线遍历多目标 Pareto 前沿。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/from_click_modeling_to_offline_and_off_policy_evaluation_in_carousel_recomm/">From Click Modeling to Offline and Off-Policy Evaluation in Carousel Recommendation</a></strong></dt>
<dd>
<p>基于 RecGaze 眼动数据推翻轮播界面四大行为假设，提出以可观测变量为核心的点击模型设计框架、离散选择点击模型与 N2DCG 重构，并规划轮播专用 OPE 路线。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/genpage_towards_end_to_end_generative_homepage_construction_at_netflix/">GenPage: Towards End-to-End Generative Homepage Construction at Netflix</a></strong></dt>
<dd>
<p>用单个 Transformer 将用户上下文 token 化后自回归生成整页多行首页，WBC/RL 后训练，线上 +0.24% 核心参与度且延迟降 20%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/heterogeneous_ranking_in_industrial_scale_recommender_systems_a_case_study/">Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study</a></strong></dt>
<dd>
<p>面向 Google Discover 异构信息流的 HA-MoE 多门控专家架构，配合 LENS 可观测性与 DL-AUC 双层评估，离线与线上均获稳定增益。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/ctr_feature_interaction/hierarchical_gtv_estimation_bridging_the_gap_between_container_level_rankin/">Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion</a></strong></dt>
<dd>
<p>面向容器式电商的层级 GTV 预估框架 HGE，用集合感知编码器 + 代理标签分解 + 残差融合解决粒度错配，离线 XAUC +0.63pp、线上 RPS +1.42%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/impression_share_prediction_an_offline_evaluation_task_for_ranking_systems/">Impression Share Prediction: An Offline Evaluation Task for Ranking Systems</a></strong></dt>
<dd>
<p>提出曝光份额预测作为离线评估新任务，用因果DAG证明反事实可识别，PatchTST编码器在首小时恢复+22% L1。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/learned_cross_task_relationships_in_multi_task_models/">Learned Cross-Task Relationships in Multi-Task Models</a></strong></dt>
<dd>
<p>用交叉标签/无条件辅助头显式逼近任务两两联合分布，在 YouTube 三大场景线上 A/B 验证迁移与 RL 状态增强双重收益。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/lightweight_ranking_heads_accelerating_multi_task_experimentation_in_produc/">Lightweight Ranking Heads: Accelerating Multi-Task Experimentation in Production Recommender Systems</a></strong></dt>
<dd>
<p>以 stop-gradient + 无状态日训的轻量头动态注入新任务，免骨干重训，将多任务实验周期从数周压缩至数天。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/rl_bandit_rec/long_term_user_engagement_optimization_through_model_agnostic_downstream_re/">Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning</a></strong></dt>
<dd>
<p>以 P2P 兔子洞深度、浅层关闭负奖励与新用例采纳三类下游奖励替代稀疏留存标签，在 Pinterest 多场景线上 A/B 中稳定提升会话与留存指标。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/multi_objective_ranking_for_live_streaming_balancing_fresh_and_delayed_sign/">Multi-Objective Ranking for Live-Streaming: Balancing Fresh and Delayed Signals with Segment-Aware Targeting</a></strong></dt>
<dd>
<p>Twitch 直播精排用延迟窗口+新鲜/延迟双模型分离+分段推理加权+MMoE，DAV +0.09%、D 段 ARPU +0.56%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/normative_alignment_of_recommender_systems_via_internal_label_shift/">Normative Alignment of Recommender Systems via Internal Label Shift</a></strong></dt>
<dd>
<p>将编辑/公平目标建模为属性层内部标签偏移，用免重训的闭式重加权实现系统级规范对齐，KL 大幅下降且 AUC 基本无损。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/pairwise_ranking_outperforms_single_action_rl_for_offline_explanation_selec/">Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson</a></strong></dt>
<dd>
<p>将 LLM 解释生成离线化为冻结候选池，用 CPU 端 LambdaRank 逐候选打分选择，F1=0.500 超越 G-Refer/XRec 且延迟&lt;100ms。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/personalized_recommendation_tool_learning_via_autonomous_language_agents/">Personalized Recommendation Tool Learning via Autonomous Language Agents</a></strong></dt>
<dd>
<p>让 LLM 智能体做个性化工具选择、传统推荐模型做全库打分，以架构设计绕开幻觉与上下文长度限制，Full-Ranking 下 R@10 最高提升 68.8%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/position_bias_undermines_preference_consistency_in_listwise_llm_based_reran/">Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking</a></strong></dt>
<dd>
<p>将 LLM 列表重排的候选置换视为诱导偏好系统，提出 PPI/GPI/LOC 三层一致性度量，揭示有效性、曝光均衡与置换一致性三者可背离。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/shape_your_feed_an_llm_based_agentic_system_for_conversational_recommendati/">Shape Your Feed: An LLM-based Agentic System for Conversational Recommendation</a></strong></dt>
<dd>
<p>三层 Agentic 架构（感知/服务/自进化）实现多模态实时用户可控重排，离线对齐打分 98.85% 准确率，线上 dislike 降 2.74%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/towards_full_candidate_interaction_a_comprehensive_comparison_network_for_b/">Towards Full Candidate Interaction: A Comprehensive Comparison Network for Better Route Recommendation</a></strong></dt>
<dd>
<p>提出比较优先范式，用非重叠路段构造显式比较张量并在成对空间直接推理，线上CR +1.2%、偏航率 -0.8%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/comprehensive_list_generation_for_multi_generator_reranking/">Comprehensive List Generation for Multi-Generator Reranking</a></strong></dt>
<dd>
<p>提出多生成器-评估器重排框架与列表全面性指标，用 CLIG 自动学习互补生成器，离线 AUC 提升 0.9%、线上观看时长 +0.48%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/csmf_cascaded_selective_mask_fine_tuning_for_multi_objective_embedding_base/">CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval</a></strong></dt>
<dd>
<p>借鉴 PEFT 选择性掩码思想，按曝光→点击→转化级联顺序分阶段微调双塔召回，零参数增量实现多目标融合检索。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/hybrid_advertising_in_the_sponsored_search/">Hybrid Advertising in the Sponsored Search</a></strong></dt>
<dd>
<p>提出混合广告拍卖模型与 HRegNet 神经网络机制，在近 DSIC/IR 约束下联合优化独立店铺与店铺-品牌捆绑包的广告位分配与定价，真实数据上收益显著超越 VCG 与 JRegNet。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/interest_changes_considering_user_interest_life_cycle_in_recommendation_sys/">Interest Changes: Considering User Interest Life Cycle in Recommendation System</a></strong></dt>
<dd>
<p>用曝光/点击/交互直方图+VQ聚类刻画兴趣生命周期，经特征重校准与专家网络融合注入MMOE精排，Lofter线上CTR+0.38%、CVR+1.04%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/generative_rec/killing_two_birds_with_one_stone_unifying_retrieval_and_ranking_with_a_sing/">Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model</a></strong></dt>
<dd>
<p>将召回与排序统一为单生成模型的位置级序列生成任务，配合排序驱动增强器与梯度自适应加权实现协同与同步优化，全库排序下召回排序双提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/nr4der_neural_re_ranking_for_diversified_exercise_recommendation/">NR4DER: Neural Re-ranking for Diversified Exercise Recommendation</a></strong></dt>
<dd>
<p>用 mLSTM 难度过滤 + 活跃学生序列增强 + 神经重排，兼顾长尾学生与个性化学习节奏的习题推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/recgaze_the_first_eye_tracking_and_user_interaction_dataset_for_carousel_in/">RecGaze: The First Eye Tracking and User Interaction Dataset for Carousel Interfaces</a></strong></dt>
<dd>
<p>首个轮播界面眼动+点击+光标+选择解释数据集，87用户3477次交互，揭示F型浏览与位置偏置。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/llm_rec/review_driven_personalized_preference_reasoning_with_large_language_models_/">Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>用评论抽取偏好画像并蒸馏逐步推理，评分预测与重排双任务均超 CF 与 LLM 基线，解释性显著提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/towards_principled_learning_for_re_ranking_in_recommender_systems/">Towards Principled Learning for Re-ranking in Recommender Systems</a></strong></dt>
<dd>
<p>提出收敛一致性与对抗一致性两条重排学习原则，以自洽重排与相邻换位扰动构造额外监督，即插即用地提升多种重排基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/additive_control_variates_dominate_self_normalisation_in_off_policy_evaluat/">Additive Control Variates Dominate Self-Normalisation in Off-Policy Evaluation</a></strong></dt>
<dd>
<p>证明最优加性基线 β*-IPS 在 MSE 上渐近严格优于 SNIPS，并给出精确方差差解析式与排序场景 β*⊥⊥-IPM 扩展。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong></dt>
<dd>
<p>用残差量化码本近似 Target Attention，将交互式建模引入粗排，AUC 0.799、线上 GMV +7.57%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/equity_vs_equality_optimizing_ranking_fairness_for_tailored_provider_needs/">Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs</a></strong></dt>
<dd>
<p>提出以增益为基的公平框架与 EquityRank 梯度重排算法，按提供者异质偏好分配曝光与销量收益，离线/在线仿真均取得更优效果-公平权衡。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/exploring_test_time_scaling_via_prediction_merging_on_large_scale_recommend/">Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出预测合并式测试时缩放，用异构架构与同构随机种子生成多样 logit 并平均融合，三数据集八骨干上稳定超越参数缩放。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/following_the_eye_tracking_evidence_established_web_search_assumptions_fail/">Following the Eye-Tracking Evidence: Established Web-Search Assumptions Fail in Carousel Interfaces</a></strong></dt>
<dd>
<p>用 RecGaze 眼动数据系统证伪 F 型注意力、层级点击与标题导航等四条 Web 搜索假设，提出轮播界面特有的 L 型注视模式。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层消除候选顺序依赖，单次前向实现近完美排列不变性，nDCG 仅微降。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/post_hoc_provider_fairness_adaptation_via_hierarchical_exposure_alignment/">Post-hoc Provider Fairness Adaptation via Hierarchical Exposure Alignment</a></strong></dt>
<dd>
<p>冻结基座+轻量公平适配器，用分层曝光对齐(HEFA)与可微NDCG联合优化，实现即插即用的提供方曝光公平。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/the_attention_market_interpreting_online_fair_re_ranking_as_manifold_optimi/">The Attention Market: Interpreting Online Fair Re-ranking as Manifold Optimization under Walrasian Equilibrium</a></strong></dt>
<dd>
<p>将在线公平重排建模为注意力市场瓦尔拉斯均衡，用供需双侧梯度校正对齐排序流形，7 数据集 42 设置几乎全胜。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/think_when_needed_model_aware_reasoning_routing_for_llm_based_ranking/">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></strong></dt>
<dd>
<p>提出轻量即插即用的模型感知推理路由器，在生成前逐实例决定 Think/Non-Think，以 Pareto 前沿策略在降 token 的同时提升排序效用。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong></dt>
<dd>
<p>用 LoRA+MoE 独立参数通路隔离稀疏问卷满意度信号，并以在线 DPO 对齐主排序模型，离线 NDCG@5 提升约 3%，线上 Q-Satisfied 提升 0.5–0.9%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong></dt>
<dd>
<p>用后验价值对齐把抽象混合分映射到有效完播率语义空间，再以独立线性 boosting 解耦多业务加权，线上 A/B 全指标正向。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/unifying_search_and_recommendation_in_llms_via_gradient_multi_subspace_tuni/">Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning</a></strong></dt>
<dd>
<p>用梯度多子空间分解+零空间投影替代 LoRA，在统一 S&amp;R 生成式建模中消解任务梯度冲突并保住通用知识，Qilin/Amazon 上大幅超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/fairsort_learning_to_fair_rank_for_personalized_recommendations_in_two_side/">FairSort: Learning to Fair Rank for Personalized Recommendations in Two-Sided Platforms</a></strong></dt>
<dd>
<p>将重排视为“跑道”而非背包，用公平提升速度+二分搜索保证用户最低效用，实现双边公平与个性化推荐的三方权衡。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/multi_task_reranking/efficient_multi_task_prompt_tuning_for_recommendation/">Efficient Multi-task Prompt Tuning for Recommendation</a></strong></dt>
<dd>
<p>提出两阶段 MPT-Rec：GAN 解耦任务共享/特有信息后冻结主干，仅用任务感知 Prompt 微调新任务，参数降至 10% 且性能保持 95%+。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/llm_rec/recranker_instruction_tuning_large_language_model_as_ranker_for_top_k_recom/">RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation</a></strong></dt>
<dd>
<p>以自适应用户采样、位置偏移与检索信号增强构建三类排序指令微调 LLM 重排器，并提出混合排序集成，Leave-One-Out 全库排序下 HR/NDCG 最高提升 213%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/multi_task_reranking/user_invariant_preference_learning_for_multi_behavior_recommendation/">User Invariant Preference Learning for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用 IRM 范式 + VAE 从多行为交互中抽取用户不变偏好以抑制辅助行为噪声，Leave-One-Out + Full-Ranking 下 HR@10 最高提升 3.98%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/dynamic_fairness_aware_recommendation_through_multi_agent_social_choice/">Dynamic Fairness-aware Recommendation through Multi-agent Social Choice</a></strong></dt>
<dd>
<p>将多利益相关方公平推荐建模为「分配+聚合」两阶段社会选择问题，SCRUF-D 框架动态调度多个公平性智能体，在真实与合成数据上实现多维权衡。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/robustness_analysis_of_pareto_based_joint_evaluation_of_fairness_and_releva/">Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems</a></strong></dt>
<dd>
<p>提出 DPFR：以测试集真实标签构建相关性-公平性帕累托前沿，用模型得分到前沿参考点的距离联合评估推荐系统，比现有联合度量更稳健。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/multi_task_reranking/sequential_diversification_with_provable_guarantees/">Sequential Diversification with Provable Guarantees</a></strong></dt>
<dd>
<p>将多样性从集合度量升级为序列期望度量，证明 MaxSSD 的 NP 难性并给出 BτI/SBτI 等常数近似算法，在 7 个数据集上稳定超越 EXPLORE/MMR/DPP。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/cross_domain_multi_scenario/towards_personalized_federated_multi_scenario_multi_task_recommendation/">Towards Personalized Federated Multi-Scenario Multi-Task Recommendation</a></strong></dt>
<dd>
<p>首个联邦多场景多任务推荐框架 PF-MSMTrec，通过专家参数解耦、联邦批归一化与冲突协调缓解数据/模型/任务三级冲突，AliExpress 与 Tenrec 上 CTR/CTCVR AUC 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/cross_domain_multi_scenario/automated_information_flow_selection_for_multi_scenario_multi_task_recommen/">Automated Information Flow Selection for Multi-scenario Multi-task Recommendation</a></strong></dt>
<dd>
<p>用 LoRA 解耦四类场景-任务信息单元并以可学习掩码剪枝无效信息流，AUC 全面超越 HiNet/M3oE，线上 CVR 均涨 2.46%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/multi_task_reranking/cold_starting_podcast_ads_and_promotions_with_multi_task_learning_on_spotif/">Cold-Starting Podcast Ads and Promotions with Multi-Task Learning on Spotify</a></strong></dt>
<dd>
<p>Spotify 用单一 5 任务 MTL 模型统一播客广告与推广定向，靠非对称损失掩码与源平衡采样实现跨渠道迁移，线上 eCPS 降 22%、i2s 升 18–24%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multi_task_reranking/a_hybrid_cross_stage_coordination_pre_ranking_model_for_online_recommendati/">A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems</a></strong></dt>
<dd>
<p>提出 HCCP 粗排模型，融合上游召回与下游精排多级未曝光样本，用 Margin InfoNCE 联合优化一致性与长尾精度，JD 线上 UCVR+14.9%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/efficient_and_practical_approximation_algorithms_for_advertising_in_content/">Efficient and Practical Approximation Algorithms for Advertising in Content Feeds</a></strong></dt>
<dd>
<p>针对信息流原生广告位分配，提出反向贪心 2-近似算法，突破前人 4-近似瓶颈，并在合成与真实数据上首次系统评测。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/joint_evaluation_of_fairness_and_relevance_in_recommender_systems_with_pare/">Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier</a></strong></dt>
<dd>
<p>提出 DPFR：以测试集构造相关性-公平性帕累托前沿，用模型得分到前沿的欧氏距离作为联合评测指标，揭示现有联合指标排序与帕累托最优解不一致。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multi_task_reranking/llm4rerank_llm_based_auto_reranking_framework_for_recommendations/">LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations</a></strong></dt>
<dd>
<p>将准确性/多样性/公平性抽象为全连接图节点，用零样本 LLM 以 CoT 多跳自动重排，三数据集上多准则均衡超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multi_task_reranking/nlgr_utilizing_neighbor_lists_for_generative_rerank_in_personalized_recomme/">NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems</a></strong></dt>
<dd>
<p>用邻居列表相对奖励对齐评估器-生成器目标，并以采样式非自回归生成灵活跳转组合空间，美团线上 CTR +3.25%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multi_task_reranking/pcl_prompt_based_continual_learning_for_user_modeling_in_recommender_system/">PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems</a></strong></dt>
<dd>
<p>用位置级+上下文提示做持续用户建模，冻结骨干仅训提示，缓解灾难性遗忘，6/8 任务超越 MTL 上界。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/prectr_a_synergistic_framework_for_integrating_personalized_search_relevanc/">PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction</a></strong></dt>
<dd>
<p>用条件概率融合把相关性分级与CTR统一进单模型，配两阶段训练、语义一致性正则与个性化相关性激励，线上CTR+0.4%、GMV+1.1%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/uncertainty_quantification_and_decomposition_for_llm_based_recommendation/">Uncertainty Quantification and Decomposition for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用 Plackett-Luce 单次推理估计列表排序预测熵，并分解为推荐/提示不确定性，指导不确定性感知提示提升 N@20 约 17%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multi_task_reranking/xmtf_a_formula_free_model_for_reinforcement_learning_based_multi_task_fusio/">xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems</a></strong></dt>
<dd>
<p>用可学习单调融合单元替代预定义融合公式，配合两阶段混合训练，离线总观看时长提升约7%，线上日观看时长+0.833%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/grank_towards_target_aware_and_streamlined_industrial_retrieval_with_a_gene/">GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework</a></strong></dt>
<dd>
<p>用训练期目标注入把 Target-Aware 能力蒸馏进 MIPS 兼容生成器，再以轻量交叉注意力重排，免结构化索引实现 Recall@500 +30% 与 1.7× QPS。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multi_task_reranking/not_all_candidates_are_created_equal_a_heterogeneity_aware_approach_to_pre_/">Not All Candidates are Created Equal: A Heterogeneity-Aware Approach to Pre-ranking in Recommender Systems</a></strong></dt>
<dd>
<p>针对粗排异构候选的梯度冲突与算力错配，提出 GHCL 分组对比损失与难度感知双模型路由，AUC 全面超越 SOTA 且线上降本 6%。</p>
</dd>
</dl>

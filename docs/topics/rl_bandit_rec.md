# 🎮 强化学习与动态探索推荐 (RL, Bandits & Long-Term Engagement)

> **研究范畴**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

当前方向共收录 **86** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/cold_start_rec/bayescns_a_unified_bayesian_approach_to_address_cold_start_and_non_stationa/">BayesCNS: A Unified Bayesian Approach to Address Cold Start and Non-Stationarity in Search Systems at Scale</a></strong></dt>
<dd>
<p>用经验贝叶斯先验网络+Thompson采样在线更新，统一解决物品冷启动与非平稳漂移，线上A/B新物品曝光+10.60%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/rl_bandit_rec/contrastive_representation_for_interactive_recommendation/">Contrastive Representation for Interactive Recommendation</a></strong></dt>
<dd>
<p>用偏好排序对比学习(PRCL)为 DRL 交互推荐提炼状态表征，冷启动下样本效率显著提升。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/agent_conversational_rec/llm_powered_user_simulator_for_recommender_system/">LLM-Powered User Simulator for Recommender System</a></strong></dt>
<dd>
<p>用 LLM 离线蒸馏物品优缺点关键词，构建关键词匹配+语义相似+序列统计三模型集成用户模拟器，为 RL 推荐提供可解释高保真环境。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/rl_bandit_rec/neural_combinatorial_clustered_bandits_for_recommendation_systems/">Neural Combinatorial Clustered Bandits for Recommendation Systems</a></strong></dt>
<dd>
<p>用双神经网络+上下文聚类替代组合老虎机的精确优化 oracle，实现 $\tilde{O}(\tilde{d}\sqrt{T})$ 遗憾并显著降低推荐累积遗憾。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/rl_bandit_rec/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augment/">Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback</a></strong></dt>
<dd>
<p>用增强置信界把相似/依赖观测注入无模型 Dueling Bandit，给出样本复杂度与遗憾权衡，在推荐、多目标优化与 LLM 响应优化上显著降遗憾。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/lamp_val_large_language_models_empower_personalized_valuation_in_auction/">LaMP-Val: Large Language Models Empower Personalized Valuation in Auction</a></strong></dt>
<dd>
<p>用 LLM 从用户评论语义中做个性化估值，构建 23k 估值数据集 + 多样化指令 SFT，并提出 PU/PV 闭环拍卖指标，PU 较 GPT-4 提升约 6.5 倍。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/user_feedback_alignment_for_llm_powered_exploration_in_large_scale_recommen/">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></strong></dt>
<dd>
<p>解耦新颖性与用户对齐双 LLM，用集体隐式反馈训练对齐模型做 best-of-n 选择，线上同时提升新颖曝光与满意度。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/rl_bandit_rec/world_model_guided_reinforcement_learning_via_counterfactual_user_engagemen/">World Model-Guided Reinforcement Learning via Counterfactual User Engagement Simulation</a></strong></dt>
<dd>
<p>将推荐视为动作、异构反馈视为观测，用语言世界模型 UEWM 做反事实用户模拟器，为 1.7B 策略提供同状态稠密奖励，跨域零样本超越大模型。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/rl_bandit_rec/generative_flow_networks_for_personalized_multimedia_systems_a_case_study_o/">Generative Flow Networks for Personalized Multimedia Systems: A Case Study on Short Video Feeds</a></strong></dt>
<dd>
<p>用 GFlowNet 多候选生成替代单动作 RL 策略，在短视频预取场景实现个性化 QoE 与带宽双优化。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/rl_bandit_rec/energy_guided_diffusion_sampling_for_long_term_user_behavior_prediction_in_/">Energy-Guided Diffusion Sampling for Long-Term User Behavior Prediction in Reinforcement Learning-based Recommendation</a></strong></dt>
<dd>
<p>用扩散策略替代行为克隆以提升离线RL4RS策略表达能力，并以Q值能量引导采样降低随机性，六数据集与仿真环境全面领先。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/rl_bandit_rec/expert_guided_diffusion_planner_for_auto_bidding/">Expert-Guided Diffusion Planner for Auto-bidding</a></strong></dt>
<dd>
<p>用专家轨迹引导条件扩散规划器生成出价序列，配合混合强制训练与跳步采样，线上转化+11.29%、收入+12.36%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/social_group_rec/heterogeneous_influence_maximization_in_user_recommendation/">Heterogeneous Influence Maximization in User Recommendation</a></strong></dt>
<dd>
<p>将异质传播概率与 RR 集影响力最大化引入好友邀请推荐，HeteroIR/HeteroIM 在 NSpread/Recall/NDCG 全面领先并落地腾讯游戏 A/B 提升 8.5%/10%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/rl_bandit_rec/kp_agent_keyword_pruning_in_sponsored_search_advertising_via_llm_powered_co/">KP-Agent: Keyword Pruning in Sponsored Search Advertising via LLM-Powered Contextual Bandits</a></strong></dt>
<dd>
<p>将关键词剪枝建模为上下文赌博机，用 LLM Agent 生成代码片段执行剪枝，美团真实数据上累计利润最高提升 49.28%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/rl_bandit_rec/maximum_in_support_return_modeling_for_dynamic_recommendation_with_language/">Maximum In-Support Return Modeling for Dynamic Recommendation with Language Model Prior</a></strong></dt>
<dd>
<p>将轨迹拼接从训练期移至动作推理期，用 expectile 回归估计最大在支撑回报并自适应截断历史，配合 LLM 先验初始化 DT，在 5 个离线数据集与 VirtualTB 仿真上全面超越 EDT4Rec。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/rl_bandit_rec/stratified_expert_cloning_for_retention_aware_recommendation_at_scale/">Stratified Expert Cloning for Retention-Aware Recommendation at Scale</a></strong></dt>
<dd>
<p>用分层专家行为克隆替代 RL 探索，配合动作熵正则与自适应专家选择，在快手双平台 A/B 中提升活跃天数 0.098%/0.122%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/rl_bandit_rec/efficient_offline_learning_of_ranking_policies_via_top_k_policy_decompositi/">Efficient Offline Learning of Ranking Policies via Top-$k$ Policy Decomposition</a></strong></dt>
<dd>
<p>将排序策略分解为 top-$k$ 策略梯度与 bottom 回归两段，在 CPC 条件下无偏且大幅降方差，小样本与大规模排序空间下显著优于 IPS/DR/RIPS/IIPS。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/rl_bandit_rec/evol_simulator_guided_evolutionary_expert_synthesis_for_deployment_free_lea/">EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation</a></strong></dt>
<dd>
<p>用知识追踪模拟器内进化搜索合成专家路径，蒸馏为仅凭初始掌握度盲规划的非对称 Actor-Critic，实现免模拟器部署的学习路径推荐。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/funnelcausalnet_funnel_aware_joint_conversion_revenue_uplift_for_multi_tier/">FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation</a></strong></dt>
<dd>
<p>以漏斗恒等式 μ_gmv=μ_conv·μ_val 耦合双头 uplift 估计，配 RCT 锚定拉格朗日预算分配与 Bonferroni 联合保形审计带，工业多档券 RCT 上 7/7 锚点 ΔROI 最优。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/learning_from_unreachable_rewards_hint_conditioned_reinforcement_learning_f/">Learning from Unreachable Rewards: Hint-Conditioned Reinforcement Learning for Generative Recommendation</a></strong></dt>
<dd>
<p>用 SFT 检查点诊断语义 ID 前缀树可达性，对不可达样本注入最短目标前缀提示并以提示感知信用分解（前缀 SFT + 后缀 GRPO）恢复学习信号，零优势组从 70%+ 降至 20% 以下。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2026/rl_bandit_rec/llm_enhanced_reinforcement_learning_for_long_term_user_satisfaction_in_inte/">LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation</a></strong></dt>
<dd>
<p>LLM 做类别级语义规划、PPO 做物品级策略，KuaiSim 全会话仿真下 Tint/Rcum 显著超越 DNaIR 等 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/rl_bandit_rec/beyond_static_llm_policies_imitation_enhanced_reinforcement_learning_for_re/">Beyond Static LLM Policies: Imitation-Enhanced Reinforcement Learning for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 轨迹作次优示范，经对抗逆强化学习提取奖励并加权模仿，离线 RL 策略在 Steam/Amazon 上累积奖励较 BiLLP 提升 13%+。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/agent_conversational_rec/enhancing_personalized_multi_turn_dialogue_with_curiosity_reward/">Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward</a></strong></dt>
<dd>
<p>将用户模型信念增益作为 PBRS 内在奖励注入多轮 RLHF，使 LLM 主动探询用户属性，习题推荐成功率 54%→87.5%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/fine_grained_list_wise_alignment_for_generative_medication_recommendation/">Fine-grained List-wise Alignment for Generative Medication Recommendation</a></strong></dt>
<dd>
<p>将用药推荐重构为逐药序列决策，提出逐步GRPO与势函数奖励塑形实现细粒度信用分配，在MIMIC-III上Jaccard达0.4836并支持安全-精度可控权衡。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/rl_bandit_rec/learning_personalized_ad_impact_via_contextual_reinforcement_learning_under/">Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards</a></strong></dt>
<dd>
<p>将自动出价建模为带延迟泊松奖励的CMDP，提出两阶段MLE与数据分割，实现个性化出价近最优遗憾界。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/rl_bandit_rec/mtrec_learning_to_align_with_user_preferences_via_mental_reward_models/">MTRec: Learning to Align with User Preferences via Mental Reward Models</a></strong></dt>
<dd>
<p>用分布化逆强化学习(QR-IQL)从行为轨迹反推用户心理奖励分布，作为辅助对齐信号注入CTR/RL推荐模型，离线AUC/NCIS全面提升，工业短视频A/B观看时长+7%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/optimal_auction_design_in_the_joint_advertising/">Optimal Auction Design in the Joint Advertising</a></strong></dt>
<dd>
<p>联合广告单槽位Myerson最优机制+多槽位BundleNet束级IC约束神经网络，逼近理论最优并显著提升平台收益</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/rl_bandit_rec/personalized_exercise_recommendation_with_semantically_grounded_knowledge_t/">Personalized Exercise Recommendation with Semantically-Grounded Knowledge Tracing</a></strong></dt>
<dd>
<p>ExRec 用 LLM 自动标注知识点+对比学习语义表征+KT 校准，并以 KT 模型做 Model-Based Value Estimation 稳定 Q-learning，在四个教育推荐任务上显著提升知识状态增益。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/think_before_recommendation_autonomous_reasoning_enhanced_recommender/">Think before Recommendation: Autonomous Reasoning-enhanced Recommender</a></strong></dt>
<dd>
<p>用纯 GRPO 强化学习替代教师蒸馏，让单个 LLM 自主习得「用户分析→物品分析→匹配→打分」推理链，评分预测 MAE 最高降 29.9%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/alm_mta_front_door_causal_multi_touch_attribution_method_for_creator_ecosys/">ALM-MTA: Front-Door Causal Multi-Touch Attribution Method for Creator-Ecosystem Optimization</a></strong></dt>
<dd>
<p>以前门识别+对抗代理中介实现大规模推荐系统多触点因果归因，离线 AUC 0.907、线上创作者 +0.6%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/rl_bandit_rec/beyond_match_maximization_and_fairness_retention_optimized_two_sided_matchi/">Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching</a></strong></dt>
<dd>
<p>将双边匹配推荐重构为留存最大化问题，用凹性 Jensen 下界把 NP-hard 排序松弛为 O(NlogN) 的 argsort 打分，在合成与真实约会平台数据上以约 70% 匹配量取得最高留存率。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/copf_an_online_framework_for_deployment_stable_counterfactual_fairness_in_e/">COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs</a></strong></dt>
<dd>
<p>以曝光反事实定义群体机会差，用倾向日志+GA-DR残差+Residual-OI证书在线审计并控制演化图链路推荐的部署稳定公平性。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/rl_bandit_rec/incentivized_exploration_with_stochastic_covariates_a_two_stage_mechanism_d/">Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System</a></strong></dt>
<dd>
<p>面向在线随机用户协变量的激励相容探索，提出两阶段 RCB 算法（冷启动补贴 + 逆比例间隙采样），在满足 ε-DBIC 约束下达到 $\tilde{O}(\sqrt{KdT})$ 遗憾。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/on_the_coordination_of_value_maximizing_bidders/">On the Coordination of Value-Maximizing Bidders</a></strong></dt>
<dd>
<p>证明价值最大化自动出价方在“最高价值者出价”协调机制下，RoS 合规与联盟总价值均优于独立出价，并给出充要条件与渐近最优性。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/rl_bandit_rec/optimal_and_efficient_contextual_combinatorial_semi_bandits_with_general_fu/">Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation</a></strong></dt>
<dd>
<p>提出 SquareCB.Comb，用回归 oracle + 对数障碍边际优化实现通用函数逼近下 CCSB 的极小极大最优 regret，并在网页推荐与最短路任务上验证。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/rl_bandit_rec/prorl_effective_reinforcement_learning_for_proactive_recommendation_via_rec/">ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation</a></strong></dt>
<dd>
<p>揭示主动推荐中策略梯度的长度捷径与高方差缺陷，提出逐步奖励中心化与位置特定优势估计，三数据集全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/IJCAI/2026/rl_bandit_rec/pace_a_personalized_adaptive_curriculum_engine_for_9_1_1_call_taker_trainin/">PACE: A Personalized Adaptive Curriculum Engine for 9-1-1 Call-taker Training</a></strong></dt>
<dd>
<p>用技能图信念传播+个体学习/遗忘动力学+Thompson Sampling 上下文老虎机为 9-1-1 调度员推荐训练场景，Z2H 缩短 19.50%、终局掌握度提升 10.95%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/rl_bandit_rec/a_unified_online_offline_framework_for_co_branding_campaign_recommendations/">A Unified Online-Offline Framework for Co-Branding Campaign Recommendations</a></strong></dt>
<dd>
<p>首个联名营销推荐系统性研究：二分图建模+方差感知CUCB在线学习+部分枚举离线预算分配，理论保证近优regret与1-1/e近似，真实数据收益提升12%-73%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/rl_bandit_rec/an_adaptable_budget_planner_for_enhancing_budget_constrained_auto_bidding_i/">An Adaptable Budget Planner for Enhancing Budget-Constrained Auto-Bidding in Online Advertising</a></strong></dt>
<dd>
<p>将竞价周期切分为多阶段，用 in-context RL 逐 episode 调整预算分配，few-shot 适配新广告主，仿真与线上 A/B 均提升累计价值。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/beyond_advertising_mechanism_design_for_platform_wide_marketing_service_qua/">Beyond Advertising: Mechanism Design for Platform-Wide Marketing Service &#x27;QuanZhanTui&#x27;</a></strong></dt>
<dd>
<p>提出全站推场景下的库存约束卖家模型与流式支付拍卖 LPA，在合成与真实数据上同时提升流式福利、用户体验与平台收入。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/beyond_item_dissimilarities_diversifying_by_intent_in_recommender_systems/">Beyond Item Dissimilarities: Diversifying by Intent in Recommender Systems</a></strong></dt>
<dd>
<p>用实时意图概率驱动整页贪心重排，以贝叶斯后验更新替代物品相似度，YouTube 线上 DAU +0.05%、用户享受度 +0.09%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/agent_conversational_rec/leveraging_the_power_of_conversations_optimal_key_term_selection_in_convers/">Leveraging the Power of Conversations: Optimal Key Term Selection in Conversational Contextual Bandits</a></strong></dt>
<dd>
<p>用平滑关键词语境与最小特征值自适应触发对话，将对话式上下文 Bandit 遗憾降至近极小极大最优 O(√(dT logT))，累积遗憾较基线降低至少 14.6%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/lightweight_auto_bidding_based_on_traffic_prediction_in_live_advertising/">Lightweight Auto-bidding based on Traffic Prediction in Live Advertising</a></strong></dt>
<dd>
<p>直播广告场景下，用LightGBM预测全时段累计成本/点击，结合LP对偶最优出价公式与投影梯度下降求解对偶变量，实现秒级轻量自动出价，线上GMV+4.28%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/measure_domain_s_gap_a_similar_domain_selection_principle_for_multi_domain_/">Measure Domain&#x27;s Gap: A Similar Domain Selection Principle for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用原型距离+bandit动态为每个域挑选相似域子集，解耦共享专家以缓解多域推荐负迁移，三数据集AUC稳定提升。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/rl_bandit_rec/multi_task_offline_reinforcement_learning_for_online_advertising_in_recomme/">Multi-task Offline Reinforcement Learning for Online Advertising in Recommender Systems</a></strong></dt>
<dd>
<p>用因果状态编码器+因果注意力+多任务解码器把广告渠道推荐与预算分配统一进离线RL，KuaiRand/Criteo 上 F1 提升约 7.5%/16.2%，线上 CTR +0.08。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/rl_bandit_rec/revisiting_clustering_of_neural_bandits_selective_reinitialization_for_miti/">Revisiting Clustering of Neural Bandits: Selective Reinitialization for Mitigating Loss of Plasticity</a></strong></dt>
<dd>
<p>针对神经聚类 Bandit 的可塑性丧失问题，提出选择性重初始化框架 SeRe，配合自适应变点检测，在六个推荐数据集上最多降低 12.82% 累积遗憾。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/rl_bandit_rec/session_level_dynamic_ad_load_optimization_using_offline_robust_reinforceme/">Session-Level Dynamic Ad Load Optimization using Offline Robust Reinforcement Learning</a></strong></dt>
<dd>
<p>将会话级广告加载建模为鲁棒MDP，用IPM不确定集导出免额外开销的鲁棒对决DQN，离线AUCC提升80%+，线上A/B双位数增益。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/deep_learning_causal_retrieval_optimization_for_efficient_e_commerce_distri/">Deep-learning Causal Retrieval Optimization for Efficient e-commerce Distribution in Pinterest</a></strong></dt>
<dd>
<p>将购物候选生成器触发建模为因果决策，用 DR 多任务提升学习+随机 Holdout+线性回放，线上触发率降 85% 而购物会话持平。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/heterogeneous_multi_treatment_uplift_modeling_for_trade_off_optimization_in/">Heterogeneous Multi-treatment Uplift Modeling for Trade-off Optimization in Short-Video Recommendation</a></strong></dt>
<dd>
<p>用异构多处理增益建模+在线动态权重，个性化权衡短视频消费与商业化指标，快手全量部署</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/hierarchical_residual_policy_optimization_for_generative_recommendations/">Hierarchical Residual Policy Optimization for Generative Recommendations</a></strong></dt>
<dd>
<p>将物品级反馈分解为 SID 前缀残差信用，用 RRPO 做保守 token 级策略优化，KuaiSim 会话回报 10.53 vs 8.14，线上 IAA Target Cost 最高 +3.49%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/hoba_hierarchical_on_policy_bidding_agents_for_adaptive_online_advertising/">HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising</a></strong></dt>
<dd>
<p>三层时间尺度解耦出价：LLM 小时级推理超参、SARSA 分钟级因果去偏选专家、专家池逐次拍卖执行，AuctionNet 与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/learning_peer_influence_probabilities_with_linear_contextual_bandits/">Learning Peer Influence Probabilities with Linear Contextual Bandits</a></strong></dt>
<dd>
<p>将同伴影响力概率学习建模为双目标线性上下文 Bandit，刻画 Regret–RMSE 帕累托前沿并提出 InfluenceCB 以单参数 β 导航权衡。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/uncertainty_calibrated_recommendations_for_low_active_users/">Uncertainty-Aware Adaptive Recommendation across User Lifecycle</a></strong></dt>
<dd>
<p>用 EPE Critic 与经验贝叶斯先验方差量化逐样本认知不确定性，对低活跃用户风险规避降权、高活跃用户 UCB 探索加成，14 天线上 A/B 验证留存与多样性双收益。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/agentic_personalisation_of_cross_channel_marketing_experiences/">Agentic Personalisation of Cross-Channel Marketing Experiences</a></strong></dt>
<dd>
<p>用 DiD 因果估计 + Thompson 采样构建模块化营销智能体，在 640 万用户线上 RCT 中显著提升漏斗事件与 GMV。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/counterfactual_inference_under_thompson_sampling/">Counterfactual Inference under Thompson Sampling</a></strong></dt>
<dd>
<p>推导 Thompson Sampling 下动作倾向性的精确可计算闭式解，使 IPS 类离线评估在 TS 策略下首次可行，OBP 仿真验证无偏收敛。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/rl_bandit_rec/exploitation_over_exploration_unmasking_the_bias_in_linear_bandit_recommend/">Exploitation Over Exploration: Unmasking the Bias in Linear Bandit Recommender Offline Evaluation</a></strong></dt>
<dd>
<p>大规模实证揭示线性 Bandit 离线回放评估系统性偏袒纯利用策略，贪心模型在 89% 数据集上不劣于探索变体。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cold_start_rec/item_centric_exploration_for_cold_start_problem/">Item-centric Exploration for Cold Start Problem</a></strong></dt>
<dd>
<p>提出以物品为中心的 Beta 后验过滤控制，在排序后剔除“错配受众”曝光，线上探索曝光降 20% 而满意度升 50%、可推荐语料扩 10%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/rl_bandit_rec/off_policy_evaluation_and_learning_for_matching_markets/">Off-Policy Evaluation and Learning for Matching Markets</a></strong></dt>
<dd>
<p>面向双向匹配市场提出 DiPS/DPR 两阶段奖励 OPE 估计器，利用一阶段 scout 信号降方差，合成+Wantedly 真实 A/B 日志验证 MSE 与策略选择显著优于 DM/IPS/DR。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/learned_cross_task_relationships_in_multi_task_models/">Learned Cross-Task Relationships in Multi-Task Models</a></strong></dt>
<dd>
<p>用交叉标签/无条件辅助头显式逼近任务两两联合分布，在 YouTube 三大场景线上 A/B 验证迁移与 RL 状态增强双重收益。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/llm_reasoning_for_subjective_tasks_failure_modes_mitigation_and_dynamic_rea/">LLM Reasoning for Subjective Tasks: Failure Modes, Mitigation, and Dynamic Reasoning Routing</a></strong></dt>
<dd>
<p>揭示主观验证任务上显式推理与标准 RLVR 会引发推理崩溃，提出条件长度惩罚奖励与人格路由蓝图，Netflix 四任务 macro-F1 最高回升至 0.851。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/llm_based_user_personas_for_recommendations_at_scale/">LLM-Based User Personas for Recommendations at Scale</a></strong></dt>
<dd>
<p>工业级实时生成自然语言用户兴趣画像，蒸馏+异步推理支撑十亿用户在线服务，A/B 测试观看时长 +0.04%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/rl_bandit_rec/long_term_user_engagement_optimization_through_model_agnostic_downstream_re/">Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning</a></strong></dt>
<dd>
<p>以 P2P 兔子洞深度、浅层关闭负奖励与新用例采纳三类下游奖励替代稀疏留存标签，在 Pinterest 多场景线上 A/B 中稳定提升会话与留存指标。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/progressive_alignment_of_recommender_foundation_model_through_multi_phase_p/">Progressive Alignment of Recommender Foundation Model through Multi-Phase Post-Training</a></strong></dt>
<dd>
<p>将推荐基础模型后训练拆为 LP→FFT→RFT 三阶段，用稠密隐式反馈训练策略、稀疏业务指标仅训练奖励模型，离线 Rank NDCG 0.437→0.463，线上全漏斗指标显著优于非 FM 基线。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/self_evolving_recommendation_system_end_to_end_autonomous_model_optimizatio/">Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</a></strong></dt>
<dd>
<p>用 Gemini 双智能体（快循环提名+慢循环上线）自动改写优化器/架构/奖励函数，在 YouTube 线上 A/B 取得显著北极星指标提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/rl_bandit_rec/darlr_dual_agent_offline_reinforcement_learning_for_recommender_systems_wit/">DARLR: Dual-Agent Offline Reinforcement Learning for Recommender Systems with Dynamic Reward</a></strong></dt>
<dd>
<p>双智能体离线 RL 用选择器聚合相似用户动态重塑奖励与不确定性惩罚，四数据集累积奖励全面超越 ROLeR/DORL。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/rl_bandit_rec/enhancing_new_item_fairness_in_dynamic_recommender_systems/">Enhancing New-item Fairness in Dynamic Recommender Systems</a></strong></dt>
<dd>
<p>用 DQN 智能体 + 三重奖励（探索/公平/准确）在动态推荐系统中提升新物品曝光公平性，KuaiRec 上 TGF 降 86%+ 且 HR 反升 39%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/revisiting_algorithmic_audits_of_tiktok_poor_reproducibility_and_short_term/">Revisiting Algorithmic Audits of TikTok: Poor Reproducibility and Short-term Validity of Findings</a></strong></dt>
<dd>
<p>复现 TikTok 推荐审计发现可复现性极差、结论仅短期有效，观看行为才是最强个性化信号，且结论高度依赖度量选择。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/generative_bid_shading_in_real_time_bidding_advertising/">Generative Bid Shading in Real-Time Bidding Advertising</a></strong></dt>
<dd>
<p>用自回归残差生成+CHNet奖励模型+GRPO偏好对齐替代两阶段出价调价，离线SR达41.74%、线上ROI+3.4%且推理耗时降37.6%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/gflowgr_fine_tuning_generative_recommendation_frameworks_with_generative_fl/">GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks</a></strong></dt>
<dd>
<p>将生成式推荐重构为集合级序列生成，用 GFlowNets 微调使生成概率正比于行为感知奖励，缓解曝光偏差，淘宝线上年收入相对提升 0.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/reinforcing_user_interest_evolution_in_multi_scenario_learning_for_recommen/">Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems</a></strong></dt>
<dd>
<p>用 Double Q-learning 估计场景级行为置信度作为门控，加权三元组对比损失以抑制多场景负迁移，KuaiSAR 全库排序 N@20 提升约 3.3 倍。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/llm_rec/reinforced_prompt_personalization_for_recommendation_with_large_language_mo/">Reinforced Prompt Personalization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>将提示词拆为四类句式模式，用多智能体强化学习为每个用户逐句选择并组合个性化提示，在三个数据集上 NDCG@10 相对任务级提示提升 107%–132%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/agent_conversational_rec/vague_preference_policy_learning_for_conversational_recommendation/">Vague Preference Policy Learning for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 VPMCR 模糊偏好场景与 VPPL 框架，用软估计替代二值过滤，SR@15 最高提升 22%，平均轮次显著下降。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/rl_bandit_rec/choosing_the_best_of_both_worlds_diverse_and_novel_recommendations_through_/">Choosing the Best of Both Worlds: Diverse and Novel Recommendations through Multi-Objective Reinforcement Learning</a></strong></dt>
<dd>
<p>用标量化多目标 RL 头正则化序列推荐主干，同时优化准确率/多样性/新颖性，聚合覆盖度提升约 20% 且准确率不降。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/rl_bandit_rec/large_language_model_driven_policy_exploration_for_recommender_systems/">Large Language Model driven Policy Exploration for Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 蒸馏用户偏好离线预训练 A2C 推荐策略，并提出冻结+可学习双策略自适应在线适配，显著提升冷启动回报与长期收益。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/rl_bandit_rec/supervised_advantage_actor_critic_for_recommender_systems/">Supervised Advantage Actor-Critic for Recommender Systems</a></strong></dt>
<dd>
<p>为 RL 头引入负采样构造 SNQN，并用采样优势 A2C 重加权监督头，在 RC15/RetailRocket 上显著超越 SQN/SAC。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/rl_bandit_rec/auro_reinforcement_learning_for_adaptive_user_retention_optimization_in_rec/">AURO: Reinforcement Learning for Adaptive User Retention Optimization in Recommender Systems</a></strong></dt>
<dd>
<p>用值对齐状态抽象+拒绝采样守护探索，解决留存优化RL在非平稳环境下的适应与隐式冷启动难题，KuaiSim与线上A/B全面领先。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/bat_benchmark_for_auto_bidding_task/">BAT: Benchmark for Auto-bidding Task</a></strong></dt>
<dd>
<p>发布 Avito 双拍卖格式（VCG/FP）真实自动出价数据集，配套 ALM/TA-PID 等 5 个基线，覆盖预算平滑与 CPC 约束两类 RTB 任务。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/rl_bandit_rec/interactive_visualization_recommendation_with_hier_sucb/">Interactive Visualization Recommendation with Hier-SUCB</a></strong></dt>
<dd>
<p>将可视化推荐建模为带可学习偏置项的分层上下文组合半臂老虎机，把遗憾界从 $O(\sqrt{T\ln^3(nm^2T\ln T)})$降至$O(\sqrt{T\ln^3(m^2T\ln T)})$，在 Plot.ly 上 80/160 轮超越离线 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/rl_bandit_rec/reward_balancing_revisited_enhancing_offline_reinforcement_learning_for_rec/">Reward Balancing Revisited: Enhancing Offline Reinforcement Learning for Recommender Systems</a></strong></dt>
<dd>
<p>用扩散世界模型显式建模奖励不确定性，并以衰减加权的交互/熵双惩罚平衡多样性与长期收益，在三个真实数据集上刷新离线RL推荐SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/rtbagent_a_llm_based_agent_system_for_real_time_bidding/">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></strong></dt>
<dd>
<p>首个 LLM Agent 出价系统，用两步决策+多类记忆+专家策略工具在 iPinYou 上以调整因子方式超越规则/RL/扩散基线。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/rl_bandit_rec/value_function_decomposition_in_markov_recommendation_process/">Value Function Decomposition in Markov Recommendation Process</a></strong></dt>
<dd>
<p>将标准 TD 损失分解为策略随机与用户环境随机两个子目标，实现更快收敛、更强探索鲁棒性的长期价值估计。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multi_task_reranking/xmtf_a_formula_free_model_for_reinforcement_learning_based_multi_task_fusio/">xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems</a></strong></dt>
<dd>
<p>用可学习单调融合单元替代预定义融合公式，配合两阶段混合训练，离线总观看时长提升约7%，线上日观看时长+0.833%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/rl_bandit_rec/ahbid_an_adaptable_hierarchical_bidding_framework_for_cross_channel_adverti/">AHBid: An Adaptable Hierarchical Bidding Framework for Cross-Channel Advertising</a></strong></dt>
<dd>
<p>用扩散模型生成跨渠道预算/约束目标轨迹、结合约束执行与实时控制出价，离线与线上 A/B 收益提升 13.57%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/rl_bandit_rec/dara_few_shot_budget_allocation_in_online_advertising_via_in_context_decisi/">DARA: Few-shot Budget Allocation in Online Advertising via In-Context Decision Making with RL-Finetuned LLMs</a></strong></dt>
<dd>
<p>双阶段 LLM 智能体分离少样本规划与精细优化，配合动态参考策略 GRPO-Adaptive 微调，在真实与合成广告环境中显著降低边际 ROI 方差。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/rl_bandit_rec/off_policy_learning_with_limited_supply/">Off-Policy Learning with Limited Supply</a></strong></dt>
<dd>
<p>首次形式化库存受限下的离线策略学习，提出按“相对奖励差”分配稀缺物品的 OPLS，在合成与 KuaiRec 上稳定超越贪心 OPL。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/taosr_agrl_adaptive_guided_reinforcement_learning_framework_for_e_commerce_/">TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance</a></strong></dt>
<dd>
<p>用规则感知稠密奖励塑形与自适应引导回放缓解 GRPO 奖励稀疏，淘宝搜索相关性 Macro-F1 提升 2.28pt 并上线。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/unveiling_and_simulating_short_video_addiction_behaviors_via_economic_addic/">Unveiling and Simulating Short-Video Addiction Behaviors via Economic Addiction Theory</a></strong></dt>
<dd>
<p>用经济学成瘾理论量化短视频成瘾曲线，并以 M2A+GRPO 训练 LLM 用户模拟器 AddictSim，会话级 MAE 较 PPO 降 11%。</p>
</dd>
</dl>

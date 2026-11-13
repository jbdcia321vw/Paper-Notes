# 🎮 强化学习与动态探索推荐 (RL, Bandits & Long-Term Engagement)

> **研究范畴**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

当前方向共收录 **35** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2026/rl_bandit_rec/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augment/">Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback</a></strong></dt>
<dd>
<p>用增强置信界把相似/依赖观测注入无模型 Dueling Bandit，给出样本复杂度与遗憾权衡，在推荐、多目标优化与 LLM 响应优化上显著降遗憾。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/rl_bandit_rec/world_model_guided_reinforcement_learning_via_counterfactual_user_engagemen/">World Model-Guided Reinforcement Learning via Counterfactual User Engagement Simulation</a></strong></dt>
<dd>
<p>将推荐视为动作、异构反馈视为观测，用语言世界模型 UEWM 做反事实用户模拟器，为 1.7B 策略提供同状态稠密奖励，跨域零样本超越大模型。</p>
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

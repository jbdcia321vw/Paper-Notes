# 🎮 强化学习与动态探索推荐 (RL, Bandits & Long-Term Engagement)

> **研究范畴**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

当前方向共收录 **12** 篇顶会/顶刊论文解读：

<dl>
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
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/agentltv_an_agent_based_unified_search_and_evolution_framework_for_automate/">AgentLTV: An Agent-Based Unified Search-and-Evolution Framework for Automated Lifetime Value Prediction</a></strong></dt>
<dd>
<p>将 LTV 建模视为可执行流水线程序的搜索问题，用 LLM 多智能体驱动 MCTS 广探索 + 岛屿进化精炼，工业与公开数据上 ER 降 41%、Norm GINI 升 15%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/hierarchical_residual_policy_optimization_for_generative_recommendations/">Hierarchical Residual Policy Optimization for Generative Recommendations</a></strong></dt>
<dd>
<p>将物品级反馈分解为 SID 前缀残差信用，用 RRPO 做保守 token 级策略优化，KuaiSim 会话回报 10.53 vs 8.14，线上 IAA Target Cost 最高 +3.49%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/hoba_hierarchical_on_policy_bidding_agents_for_adaptive_online_advertising/">HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising</a></strong></dt>
<dd>
<p>三层时间尺度解耦出价：LLM 小时级推理超参、SARSA 分钟级因果去偏选专家、专家池逐次拍卖执行，AuctionNet 与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/uncertainty_calibrated_recommendations_for_low_active_users/">Uncertainty-Aware Adaptive Recommendation across User Lifecycle</a></strong></dt>
<dd>
<p>用 EPE Critic 与经验贝叶斯先验方差量化逐样本认知不确定性，对低活跃用户风险规避降权、高活跃用户 UCB 探索加成，14 天线上 A/B 验证留存与多样性双收益。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong></dt>
<dd>
<p>提出 EASQ，用 LoRA 独立参数通路 + MoE 解耦多任务 + 在线 DPO，将极稀疏问卷满意度信号端到端对齐进短视频精排在线学习。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/rl_bandit_rec/ahbid_an_adaptable_hierarchical_bidding_framework_for_cross_channel_adverti/">AHBid: An Adaptable Hierarchical Bidding Framework for Cross-Channel Advertising</a></strong></dt>
<dd>
<p>用扩散模型生成跨渠道预算/约束目标轨迹、结合约束执行与实时控制出价，离线与线上 A/B 收益提升 13.57%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/rl_bandit_rec/off_policy_learning_with_limited_supply/">Off-Policy Learning with Limited Supply</a></strong></dt>
<dd>
<p>首次形式化库存受限下的离线策略学习，提出按“相对奖励差”分配稀缺物品的 OPLS，在合成与 KuaiRec 上稳定超越贪心 OPL。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/unveiling_and_simulating_short_video_addiction_behaviors_via_economic_addic/">Unveiling and Simulating Short-Video Addiction Behaviors via Economic Addiction Theory</a></strong></dt>
<dd>
<p>用经济学成瘾理论量化短视频成瘾曲线，并以 M2A+GRPO 训练 LLM 用户模拟器 AddictSim，会话级 MAE 较 PPO 降 11%。</p>
</dd>
</dl>

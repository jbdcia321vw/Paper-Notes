# 🎮 强化学习与动态探索推荐

**ICLR / NeurIPS / ICML (RecSys) 2026** · **4** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/rl_bandit_rec/)

🔥 **高频主题：** ICML ×2 · 重排 ×2

<dl>
<dt><strong><a href="beyond_match_maximization_and_fairness_retention_optimized_two_sided_matchi/">Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching</a></strong></dt>
<dd>
<p>将双边匹配推荐重构为留存最大化问题，用凹性 Jensen 下界把 NP-hard 排序松弛为 O(NlogN) 的 argsort 打分，在合成与真实约会平台数据上以约 70% 匹配量取得最高留存率。</p>
</dd>
<dt><strong><a href="incentivized_exploration_with_stochastic_covariates_a_two_stage_mechanism_d/">Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System</a></strong></dt>
<dd>
<p>面向在线随机用户协变量的激励相容探索，提出两阶段 RCB 算法（冷启动补贴 + 逆比例间隙采样），在满足 ε-DBIC 约束下达到 $\tilde{O}(\sqrt{KdT})$ 遗憾。</p>
</dd>
<dt><strong><a href="optimal_and_efficient_contextual_combinatorial_semi_bandits_with_general_fu/">Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation</a></strong></dt>
<dd>
<p>提出 SquareCB.Comb，用回归 oracle + 对数障碍边际优化实现通用函数逼近下 CCSB 的极小极大最优 regret，并在网页推荐与最短路任务上验证。</p>
</dd>
<dt><strong><a href="prorl_effective_reinforcement_learning_for_proactive_recommendation_via_rec/">ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation</a></strong></dt>
<dd>
<p>揭示主动推荐中策略梯度的长度捷径与高方差缺陷，提出逐步奖励中心化与位置特定优势估计，三数据集全面超越 SOTA。</p>
</dd>
</dl>

# 🛡️ 去偏公平、可信隐私与工业架构

**SIGIR 2026** · **13** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与工业架构 ×8 · 重排 ×4

<dl>
<dt><strong><a href="adaptive_autoguidance_for_item_side_fairness_in_diffusion_recommender_syste/">Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems</a></strong></dt>
<dd>
<p>用可学习逐步引导权重替代固定自引导，配合流行度正则，在扩散推荐中实现物品侧公平与精度近无损的平衡。</p>
</dd>
<dt><strong><a href="additive_control_variates_dominate_self_normalisation_in_off_policy_evaluat/">Additive Control Variates Dominate Self-Normalisation in Off-Policy Evaluation</a></strong></dt>
<dd>
<p>证明最优加性基线 β*-IPS 在 MSE 上渐近严格优于 SNIPS，并给出精确方差差解析式与排序场景 β*⊥⊥-IPM 扩展。</p>
</dd>
<dt><strong><a href="equity_vs_equality_optimizing_ranking_fairness_for_tailored_provider_needs/">Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs</a></strong></dt>
<dd>
<p>提出以增益为基的公平框架与 EquityRank 梯度重排算法，按提供者异质偏好分配曝光与销量收益，离线/在线仿真均取得更优效果-公平权衡。</p>
</dd>
<dt><strong><a href="erase_a_real_world_aligned_benchmark_for_unlearning_in_recommender_systems/">ERASE -- A Real-World Aligned Benchmark for Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>面向真实场景的推荐系统机器遗忘基准，覆盖 CF/SBR/NBR 三任务、七种遗忘算法、九数据集九模型，揭示近似遗忘的鲁棒性与效率短板。</p>
</dd>
<dt><strong><a href="following_the_eye_tracking_evidence_established_web_search_assumptions_fail/">Following the Eye-Tracking Evidence: Established Web-Search Assumptions Fail in Carousel Interfaces</a></strong></dt>
<dd>
<p>用 RecGaze 眼动数据系统证伪 F 型注意力、层级点击与标题导航等四条 Web 搜索假设，提出轮播界面特有的 L 型注视模式。</p>
</dd>
<dt><strong><a href="from_top_1_to_top_k_a_reproducibility_study_and_benchmarking_of_counterfact/">From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems</a></strong></dt>
<dd>
<p>统一复现11种反事实解释器，首次提出列表级Top-K评估协议与PN-R位置敏感指标，揭示显式格式下有效性-稀疏性权衡与图解释器可扩展性瓶颈。</p>
</dd>
<dt><strong><a href="kuailive_a_real_time_interactive_dataset_for_live_streaming_recommendation/">KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation</a></strong></dt>
<dd>
<p>首个含直播间生命周期与多行为实时交互的快手直播推荐数据集，构建 Top-K/CTR 基准并揭示主播长尾公平性挑战。</p>
</dd>
<dt><strong><a href="post_hoc_provider_fairness_adaptation_via_hierarchical_exposure_alignment/">Post-hoc Provider Fairness Adaptation via Hierarchical Exposure Alignment</a></strong></dt>
<dd>
<p>冻结基座+轻量公平适配器，用分层曝光对齐(HEFA)与可微NDCG联合优化，实现即插即用的提供方曝光公平。</p>
</dd>
<dt><strong><a href="recnexteval_a_reference_implementation_for_temporal_next_batch_recommendati/">RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation</a></strong></dt>
<dd>
<p>提出滑动时间窗 next-batch 评估框架，强制全局时间线防泄漏并支持增量学习，含 Python 库与 Web UI。</p>
</dd>
<dt><strong><a href="solaris_speculative_offloading_of_latent_based_representation_for_inference/">SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling</a></strong></dt>
<dd>
<p>借鉴投机解码思想，异步预计算基础模型用户-物品隐表征并分层补全覆盖，实现推理时知识蒸馏，Meta 广告线上收入 +0.67%。</p>
</dd>
<dt><strong><a href="splitlight_an_exploratory_toolkit_for_recommender_systems_datasets_and_spli/">SplitLight: An Exploratory Toolkit for Recommender Systems Datasets and Splits</a></strong></dt>
<dd>
<p>开源数据集与划分审计工具箱，量化时间泄漏、冷启动暴露、重复消费与分布漂移，揭示预处理决策可使 NDCG@10 变动 1.2%–60%。</p>
</dd>
<dt><strong><a href="the_attention_market_interpreting_online_fair_re_ranking_as_manifold_optimi/">The Attention Market: Interpreting Online Fair Re-ranking as Manifold Optimization under Walrasian Equilibrium</a></strong></dt>
<dd>
<p>将在线公平重排建模为注意力市场瓦尔拉斯均衡，用供需双侧梯度校正对齐排序流形，7 数据集 42 设置几乎全胜。</p>
</dd>
<dt><strong><a href="variance_reduction_for_heavy_tailed_monetization_metrics_in_ranking_experim/">Variance Reduction for Heavy-Tailed Monetization Metrics in Ranking Experiments via Post-Stratification</a></strong></dt>
<dd>
<p>用后分层+CUPED压制重尾GMV方差，等效置信度下节省约45%流量，已在ShareChat 40+线上实验落地。</p>
</dd>
</dl>

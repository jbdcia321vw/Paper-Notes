# 🛡️ 去偏公平、可信隐私与工业架构 (Debiasing, Fairness, Privacy & Industrial Systems)

> **研究范畴**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

当前方向共收录 **22** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/a_standardized_re_evaluation_of_conversational_recommender_systems_on_the_r/">A Standardized Re-evaluation of Conversational Recommender Systems on the ReDial Dataset</a></strong></dt>
<dd>
<p>在 ReDial 上标准化复现 7 个 CRS 方法，揭示 Recall@1 粒度鸿沟、约 50% 准确率来自重复捷径，且增益多源于 LLM 骨干而非架构创新。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/adaptive_autoguidance_for_item_side_fairness_in_diffusion_recommender_syste/">Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems</a></strong></dt>
<dd>
<p>用自适应自引导网络逐步学习引导权重，配合流行度正则化，在扩散推荐中低成本提升物品侧公平性。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/rl_bandit_rec/additive_control_variates_dominate_self_normalisation_in_off_policy_evaluat/">Additive Control Variates Dominate Self-Normalisation in Off-Policy Evaluation</a></strong></dt>
<dd>
<p>证明最优加性基线 β*-IPS 在 MSE 上渐近严格优于 SNIPS，并扩展至 IPM 排序场景的 β*⊥⊥-IPM。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/an_epistemic_position_based_click_model_from_interactions_to_epistemic_dist/">An Epistemic Position-Based Click Model: From Interactions to Epistemic Distributions of Relevance and Bias</a></strong></dt>
<dd>
<p>首个上下文认知 PBM：用 Beta 分布刻画相关性与位置偏置的认知不确定性，以自归一化+位置条件化梯度估计稳定训练，测试集对数似然显著优于点式 PBM。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/cheaper_is_better_a_discount_aware_network_for_conversion_rate_prediction_i/">Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System</a></strong></dt>
<dd>
<p>用 FFT 时频分解 + 用户/上下文双重去偏 + 折扣率回归辅助任务建模商品折扣率时序，离线 AUC +1.61%、线上 pCVR +3.63%、GMV +2.23%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/clax_fast_and_flexible_neural_click_models_in_jax/">CLAX: Fast and Flexible Neural Click Models in JAX</a></strong></dt>
<dd>
<p>用梯度优化替代 EM 训练经典点击模型，JAX 实现单卡 2 小时跑完 10 亿级 Baidu-ULTR 会话</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/equity_vs_equality_optimizing_ranking_fairness_for_tailored_provider_needs/">Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs</a></strong></dt>
<dd>
<p>提出面向供给方异质需求的权益公平框架与梯度式重排算法 EquityRank，在离线/在线仿真中取得更优效果-公平权衡。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/erase_a_real_world_aligned_benchmark_for_unlearning_in_recommender_systems/">ERASE -- A Real-World Aligned Benchmark for Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>面向推荐系统机器遗忘的真实对齐基准，覆盖CF/SBR/NBR三任务、七种遗忘算法、九数据集九模型，揭示近似遗忘的鲁棒性与效率短板。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/exploring_test_time_scaling_via_prediction_merging_on_large_scale_recommend/">Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出测试时缩放新维度：通过异构架构组合与同构随机初始化生成多样预测并在 logit 空间融合，在 Criteo/Avazu/KDD12 上稳定超越参数缩放。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/from_top_1_to_top_k_a_reproducibility_study_and_benchmarking_of_counterfact/">From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems</a></strong></dt>
<dd>
<p>统一复现11种反事实解释方法，提出item/list双层与隐式/显式双格式评测协议，揭示效果-稀疏性权衡与图解释器可扩展性瓶颈</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/kuailive_a_real_time_interactive_dataset_for_live_streaming_recommendation/">KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation</a></strong></dt>
<dd>
<p>首个含直播间生命周期、多行为与负反馈的实时直播推荐数据集，构建Top-K与CTR基准。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/post_hoc_provider_fairness_adaptation_via_hierarchical_exposure_alignment/">Post-hoc Provider Fairness Adaptation via Hierarchical Exposure Alignment</a></strong></dt>
<dd>
<p>冻结基座推荐器，用轻量公平适配器做加性分数修正，并以分层曝光对齐损失+可微NDCG实现组间/组内曝光公平的端到端后处理适配。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/prompt_unknown_promotion_attacks_against_llm_based_sequential_recommender_s/">Prompt-Unknown Promotion Attacks against LLM-based Sequential Recommender Systems</a></strong></dt>
<dd>
<p>黑盒LLM序列推荐下的提示蒸馏+双重投毒商品推广攻击，HR@50最高提升约3.9倍</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/r_f_inventory_a_large_scale_dataset_for_monotonic_inventory_estimation_in_r/">R&amp;F-Inventory: A Large-Scale Dataset for Monotonic Inventory Estimation in Reach and Frequency Advertising</a></strong></dt>
<dd>
<p>发布首个含预算-触达曲线结构的 R&amp;F 合约库存数据集，定义单点预测与多维单调曲线重建两任务及可复现基准。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/recnexteval_a_reference_implementation_for_temporal_next_batch_recommendati/">RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation</a></strong></dt>
<dd>
<p>提出滑动时间窗 next-batch 评估框架，强制全局时间线数据释放以消除数据穿越，支持增量学习与宏/微平均指标聚合。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/request_only_optimization_for_recommendation_systems/">Request-Only Optimization for Recommendation Systems</a></strong></dt>
<dd>
<p>以用户请求为训练样本单元，通过数据/系统/模型协同设计消除请求内特征冗余，训练吞吐提升最高570%，同等算力下模型FLOPs扩展7倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/res_mr_risk_aware_reasoning_for_explainable_and_safe_medication_recommendat/">RES-MR: Risk-Aware Reasoning for Explainable and Safe Medication Recommendation</a></strong></dt>
<dd>
<p>两阶段 LLM 框架：KG 检索增强推理蒸馏 + 双因子解耦原型驱动的风险感知 RL 策略优化，实现可解释且个性化安全的用药推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom 索引 + Int8 融合核，吞吐提升 23.7×、成本效率 13.35×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/solaris_speculative_offloading_of_latent_based_representation_for_inference/">SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling</a></strong></dt>
<dd>
<p>借鉴投机解码思想，异步预计算并缓存基础模型用户-物品交互表征，实现推理时知识蒸馏，Meta 广告线上收入 +0.67%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/splitlight_an_exploratory_toolkit_for_recommender_systems_datasets_and_spli/">SplitLight: An Exploratory Toolkit for Recommender Systems Datasets and Splits</a></strong></dt>
<dd>
<p>开源数据审计工具，量化过滤/重复/冷启动/切分策略对离线评估的影响，诊断时间泄露与分布漂移。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/the_attention_market_interpreting_online_fair_re_ranking_as_manifold_optimi/">The Attention Market: Interpreting Online Fair Re-ranking as Manifold Optimization under Walrasian Equilibrium</a></strong></dt>
<dd>
<p>将在线公平重排建模为注意力市场瓦尔拉斯均衡，用流形优化统一供需两侧梯度调整，兼顾公平与精度。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong></dt>
<dd>
<p>提出后验价值对齐+独立线性提升的统一流量分配框架，解耦多业务加权方案，线上 A/B 提升 VV +1.69%、Valued VV +3.07%。</p>
</dd>
</dl>

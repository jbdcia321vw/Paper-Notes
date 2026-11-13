# ⚖️ 多任务学习与列表级重排 (Multi-Task Learning & Listwise Re-ranking)

> **研究范畴**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

当前方向共收录 **20** 篇顶会/顶刊论文解读：

<dl>
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
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/sequential_multimodal_evidence_optimization_for_product_media_ranking_in_e_/">Sequential Multimodal Evidence Optimization for Product Media Ranking in E-Commerce</a></strong></dt>
<dd>
<p>将商品媒体排序建模为证据累积序列决策，两阶段解耦效用学习与生存加权自回归策略，离线 DR-CVR +5.4%、MSC −15.3%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/task_blind_no_more_multi_task_information_flow_in_unified_ranking_backbones/">Task-Blind No MORE: Multi-Task Information Flow in Unified Ranking Backbones</a></strong></dt>
<dd>
<p>将多任务信息流下沉至统一排序主干，以共享/私有 Anchor Token 逐层共演化，离线七任务全胜、线上时长+3.0%、P99 延迟降约30%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/degre_dense_supervised_generative_reranking_for_recommendation/">DeGRe: Dense-supervised Generative Reranking for Recommendation</a></strong></dt>
<dd>
<p>离线束搜索挖掘前瞻序列+累积回归稠密监督蒸馏至轻量生成器，单次贪心解码逼近全局最优，线上GMV+3.75%</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/hmaf_a_hierarchical_multi_slot_gd_rtb_allocation_framework/">HMAF: A Hierarchical Multi-Slot GD-RTB Allocation Framework</a></strong></dt>
<dd>
<p>以 Plan-Calibrate-Execute 范式将 PV 约束对偶优化与在线列表级生成-评估排序耦合，美团线上 GD 交付率 +3.72%、广告总收入 +1.59%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/mac_a_conversion_rate_prediction_benchmark_featuring_labels_under_multiple_/">MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms</a></strong></dt>
<dd>
<p>首个多归因机制CVR公开基准MAC与PyMAL库，提出MoAE模型，GAUC最高提升0.39pt。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/onerank_unified_transformer_native_ranking_architecture_for_multi_task_reco/">OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation</a></strong></dt>
<dd>
<p>将多任务推理内化进 Transformer 栈，用任务私有 token 通道 + 梯度截断 + 匹配式打分替代编码器-预测器分离，Shopee 离线与线上 A/B 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/personalizing_marketplace_policies_with_competing_objectives_and_constraine/">Personalizing Marketplace Policies with Competing Objectives and Constrained Experiments: Evidence from a Job Marketplace</a></strong></dt>
<dd>
<p>双边市场免费额度阈值个性化：集成CATE+混合排序+线性外推，护栏风险降10%+，线上验证外推精度r=0.85</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/counterfactual_multi_task_learning_for_delayed_conversion_modeling_in_e_com/">Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion</a></strong></dt>
<dd>
<p>面向大促预热期延迟转化，提出多任务+个性化门控迁移+反事实因果正则的 CM-DCM，离线 AUC 与线上 GMV 双收益。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/equity_vs_equality_optimizing_ranking_fairness_for_tailored_provider_needs/">Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs</a></strong></dt>
<dd>
<p>提出面向供给方异质需求的权益公平框架与梯度式重排算法 EquityRank，在离线/在线仿真中取得更优效果-公平权衡。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/post_hoc_provider_fairness_adaptation_via_hierarchical_exposure_alignment/">Post-hoc Provider Fairness Adaptation via Hierarchical Exposure Alignment</a></strong></dt>
<dd>
<p>冻结基座推荐器，用轻量公平适配器做加性分数修正，并以分层曝光对齐损失+可微NDCG实现组间/组内曝光公平的端到端后处理适配。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/the_attention_market_interpreting_online_fair_re_ranking_as_manifold_optimi/">The Attention Market: Interpreting Online Fair Re-ranking as Manifold Optimization under Walrasian Equilibrium</a></strong></dt>
<dd>
<p>将在线公平重排建模为注意力市场瓦尔拉斯均衡，用流形优化统一供需两侧梯度调整，兼顾公平与精度。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/think_when_needed_model_aware_reasoning_routing_for_llm_based_ranking/">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></strong></dt>
<dd>
<p>提出轻量即插即用路由器，基于排序感知特征与模型感知难度信号，逐实例决定是否触发推理，实现精度-效率帕累托最优。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong></dt>
<dd>
<p>提出 EASQ，用 LoRA 独立参数通路 + MoE 解耦多任务 + 在线 DPO，将极稀疏问卷满意度信号端到端对齐进短视频精排在线学习。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/robustness_analysis_of_pareto_based_joint_evaluation_of_fairness_and_releva/">Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems</a></strong></dt>
<dd>
<p>提出 DPFR：以测试集真实标签构建相关性-公平性帕累托前沿，用模型得分到前沿参考点的距离联合评估推荐系统，比现有联合度量更稳健。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/cross_domain_multi_scenario/automated_information_flow_selection_for_multi_scenario_multi_task_recommen/">Automated Information Flow Selection for Multi-scenario Multi-task Recommendation</a></strong></dt>
<dd>
<p>用 LoRA 解耦四类场景-任务信息单元并以可学习掩码剪枝无效信息流，AUC 全面超越 HiNet/M3oE，线上 CVR 均涨 2.46%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/multi_task_reranking/cold_starting_podcast_ads_and_promotions_with_multi_task_learning_on_spotif/">Cold-Starting Podcast Ads and Promotions with Multi-Task Learning on Spotify</a></strong></dt>
<dd>
<p>Spotify 用单一 5 任务 MTL 模型统一播客广告与推广定向，靠非对称损失掩码与源平衡采样实现跨渠道迁移，线上 eCPS 降 22%、i2s 升 18–24%。</p>
</dd>
</dl>

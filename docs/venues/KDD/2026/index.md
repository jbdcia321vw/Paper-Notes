---
hide:
  - toc
---

# 🏛️ KDD 2026 推荐系统论文解读

38 篇 KDD 2026 论文解读，覆盖 13 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl>
<dt><strong><a href="sequential_rec/journeyformer_encoding_airbnb_guest_journey_with_sequence_modeling/">JourneyFormer: Encoding Airbnb Guest Journey with Sequence Modeling</a></strong></dt>
<dd>
<p>Airbnb 搜索排序的工业级序列建模方案，以频次分层长短期序列+统一ID嵌入+因果Transformer编码访客旅程，离线NDCG+1.48%、线上Bookers+0.55%。</p>
</dd>
<dt><strong><a href="sequential_rec/mixformer_co_scaling_up_dense_and_sequence_in_industrial_recommenders/">MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders</a></strong></dt>
<dd>
<p>用单一参数空间统一序列建模与稠密特征交叉，配合用户-物品解耦实现请求级复用，AUC 提升 1.28% 且 FLOPs 降 36%。</p>
</dd>
<dt><strong><a href="sequential_rec/one_sequential_recommendation_model_pretrained_from_synthetic_priors_predic/">One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets</a></strong></dt>
<dd>
<p>用 hDCSBM 合成先验预训练 PFN，以支持集条件化实现免更新推理的序列推荐，五基准平均超最优基线 7.53%。</p>
</dd>
<dt><strong><a href="sequential_rec/unleashing_the_potential_of_neighbors_diffusion_based_latent_neighbor_gener/">Unleashing the Potential of Neighbors: Diffusion-based Latent Neighbor Generation for Session-based Recommendation</a></strong></dt>
<dd>
<p>用检索先验与多模态信号双路引导扩散模型生成“潜在邻居”，突破可观测邻居边界，Full-Ranking 下 P@20 最高提升 18.45%。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (8)</h2>
<dl>
<dt><strong><a href="generative_rec/beyond_interleaving_causal_attention_reformulations_for_generative_recommen/">Beyond Interleaving: Causal Attention Reformulations for Generative Recommender Systems</a></strong></dt>
<dd>
<p>揭示交错式生成推荐是相似度加权动作池化的低效代理，提出 AttnLFA/AttnMVP 显式编码 item→action 因果，序列长度减半、训练提速 23%/12% 且 NE 全面下降</p>
</dd>
<dt><strong><a href="generative_rec/collaborative_memory_augmentation_for_generative_recommendation/">Collaborative Memory Augmentation for Generative Recommendation</a></strong></dt>
<dd>
<p>为生成式推荐引入跨用户协同记忆库，用可学习查询token压缩序列、目标感知检索与门控交叉注意力融合，Full-Ranking下NDCG@5最高提升23.7%。</p>
</dd>
<dt><strong><a href="generative_rec/hi_sam_a_hierarchical_structure_aware_multi_modal_framework_for_large_scale/">Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>解耦语义分词器+分层记忆锚点Transformer，修复语义ID推荐中分词冗余与扁平架构失配，工业A/B响应率+6.55%</p>
</dd>
<dt><strong><a href="generative_rec/hierarchical_residual_policy_optimization_for_generative_recommendations/">Hierarchical Residual Policy Optimization for Generative Recommendations</a></strong></dt>
<dd>
<p>将物品级反馈分解为 SID 前缀残差信用，用 RRPO 做保守 token 级策略优化，KuaiSim 会话回报 10.53 vs 8.14，线上 IAA Target Cost 最高 +3.49%。</p>
</dd>
<dt><strong><a href="generative_rec/reasoning_over_semantic_ids_enhances_generative_recommendation/">Reasoning over Semantic IDs Enhances Generative Recommendation</a></strong></dt>
<dd>
<p>通过教师模型合成富化语料强化 SID-语言对齐，再用 GRPO 结果驱动强化学习激发 SID 推理，三数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="generative_rec/s_2_gr_stepwise_semantic_guided_reasoning_in_latent_space_for_generative_re/">S$^2$GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation</a></strong></dt>
<dd>
<p>在每级 SID 码生成前插入受码本聚类语义对比监督的思考 token，实现逐步可验证潜在推理，工业短视频 A/B 显著提升时长。</p>
</dd>
<dt><strong><a href="generative_rec/uncertainty_aware_generative_recommendation/">Uncertainty-aware Generative Recommendation</a></strong></dt>
<dd>
<p>将生成置信度与样本难度转化为GRPO奖励与梯度权重，并显式对齐置信度token，稳定训练并支持风险感知推荐。</p>
</dd>
<dt><strong><a href="generative_rec/vectorizing_the_trie_efficient_constrained_decoding_for_llm_based_generativ/">Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators</a></strong></dt>
<dd>
<p>将前缀树约束展平为静态 CSR 稀疏转移矩阵，把不规则树遍历转为全向量化稀疏算子，实现 O(1) I/O 的严格约束解码，线上延迟仅 0.033ms/步。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (3)</h2>
<dl>
<dt><strong><a href="llm_rec/enriching_semantic_profiles_into_knowledge_graph_for_recommender_systems_us/">Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 为 KG 全实体生成语义画像，以可逆加性注入+成对匹配对齐，Full-Ranking 下 Recall@40 最高提升 2.6%。</p>
</dd>
<dt><strong><a href="llm_rec/improving_item_discoverability_in_e_commerce_search_via_related_intent_gene/">Improving Item Discoverability in e-Commerce Search via Related Intent Generation</a></strong></dt>
<dd>
<p>用闭源LLM+蒸馏SLM生成替代/互补/主题三类隐式意图以扩展电商搜索召回，会话购买预测与LLM评审双轨评估，覆盖流量60%→80%、成本仅30%。</p>
</dd>
<dt><strong><a href="llm_rec/mixture_of_experts_knowledge_graph_retrieval_augmented_generation_for_multi/">Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation</a></strong></dt>
<dd>
<p>用 MoE 多粒度 KG 检索智能体按查询复杂度路由检索粒度，配合知识对齐与对比偏好优化，在 20 候选池协议下 Accuracy 相对提升 5.8%–20.4%。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/hierarchical_long_term_semantic_memory_for_linkedin_s_hiring_agent/">Hierarchical Long-Term Semantic Memory for LinkedIn&#x27;s Hiring Agent</a></strong></dt>
<dd>
<p>HLTM 用 schema 对齐的层级记忆树 + 多视图表示 + 无损增量索引，为 LinkedIn 招聘智能体实现低延迟、隐私隔离的长期个性化记忆，正确率提升 5%+、检索 F1 提升 10%+。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/safecrs_personalized_safety_alignment_for_llm_based_conversational_recommen/">SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出 SafeRec 个性化安全基准与 Safe-SFT+Safe-GDPO 两阶段框架，在 SafeMovie 上将安全违规率相对降低 96.5% 且保持竞争力推荐质量。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/text_guided_visual_representation_learning_for_robust_multimodal_e_commerce/">Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation</a></strong></dt>
<dd>
<p>以结构化元数据初始化语义查询、双流解耦视觉证据并用可靠性双门控校准，在 134K 全库 I2I 检索上 H@100 相对提升 6.04%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl>
<dt><strong><a href="cold_start_rec/denoising_implicit_feedback_for_cold_start_recommendation/">Denoising Implicit Feedback for Cold-start Recommendation</a></strong></dt>
<dd>
<p>用内容相似热物品协同表征为冷物品生成置信加权伪标签，并以相对熵×冷启动状态估计样本不确定性做自适应软标签纠正，快手十亿级 A/B 全指标提升。</p>
</dd>
<dt><strong><a href="cold_start_rec/pinequalizer_full_funnel_content_exploration_and_debiasing_system_at_pinter/">PinEqualizer: Full Funnel Content Exploration and Debiasing System at Pinterest</a></strong></dt>
<dd>
<p>面向工业级搜索/推荐全漏斗的新内容冷启动系统，以去偏替代纯探索，配合三层可扩展度量框架，线上新鲜内容曝光提升350%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/causalpoi_spatio_temporal_graph_based_causal_modeling_for_cold_start_poi_ch/">CausalPOI: Spatio-Temporal Graph-Based Causal Modeling for Cold-Start POI Check-in Forecasting</a></strong></dt>
<dd>
<p>用功能交互图+结构对齐的干预/对照图做冷启动POI签到量预测与因果提升估计，四区域RMSE最高降57.8%。</p>
</dd>
<dt><strong><a href="poi_spatiotemporal_rec/think2go_generative_next_poi_recommendation_with_llm_reasoning/">Think2Go: Generative Next POI Recommendation with LLM Reasoning</a></strong></dt>
<dd>
<p>统一SFT与RL推理的生成式Next-POI框架，用KDE时空认知不确定性与难度感知奖励差校准优势函数，Acc@1较GNPR提升5.8–7.5%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/expand_more_shrink_less_shaping_effective_rank_dynamics_for_dense_scaling_i/">Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation</a></strong></dt>
<dd>
<p>从有效秩视角诊断 RankMixer 的阻尼振荡塌缩，提出参数化全混合与 GLU-PFFN 实现“多扩张少收缩”，Criteo/Avazu 上 AUC 提升超 0.001。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/hmaf_a_hierarchical_multi_slot_gd_rtb_allocation_framework/">HMAF: A Hierarchical Multi-Slot GD-RTB Allocation Framework</a></strong></dt>
<dd>
<p>以 Plan-Calibrate-Execute 范式将 PV 约束对偶优化与在线列表级生成-评估排序耦合，美团线上 GD 交付率 +3.72%、广告总收入 +1.59%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/mac_a_conversion_rate_prediction_benchmark_featuring_labels_under_multiple_/">MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms</a></strong></dt>
<dd>
<p>首个多归因机制CVR公开基准MAC与PyMAL库，提出MoAE模型，GAUC最高提升0.39pt。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (3)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/degre_dense_supervised_generative_reranking_for_recommendation/">DeGRe: Dense-supervised Generative Reranking for Recommendation</a></strong></dt>
<dd>
<p>离线束搜索挖掘前瞻序列+累积回归稠密监督蒸馏至轻量生成器，单次贪心解码逼近全局最优，线上GMV+3.75%</p>
</dd>
<dt><strong><a href="multi_task_reranking/onerank_unified_transformer_native_ranking_architecture_for_multi_task_reco/">OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation</a></strong></dt>
<dd>
<p>将多任务推理内化进 Transformer 栈，用任务私有 token 通道 + 梯度截断 + 匹配式打分替代编码器-预测器分离，Shopee 离线与线上 A/B 全面领先。</p>
</dd>
<dt><strong><a href="multi_task_reranking/personalizing_marketplace_policies_with_competing_objectives_and_constraine/">Personalizing Marketplace Policies with Competing Objectives and Constrained Experiments: Evidence from a Job Marketplace</a></strong></dt>
<dd>
<p>双边市场免费额度阈值个性化：集成CATE+混合排序+线性外推，护栏风险降10%+，线上验证外推精度r=0.85</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/agentltv_an_agent_based_unified_search_and_evolution_framework_for_automate/">AgentLTV: An Agent-Based Unified Search-and-Evolution Framework for Automated Lifetime Value Prediction</a></strong></dt>
<dd>
<p>将 LTV 建模视为可执行流水线程序的搜索问题，用 LLM 多智能体驱动 MCTS 广探索 + 岛屿进化精炼，工业与公开数据上 ER 降 41%、Norm GINI 升 15%。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/hoba_hierarchical_on_policy_bidding_agents_for_adaptive_online_advertising/">HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising</a></strong></dt>
<dd>
<p>三层时间尺度解耦出价：LLM 小时级推理超参、SARSA 分钟级因果去偏选专家、专家池逐次拍卖执行，AuctionNet 与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/uncertainty_calibrated_recommendations_for_low_active_users/">Uncertainty-Aware Adaptive Recommendation across User Lifecycle</a></strong></dt>
<dd>
<p>用 EPE Critic 与经验贝叶斯先验方差量化逐样本认知不确定性，对低活跃用户风险规避降权、高活跃用户 UCB 探索加成，14 天线上 A/B 验证留存与多样性双收益。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (5)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/canniuplift_a_holistic_framework_for_mitigating_seller_and_incentive_cannib/">CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling</a></strong></dt>
<dd>
<p>用平台级GMV一致性约束与核销路径分解，缓解多卖家电商uplift建模中的卖家级与激励级蚕食，线上ΔGMV相对提升4.08%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/isotonic_layer_a_unified_framework_for_recommendation_calibration_and_debia/">Isotonic Layer: A Unified Framework for Recommendation Calibration and Debiasing</a></strong></dt>
<dd>
<p>将非负桶权重参数化为上下文嵌入，用单个可微分段线性层统一校准与位置去偏，LinkedIn 线上 A/B 显著提升。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/llm_as_a_judge_for_reliable_and_explainable_offline_evaluation_in_top_k_rec/">LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation</a></strong></dt>
<dd>
<p>用 LLM 从用户文本行为构建语义偏好代理替代 holdout ID 匹配，以 reasoning-then-scoring 生成可解释 Top-K 评测，模型排序一致性较常规测试提升 40–150%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/rethinking_popularity_bias_in_collaborative_filtering_via_analytical_vector/">Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition</a></strong></dt>
<dd>
<p>揭示 BPR 优化在嵌入空间刻出“流行度方向”的几何偏差，提出非对称方向分解校正 DDC，在 5 个骨干上 MRR@10 提升 6%–29% 且 AvgPop@10 降 21%–47%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/the_pitfall_of_scaling_up_uncovering_and_mitigating_popularity_bias_amplifi/">The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders</a></strong></dt>
<dd>
<p>揭示 Transformer 序列推荐随深度加深发生谱坍缩并放大流行度偏差，提出逐层注意力列和与 FFN 谱范数正则 SPRINT，在 6 数据集上同时提升准确率与长尾公平性。</p>
</dd>
</dl>
</div>

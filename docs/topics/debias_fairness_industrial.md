# 🛡️ 去偏公平、可信隐私与工业架构 (Debiasing, Fairness, Privacy & Industrial Systems)

> **研究范畴**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

当前方向共收录 **77** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/CIKM/2026/graph_rec/beyond_observed_auxiliary_relations_environment_conditioned_modeling_for_mu/">Beyond Observed Auxiliary Relations: Environment-Conditioned Modeling for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>以辅助行为可观测性为环境条件，用去偏稠密化与目标引导剪枝双模块重连多行为图，隐藏目标物品 HR@10 最高提升 44.2%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/bidding_aware_retrieval_for_multi_stage_consistency_in_online_advertising/">Bidding-Aware Retrieval for Multi-Stage Consistency in Online Advertising</a></strong></dt>
<dd>
<p>将竞价信号注入召回打分，用单调性约束+多任务蒸馏+任务注意力精炼实现召回-精排 eCPM 一致性，线上收入 +4.32%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/debias_fairness_industrial/echotrace_diagnosing_recursive_risks_in_llm_powered_recommender_systems/">EchoTrace: Diagnosing Recursive Risks in LLM-Powered Recommender Systems</a></strong></dt>
<dd>
<p>提出角色感知三阶段诊断框架，用受控反馈回路仿真追踪 LLM 偏见与幻觉在 LLM4RS 中的递归放大与生态极化。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/jointly_optimizing_debiased_ctr_and_uplift_for_coupons_marketing_a_unified_/">Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing: A Unified Causal Framework</a></strong></dt>
<dd>
<p>UniMVT 用 MoE 解耦混淆表征 + logit 空间单调线性假设，统一估计去偏 Base CTR 与多值券面额 Uplift，线上收入 +18.14%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/debias_fairness_industrial/llm_based_user_product_relevance_evaluation_for_dynamic_product_advertising/">LLM-based User-Product Relevance Evaluation for Dynamic Product Advertising</a></strong></dt>
<dd>
<p>用 SID 分组意图压缩 + 智能体指南蒸馏小模型，构建无查询的 DPA 召回离线相关性指标，与线上 CP/CR/CVR 强对齐。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/sidinspector_a_mapping_first_diagnostic_resource_for_semantic_id_tokenizers/">SIDInspector: A Mapping-First Diagnostic Resource for Semantic-ID Tokenizers</a></strong></dt>
<dd>
<p>提出映射优先的 SID 分词器诊断资源，用 D1–D5 探针在生成器训练前分离容量、可寻址性与行为前缀对齐，并给出四类分词器工件画像。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/the_recall_ceiling_of_llm_recommendation_reranking/">The Recall Ceiling of LLM Recommendation Reranking</a></strong></dt>
<dd>
<p>证明闭候选重排器 NDCG 受自身窗口召回率硬上界约束，Oracle 协议高估 92–95%，九类 LLM 与监督重排器在真实检索下均无法显著超越 CF。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/canniuplift_a_holistic_framework_for_mitigating_seller_and_incentive_cannib/">CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling</a></strong></dt>
<dd>
<p>用平台级GMV一致性约束与核销路径分解，缓解多卖家电商uplift建模中的卖家级与激励级蚕食，线上ΔGMV相对提升4.08%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/expand_more_shrink_less_shaping_effective_rank_dynamics_for_dense_scaling_i/">Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation</a></strong></dt>
<dd>
<p>从有效秩视角诊断 RankMixer 的阻尼振荡塌缩，提出参数化全混合与 GLU-PFFN 实现“多扩张少收缩”，Criteo/Avazu 上 AUC 提升超 0.001。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/isotonic_layer_a_unified_framework_for_recommendation_calibration_and_debia/">Isotonic Layer: A Unified Framework for Recommendation Calibration and Debiasing</a></strong></dt>
<dd>
<p>将非负桶权重参数化为上下文嵌入，用单个可微分段线性层统一校准与位置去偏，LinkedIn 线上 A/B 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/sequential_rec/journeyformer_encoding_airbnb_guest_journey_with_sequence_modeling/">JourneyFormer: Encoding Airbnb Guest Journey with Sequence Modeling</a></strong></dt>
<dd>
<p>Airbnb 搜索排序的工业级序列建模方案，以频次分层长短期序列+统一ID嵌入+因果Transformer编码访客旅程，离线NDCG+1.48%、线上Bookers+0.55%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/llm_as_a_judge_for_reliable_and_explainable_offline_evaluation_in_top_k_rec/">LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation</a></strong></dt>
<dd>
<p>用 LLM 从用户文本行为构建语义偏好代理替代 holdout ID 匹配，以 reasoning-then-scoring 生成可解释 Top-K 评测，模型排序一致性较常规测试提升 40–150%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multi_task_reranking/personalizing_marketplace_policies_with_competing_objectives_and_constraine/">Personalizing Marketplace Policies with Competing Objectives and Constrained Experiments: Evidence from a Job Marketplace</a></strong></dt>
<dd>
<p>双边市场免费额度阈值个性化：集成CATE+混合排序+线性外推，护栏风险降10%+，线上验证外推精度r=0.85</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cold_start_rec/pinequalizer_full_funnel_content_exploration_and_debiasing_system_at_pinter/">PinEqualizer: Full Funnel Content Exploration and Debiasing System at Pinterest</a></strong></dt>
<dd>
<p>面向工业级搜索/推荐全漏斗的新内容冷启动系统，以去偏替代纯探索，配合三层可扩展度量框架，线上新鲜内容曝光提升350%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/rethinking_popularity_bias_in_collaborative_filtering_via_analytical_vector/">Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition</a></strong></dt>
<dd>
<p>揭示 BPR 优化在嵌入空间刻出“流行度方向”的几何偏差，提出非对称方向分解校正 DDC，在 5 个骨干上 MRR@10 提升 6%–29% 且 AvgPop@10 降 21%–47%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/agent_conversational_rec/safecrs_personalized_safety_alignment_for_llm_based_conversational_recommen/">SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出 SafeRec 个性化安全基准与 Safe-SFT+Safe-GDPO 两阶段框架，在 SafeMovie 上将安全违规率相对降低 96.5% 且保持竞争力推荐质量。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/the_pitfall_of_scaling_up_uncovering_and_mitigating_popularity_bias_amplifi/">The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders</a></strong></dt>
<dd>
<p>揭示 Transformer 序列推荐随深度加深发生谱坍缩并放大流行度偏差，提出逐层注意力列和与 FFN 谱范数正则 SPRINT，在 6 数据集上同时提升准确率与长尾公平性。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/vectorizing_the_trie_efficient_constrained_decoding_for_llm_based_generativ/">Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators</a></strong></dt>
<dd>
<p>将前缀树约束展平为静态 CSR 稀疏转移矩阵，把不规则树遍历转为全向量化稀疏算子，实现 O(1) I/O 的严格约束解码，线上延迟仅 0.033ms/步。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/rec_a_verifiable_benchmark_for_agentic_recommender_systems/">$τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems</a></strong></dt>
<dd>
<p>用可验证目录谓词奖励、reveal-tagged 信息揭示机制与 pass^k 可靠性指标替代 LLM-as-judge，揭示智能体推荐在隐藏约束下的可靠性悬崖。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/social_group_rec/are_we_really_making_progress_in_group_recommendation_unmasking_the_tie_bre/">Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion</a></strong></dt>
<dd>
<p>揭示群组推荐中训练期 sigmoid 分数压缩与评估期确定性破平交互导致的系统性评估偏差，提出并列感知 HR/NDCG 精确期望协议，使多个 SOTA 方法报告增益大幅缩水。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/decoupled_learning_and_selection_in_slate_recommendation_for_privacy_and_st/">Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores</a></strong></dt>
<dd>
<p>将列表推荐拆为随机打分器+确定性选择器，给出隐私后处理契约与可审计的 margin 稳定性证书，锚点加权使 Flip@K 降 53–81%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/sequential_rec/do_sequential_recommendation_benchmarks_really_require_higher_order_sequenc/">Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?</a></strong></dt>
<dd>
<p>用两个不学高阶序列表征的近期加权成对探针（SeqRules/PCTM）在 eSASRec 协议下反超 Transformer，质疑主流基准的容量测量效度。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/do_we_care_about_personalization_and_explainability_an_interview_study_with/">Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers</a></strong></dt>
<dd>
<p>对 9 家新闻机构 15 位工程师的半结构化访谈，揭示个性化采纳受编辑控制/隐私/资源约束，可解释性几乎只用于内部调试而非终端用户。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/lightweight_ranking_heads_accelerating_multi_task_experimentation_in_produc/">Lightweight Ranking Heads: Accelerating Multi-Task Experimentation in Production Recommender Systems</a></strong></dt>
<dd>
<p>以 stop-gradient + 无状态日训的轻量头动态注入新任务，免骨干重训，将多任务实验周期从数周压缩至数天。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/membership_inference_attacks_on_in_context_examples_in_llm_based_recommende/">Membership Inference Attacks on In-Context Examples in LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>首次针对 ICL 式 LLM 推荐系统提出 ItemMem 与 RecInertia 两种成员推理攻击，黑盒下攻击优势最高近 1.0，现有提示抽取防御基本失效。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/on_the_convergent_validity_of_offline_evaluation_designs_for_recommender_sy/">On the Convergent Validity of Offline Evaluation Designs for Recommender Systems</a></strong></dt>
<dd>
<p>以稠密真值反馈为参照，用 Kendall&#x27;s τ 检验稀疏离线评测设计的收敛效度，发现无普适最优设计且 KuaiRec 上排序近乎反转。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/cold_start_rec/predicting_custom_feed_returns_for_new_bluesky_posts_a_prospective_study/">Predicting Custom-Feed Returns for New Bluesky Posts: A Prospective Study</a></strong></dt>
<dd>
<p>将新帖作为查询、信息流作为候选的冷启动路由任务，构建 collect-first/label-later 基准，LambdaRank 达 cR@10 0.7361。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/probabilistic_residual_learning_for_online_recommendations/">Probabilistic Residual Learning for Online Recommendations</a></strong></dt>
<dd>
<p>以即插即用的层次贝叶斯残差模型对固定基座推荐器做概率用户聚类与 do-calculus 混杂去偏，冷启动跨域 Recall@20 最高提升约 7.6 倍。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/the_utility_of_llms_in_recommender_systems_explanation_evaluation/">The Utility of LLMs in Recommender Systems Explanation Evaluation</a></strong></dt>
<dd>
<p>系统评测 14 个 LLM 作为推荐解释评审员与 216 名人类评分的一致性，发现排序相关中等但绝对评分一致性低，并给出四条实用准则。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/training_seeds_and_model_selection_stability_in_recommender_system_evaluati/">Training seeds and model-selection stability in recommender-system evaluation</a></strong></dt>
<dd>
<p>固定数据划分与超参、仅变训练种子，从用户级指标敏感性、验证选模稳定性、验证-测试一致性与推荐列表一致性四层量化种子噪声，主张种子应纳入评估协议。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/watchlens_a_configurable_platform_for_online_video_recommendation_experimen/">WatchLens: A Configurable Platform for Online Video Recommendation Experiments</a></strong></dt>
<dd>
<p>开源视频推荐在线实验平台，解耦界面/内容池/双表面策略并原生记录曝光上下文，30人被试验证观看页策略对会话留存的影响。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/whose_posts_get_ranked_identical_text_exposure_gaps_in_bluesky_custom_feeds/">Whose Posts Get Ranked: Identical-Text Exposure Gaps in Bluesky Custom Feeds</a></strong></dt>
<dd>
<p>基于自然重复帖的曝光审计：同文本跨作者在250个Bluesky自定义Feed中曝光差异达33%，作者历史主导排名</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/a_standardized_re_evaluation_of_conversational_recommender_systems_on_the_r/">A Standardized Re-evaluation of Conversational Recommender Systems on the ReDial Dataset</a></strong></dt>
<dd>
<p>在 ReDial 上标准化复现 7 个 CRS 方法，揭示 Recall@1 粒度鸿沟、约 50% 准确率来自重复捷径，且增益多源于 LLM 骨干而非架构创新。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/adaptive_autoguidance_for_item_side_fairness_in_diffusion_recommender_syste/">Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems</a></strong></dt>
<dd>
<p>用可学习逐步引导权重替代固定自引导，配合流行度正则，在扩散推荐中实现物品侧公平与精度近无损的平衡。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/additive_control_variates_dominate_self_normalisation_in_off_policy_evaluat/">Additive Control Variates Dominate Self-Normalisation in Off-Policy Evaluation</a></strong></dt>
<dd>
<p>证明最优加性基线 β*-IPS 在 MSE 上渐近严格优于 SNIPS，并给出精确方差差解析式与排序场景 β*⊥⊥-IPM 扩展。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>揭示稠密MLP隐式权重稀疏现象，提出 filter-then-fuse 多视图显式稀疏框架 SSR，在十亿级工业数据与三个公开集上以更低算力超越 RankMixer。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/bridging_passive_and_active_enhancing_conversation_starter_recommendation_v/">Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling</a></strong></dt>
<dd>
<p>用对抗分布对齐+RQ-VAE语义离散化把用户主动查询引入对话开场白推荐，打破曝光-点击回音室，线上活跃天数+0.04%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/cheaper_is_better_a_discount_aware_network_for_conversion_rate_prediction_i/">Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System</a></strong></dt>
<dd>
<p>用 FFT 时频分解 + 用户/上下文双重去偏 + 折扣率回归辅助任务建模商品折扣率时序，离线 AUC +1.61%、线上 pCVR +3.63%、GMV +2.23%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/conv_finre_a_conversational_and_longitudinal_benchmark_for_utility_grounded/">Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation</a></strong></dt>
<dd>
<p>用逆优化从纵向决策轨迹反推用户风险偏好，构建四视图多视角金融推荐基准，揭示 LLM 效用对齐与行为模仿的根本张力。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong></dt>
<dd>
<p>以塔内自校正、跨塔EMA同步与级联模型共享三机制增强双塔召回，离线三数据集AUC一致提升，线上广告收入+8.36%且QPS仅降0.59%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/equity_vs_equality_optimizing_ranking_fairness_for_tailored_provider_needs/">Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs</a></strong></dt>
<dd>
<p>提出面向供给方异质需求的权益公平框架与梯度式重排算法 EquityRank，在离线/在线仿真中取得更优效果-公平权衡。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/erase_a_real_world_aligned_benchmark_for_unlearning_in_recommender_systems/">ERASE -- A Real-World Aligned Benchmark for Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>面向推荐系统机器遗忘的真实对齐基准，覆盖CF/SBR/NBR三任务、七种遗忘算法、九数据集九模型，揭示近似遗忘的鲁棒性与效率短板。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/fedmm_federated_collaborative_signal_quantization_for_multi_market_ctr_pred/">FedMM: Federated Collaborative Signal Quantization for Multi-Market CTR Prediction</a></strong></dt>
<dd>
<p>用双层 RQ-VAE 码本作联邦传输介质，对齐跨市场不相交 ID 空间，AUC 全面超越本地/联邦基线且通信量降 20×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/follow_the_trace_exploiting_post_click_trajectories_for_online_delayed_conv/">Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将点击后行为演化建模为反馈轨迹，用贝叶斯分解+可靠性门控回溯补全动态修正未转化样本后验，Criteo AUC 0.8382 超越 DEFUSE。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/from_top_1_to_top_k_a_reproducibility_study_and_benchmarking_of_counterfact/">From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems</a></strong></dt>
<dd>
<p>统一复现11种反事实解释方法，提出item/list双层与隐式/显式双格式评测协议，揭示效果-稀疏性权衡与图解释器可扩展性瓶颈</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/kuailive_a_real_time_interactive_dataset_for_live_streaming_recommendation/">KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation</a></strong></dt>
<dd>
<p>首个含直播间生命周期与多行为实时交互的快手直播推荐数据集，构建 Top-K/CTR 基准并揭示主播长尾公平性挑战。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致模型评估失真，系统分析SSS生效条件与数据分布重塑机理。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/post_hoc_provider_fairness_adaptation_via_hierarchical_exposure_alignment/">Post-hoc Provider Fairness Adaptation via Hierarchical Exposure Alignment</a></strong></dt>
<dd>
<p>冻结基座推荐器，用轻量公平适配器做加性分数修正，并以分层曝光对齐损失+可微NDCG实现组间/组内曝光公平的端到端后处理适配。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/prompt_unknown_promotion_attacks_against_llm_based_sequential_recommender_s/">Prompt-Unknown Promotion Attacks against LLM-based Sequential Recommender Systems</a></strong></dt>
<dd>
<p>在系统提示词与受害模型均不可见的全黑盒设定下，用进化式提示蒸馏+代理模型+文本与行为双投毒实现长尾商品推广攻击，Beauty 上 HR@10 提升约 3.9 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/r_f_inventory_a_large_scale_dataset_for_monotonic_inventory_estimation_in_r/">R&amp;F-Inventory: A Large-Scale Dataset for Monotonic Inventory Estimation in Reach and Frequency Advertising</a></strong></dt>
<dd>
<p>首个公开的 R&amp;F 合约库存预估数据集，以「定向-排期-频控」为上下文提供完整预算-效果曲线，定义单点预测与多维单调曲线重建两大基准任务。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/recnexteval_a_reference_implementation_for_temporal_next_batch_recommendati/">RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation</a></strong></dt>
<dd>
<p>提出滑动时间窗 next-batch 评估框架，强制全局时间线防泄漏并支持增量学习，含 Python 库与 Web UI。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom 索引 + Int8 融合核，吞吐提升 23.7×、成本效率 13.35×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/solaris_speculative_offloading_of_latent_based_representation_for_inference/">SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling</a></strong></dt>
<dd>
<p>借鉴投机解码思想，异步预计算基础模型用户-物品隐表征并分层补全覆盖，实现推理时知识蒸馏，Meta 广告线上收入 +0.67%。</p>
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
<p>用后验价值对齐把抽象混合分映射到有效完播率语义空间，再以独立线性 boosting 解耦多业务加权，线上 A/B 全指标正向。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动为相关项内偏好信号，融合交叉编码器分级标签、多通道检索先验与去偏互动构建统一双塔训练目标，离线 P@25 +10.5%、NDCG@25 +5.7%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/debias_fairness_industrial/mitigating_popularity_bias_in_recommendation_with_global_listwise_learning_/">Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting</a></strong></dt>
<dd>
<p>用 IPS 加权多项似然实现列表级全局无偏学习，并以 Bi-Weighting 平滑倾向估计与渐进式去偏，六数据集 Recall@20 平均提升 9.43%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/agent_conversational_rec/collab_rec_an_llm_based_agentic_framework_for_balancing_recommendations_in_/">Collab-Rec: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></strong></dt>
<dd>
<p>三角色 LLM 智能体 + 确定性 Moderator 多轮约束精炼，在 900 条旅游查询上兼顾约束满足与长尾多样性。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/investigating_the_effects_of_different_levels_of_user_control_in_an_interac/">Investigating the Effects of Different Levels of User Control in an Interactive Educational Recommender System</a></strong></dt>
<dd>
<p>在 MOOC 平台 CourseMapper 中实现输入/过程/输出三级用户控制的教育推荐系统，通过 N=184 被试间实验揭示输入控制是提升感知控制感的关键，并系统刻画控制感、透明度、信任、满意度与感知质量之间的结构关系。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/reproducibility_in_recommender_systems_a_survey/">Reproducibility in Recommender Systems: A Survey</a></strong></dt>
<dd>
<p>对 RecSys 可复现性赛道 2020–2025 年 51 篇论文的结构化元分析，揭示其方法论高度同质化、以“扩展”替代“严格复现”的实践落差。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/robustness_analysis_of_pareto_based_joint_evaluation_of_fairness_and_releva/">Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems</a></strong></dt>
<dd>
<p>提出 DPFR：以测试集真实标签构建相关性-公平性帕累托前沿，用模型得分到前沿参考点的距离联合评估推荐系统，比现有联合度量更稳健。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/robustness_and_user_perceived_value_of_popularity_calibration_in_music_reco/">Robustness and User-Perceived Value of Popularity Calibration in Music Recommendation: A User Study</a></strong></dt>
<dd>
<p>60人音乐推荐用户实验揭示：用户能感知流行度构成差异但不偏好校准列表，JSD 与感知流行度的相关性受熟悉度/列表构成/历史量调节，计算标签与用户判断仅弱一致。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/llm_rec/hadsf_aspect_aware_semantic_control_for_explainable_recommendation/">HADSF: Aspect Aware Semantic Control for Explainable Recommendation</a></strong></dt>
<dd>
<p>两阶段受控方面词表+动态个性化抽取，提出ADR/OFR幻觉度量，揭示幻觉-性能非单调关系，小模型可媲美大模型</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/bridging_semantic_understanding_and_popularity_bias_with_llms/">Bridging Semantic Understanding and Popularity Bias with LLMs</a></strong></dt>
<dd>
<p>将流行度偏差拆解为物品侧与用户侧语义，用结构化指令提示引导 RecLLM 双侧去偏，在 LtC/MRMC 与 MRR/F1 上同时超越多样性提示基线。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/social_group_rec/from_aggregation_to_selection_user_validated_distributed_social_recommendat/">From Aggregation to Selection: User-Validated Distributed Social Recommendation</a></strong></dt>
<dd>
<p>将分布式社交推荐从模型聚合转为用户驱动的算法选择+验证者多数共识投票，提出 Acc@K 共识评测，四数据集 Acc@2 平均提升 4.51%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/identifying_and_upweighting_power_niche_users_to_mitigate_popularity_bias_i/">Identifying and Upweighting Power-Niche Users to Mitigate Popularity Bias in Recommendations</a></strong></dt>
<dd>
<p>识别高活跃-小众偏好用户并对其在BPR损失中双重加权，在三个基准上实现性能与流行度去偏的帕累托双赢。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/learning_evolving_preferences_a_federated_continual_framework_for_user_cent/">Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation</a></strong></dt>
<dd>
<p>联邦持续推荐框架 FCUCR：时间感知自蒸馏抗遗忘 + 跨用户原型迁移促协同，四数据集全库排序 HR@10 最高提升 44.0%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/make_it_long_keep_it_fast_end_to_end_10k_long_user_behavior_sequence_modeli/">Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation</a></strong></dt>
<dd>
<p>以单查询目标-历史交叉注意力(STCA)将长序列建模降至 O(L)，配合请求级批处理(RLB)与稀疏训练稠密推理，在抖音十亿级流量实现 10K 序列端到端排序。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/mclmr_a_model_agnostic_causal_learning_framework_for_multi_behavior_recomme/">MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用后门干预+MoE自适应聚合+偏差感知对比学习，作为即插即用框架增强多行为推荐，三数据集全库排序显著提升。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multimodal_rec/multimodal_enhanced_federated_recommendation_a_group_wise_fusion_approach/">Multimodal-enhanced Federated Recommendation: A Group-wise Fusion Approach</a></strong></dt>
<dd>
<p>将多模态表征学习卸载至服务器，以组感知聚合+偏好引导蒸馏把多模态语义注入联邦推荐，五数据集全库排序 HR@50 最高提升约 30%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/rl_bandit_rec/off_policy_learning_with_limited_supply/">Off-Policy Learning with Limited Supply</a></strong></dt>
<dd>
<p>首次形式化库存受限下的离线策略学习，提出按“相对奖励差”分配稀缺物品的 OPLS，在合成与 KuaiRec 上稳定超越贪心 OPL。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/prototype_aligned_federated_soft_prompts_for_continual_web_personalization/">Prototype-Aligned Federated Soft-Prompts for Continual Web Personalization</a></strong></dt>
<dd>
<p>冻结骨干上注入双时间尺度软提示，长期提示锚定DP联邦原型库，八基准NDCG@10平均+2.9%且降低遗忘与适应步数。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/quantifying_user_coherence_a_unified_framework_for_analyzing_recommender_sy/">Quantifying User Coherence: A Unified Framework for Analyzing Recommender Systems Across Domains</a></strong></dt>
<dd>
<p>提出 Mean Surprise 与 Mean Conditional Surprise 两个信息论度量刻画用户画像一致性，揭示复杂模型增益集中于高一致用户，并支撑分层评测与用户分群建模。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/spattack_subgroup_poisoning_attacks_on_federated_recommender_systems/">Spattack: Subgroup Poisoning Attacks on Federated Recommender Systems</a></strong></dt>
<dd>
<p>首个联邦推荐子群投毒攻击：近似-提升范式+对比排斥与自适应加权，0.1%恶意客户端即达100% ER@5且几乎不伤非目标群。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/talos_optimizing_top_k_accuracy_in_recommender_systems/">Talos: Optimizing Top-$K$ Accuracy in Recommender Systems</a></strong></dt>
<dd>
<p>用量化阈值把 Top-K 截断改写为分数-阈值比较，配合采样分位回归与约束项，直接优化 Precision@K/Recall@K 并具备 DRO 鲁棒性。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/towards_fair_large_language_model_based_recommender_systems_without_costly_/">Towards Fair Large Language Model-based Recommender Systems without Costly Retraining</a></strong></dt>
<dd>
<p>将 LLM-RS 去偏重构为免重训机器遗忘任务：可学习掩码定位致偏样本 + 影响函数一步参数修正，公平性提升同时精度不降、耗时降 96%。</p>
</dd>
</dl>

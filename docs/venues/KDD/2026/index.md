---
hide:
  - toc
---

# 🏛️ KDD 2026 推荐系统论文解读

63 篇 KDD 2026 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/breaking_information_cocoons_a_hyperbolic_framework_for_balancing_explorati/">Breaking Information Cocoons: A Hyperbolic Framework for Balancing Exploration and Exploitation in Recommender Systems</a></strong></dt>
<dd>
<p>洛伦兹双曲空间对齐 LLM 语义与协同信号，并用 Dasgupta 代价驱动的无超参层次聚类实现用户可调探索-利用权衡，效用最高 +5.49%、多样性 +11.39%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/scaling_multilingual_semantic_search_in_uber_eats_delivery/">Scaling Multilingual Semantic Search in Uber Eats Delivery</a></strong></dt>
<dd>
<p>基于 gte-Qwen2 双塔统一多语言多垂类语义召回，InfoNCE+Triplet NCE 两阶段训练与 MRL 多维度服务，Store R@20 相对 Legacy 提升 88%。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl>
<dt><strong><a href="graph_rec/how_do_graph_signals_affect_recommendation_unveiling_the_mystery_of_low_and/">How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals</a></strong></dt>
<dd>
<p>理论证明同绝对值低频/高频图信号在推荐中等价，提出频率信号缩放器与空间翻转插件，在四个数据集全库排序下 Recall@20 最高提升约 3.0%。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/dynamic_spectral_denoising_with_global_context_attention_for_multi_behavior/">Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>沿特征维做 RFFT 学习视图自适应谱调制净化表征，再以净化全局锚做可靠性感知注意力融合，全库排序下 Taobao HR@10 相对提升 26.17%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (6)</h2>
<dl>
<dt><strong><a href="sequential_rec/fuxi_efficient_sequential_recommendation_with_exponential_power_temporal_en/">FuXi-$γ$: Efficient Sequential Recommendation with Exponential-Power Temporal Encoder and Diagonal-Sparse Positional Mechanism</a></strong></dt>
<dd>
<p>用艾宾浩斯遗忘曲线启发的指数幂时间编码器替代分桶查表，配合 Toeplitz 对角稀疏位置剪枝，在 Full-Ranking 下精度 SOTA 且训练/推理加速最高 4.74×/6.18×。</p>
</dd>
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
<dt><strong><a href="sequential_rec/scaling_recommender_transformers_to_one_billion_parameters/">Scaling Recommender Transformers to One Billion Parameters</a></strong></dt>
<dd>
<p>ARGUS 将自回归预训练解耦为反馈预测与下一物品预测双任务，把推荐 Transformer 编码器扩到 10 亿参数，线上听歌时长 +2.26%、点赞率 +6.37%。</p>
</dd>
<dt><strong><a href="sequential_rec/unleashing_the_potential_of_neighbors_diffusion_based_latent_neighbor_gener/">Unleashing the Potential of Neighbors: Diffusion-based Latent Neighbor Generation for Session-based Recommendation</a></strong></dt>
<dd>
<p>用检索先验与多模态信号双路引导扩散模型生成“潜在邻居”，突破可观测邻居边界，Full-Ranking 下 P@20 最高提升 18.45%。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (13)</h2>
<dl>
<dt><strong><a href="generative_rec/apao_bridging_the_training_inference_gap_in_generative_recommendation_via_a/">APAO: Bridging the Training-Inference Gap in Generative Recommendation via Adaptive Prefix-Aware Optimization</a></strong></dt>
<dd>
<p>揭示 Beam Search 剪枝与 teacher-forcing 训练的前缀级不一致，用自适应最差前缀损失对齐训练与推理，全库排序下 Recall@20 最高提升 13.3%。</p>
</dd>
<dt><strong><a href="generative_rec/breaking_the_likelihood_trap_consistent_generative_recommendation_with_grap/">Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model</a></strong></dt>
<dd>
<p>用有向无环图解码器扩展非自回归生成空间，并以 Gumbel-Softmax 打通评估器梯度，破解似然陷阱，快手线上 Views +0.78%。</p>
</dd>
<dt><strong><a href="generative_rec/collaborative_memory_augmentation_for_generative_recommendation/">Collaborative Memory Augmentation for Generative Recommendation</a></strong></dt>
<dd>
<p>为生成式推荐引入跨用户协同记忆库，用可学习查询token压缩序列、目标感知检索与门控交叉注意力融合，Full-Ranking下NDCG@5最高提升23.7%。</p>
</dd>
<dt><strong><a href="generative_rec/flowtime_towards_continuous_generative_watch_time_prediction_via_flow_based/">FlowTime: Towards Continuous Generative Watch Time Prediction via Flow-based Personalized Priors</a></strong></dt>
<dd>
<p>提出连续生成式回归新范式，用一步VAE+归一化流个性化先验建模多峰观看时长分布，KuaiRec MAE降0.676%、线上播放时长+1.044%。</p>
</dd>
<dt><strong><a href="generative_rec/forge_forming_semantic_identifiers_for_generative_retrieval_in_industrial_d/">FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets</a></strong></dt>
<dd>
<p>系统化拆解 SID 构建全流程（多模态特征/量化编码/冲突后处理），提出免训练的 Embedding HitRate 与 Gini 系数代理指标，淘宝线上交易量 +0.35%。</p>
</dd>
<dt><strong><a href="generative_rec/hi_sam_a_hierarchical_structure_aware_multi_modal_framework_for_large_scale/">Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>解耦语义分词器+分层记忆锚点Transformer，修复语义ID推荐中分词冗余与扁平架构失配，工业A/B响应率+6.55%</p>
</dd>
<dt><strong><a href="generative_rec/hierarchical_residual_policy_optimization_for_generative_recommendations/">Hierarchical Residual Policy Optimization for Generative Recommendations</a></strong></dt>
<dd>
<p>将物品级反馈分解为 SID 前缀残差信用，用 RRPO 做保守 token 级策略优化，KuaiSim 会话回报 10.53 vs 8.14，线上 IAA Target Cost 最高 +3.49%。</p>
</dd>
<dt><strong><a href="generative_rec/pcr_ca_parallel_codebook_representations_with_contrastive_alignment_for_mul/">PCR-CA: Parallel Codebook Representations with Contrastive Alignment for Multiple-Category App Recommendation</a></strong></dt>
<dd>
<p>用并行码本VQ-AE替代RQ-VAE层级量化建模多类别App语义，配合用户/物品双层对比对齐与双流注意力，微软商店线上CTR+10.52%。</p>
</dd>
<dt><strong><a href="generative_rec/reasoning_over_semantic_ids_enhances_generative_recommendation/">Reasoning over Semantic IDs Enhances Generative Recommendation</a></strong></dt>
<dd>
<p>通过教师模型合成富化语料强化 SID-语言对齐，再用 GRPO 结果驱动强化学习激发 SID 推理，三数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="generative_rec/s_2_gr_stepwise_semantic_guided_reasoning_in_latent_space_for_generative_re/">S$^2$GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation</a></strong></dt>
<dd>
<p>在每级 SID 码生成前插入受码本聚类语义对比监督的思考 token，实现逐步可验证潜在推理，工业短视频 A/B 显著提升时长。</p>
</dd>
</dl>
<p><a href="generative_rec/">查看全部13篇「生成式推荐与离散语义 ID」论文 →</a></p>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (7)</h2>
<dl>
<dt><strong><a href="llm_rec/copersona_collaborative_persona_graphs_for_robust_llm_personalization/">CoPersona: Collaborative Persona Graphs for Robust LLM Personalization</a></strong></dt>
<dd>
<p>用方面级多重人物图+双分支推理补全稀疏用户画像，Amazon六域生成式评测全面超越DEP/DPL。</p>
</dd>
<dt><strong><a href="llm_rec/enriching_semantic_profiles_into_knowledge_graph_for_recommender_systems_us/">Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 为 KG 全实体生成语义画像，以可逆加性注入+成对匹配对齐，Full-Ranking 下 Recall@40 最高提升 2.6%。</p>
</dd>
<dt><strong><a href="llm_rec/improving_item_discoverability_in_e_commerce_search_via_related_intent_gene/">Improving Item Discoverability in e-Commerce Search via Related Intent Generation</a></strong></dt>
<dd>
<p>用闭源LLM+蒸馏SLM生成替代/互补/主题三类隐式意图以扩展电商搜索召回，会话购买预测与LLM评审双轨评估，覆盖流量60%→80%、成本仅30%。</p>
</dd>
<dt><strong><a href="llm_rec/mgfrec_towards_reinforced_reasoning_recommendation_with_multiple_groundings/">MGFRec: Towards Reinforced Reasoning Recommendation with Multiple Groundings and Feedback</a></strong></dt>
<dd>
<p>将单步 grounding 扩展为多轮物品空间接地并引入用户 Agent 反馈，用 GRPO 训练 LLM 推理推荐，Full-Ranking 下 H@5 最高提升约 8.8%。</p>
</dd>
<dt><strong><a href="llm_rec/mixture_of_experts_knowledge_graph_retrieval_augmented_generation_for_multi/">Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation</a></strong></dt>
<dd>
<p>用 MoE 多粒度 KG 检索智能体按查询复杂度路由检索粒度，配合知识对齐与对比偏好优化，在 20 候选池协议下 Accuracy 相对提升 5.8%–20.4%。</p>
</dd>
<dt><strong><a href="llm_rec/proex_a_unified_framework_leveraging_large_language_model_with_profile_extr/">ProEx: A Unified Framework Leveraging Large Language Model with Profile Extrapolation for Recommendation</a></strong></dt>
<dd>
<p>用四步 CoT 为每个用户/物品生成多面画像并做环境不变性融合，在三个数据集上对 6 个基模型平均提升 5%+。</p>
</dd>
<dt><strong><a href="llm_rec/whose_name_comes_up_ii_benchmarking_and_intervention_based_auditing_of_llm_/">Whose Name Comes Up? II: Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation</a></strong></dt>
<dd>
<p>提出 LLMScholarBench，联合审计 22 个 LLM 在 5 类学者推荐任务、3 类推理期干预下的技术质量与社会代表性权衡。</p>
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
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (3)</h2>
<dl>
<dt><strong><a href="multimodal_rec/text_guided_visual_representation_learning_for_robust_multimodal_e_commerce/">Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation</a></strong></dt>
<dd>
<p>以结构化元数据初始化语义查询、双流解耦视觉证据并用可靠性双门控校准，在 134K 全库 I2I 检索上 H@100 相对提升 6.04%。</p>
</dd>
<dt><strong><a href="multimodal_rec/uninote_a_unified_embedding_model_for_multimodal_representation_and_ranking/">UniNote: A Unified Embedding Model for Multimodal Representation and Ranking</a></strong></dt>
<dd>
<p>小红书工业级 I2I 统一嵌入模型：对比 SFT + GRPO 相对重排两阶段训练，10 类检索任务全 SOTA，MRL 支持弹性维度部署。</p>
</dd>
<dt><strong><a href="multimodal_rec/vi_mmrec_similarity_aware_training_cost_free_virtual_user_item_interactions/">VI-MMRec: Similarity-Aware Training Cost-free Virtual User-Item Interactions for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用模态相似度构造虚拟交互矩阵并以统计权重融合，零训练开销即插即用地缓解多模态推荐数据稀疏，Full-Ranking 下 Recall@10 最高提升 22.25%。</p>
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
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (4)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/atomic_intent_reasoning_bringing_llm_semantics_to_industrial_cross_domain_r/">Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations</a></strong></dt>
<dd>
<p>将 LLM 意图推理离线化为原子行为-意图对缓存，在线经意图树构建与目标感知检索实现约 400× 加速，快手 A/B 测试 GMV +3.446%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/meta_lattice_model_space_redesign_for_cost_effective_industry_scale_ads_rec/">Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations</a></strong></dt>
<dd>
<p>以模型空间重构统一碎片化广告组合，融合归因窗口混合、帕累托特征选择与推理期知识迁移，线上Topline +10%、容量省20%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/sharpness_aware_model_merging_with_salience_recovery_for_llm_based_cross_do/">Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用锐度感知平坦极小值对齐 + 重尾重参数化恢复显著偏好信号，破解 LLM 跨域序列推荐模型合并中的参数冲突与多域饱和瓶颈。</p>
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
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (5)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/ctr_sink_attention_sink_for_language_models_in_click_through_rate_predictio/">CTR-Sink: Attention Sink for Language Models in Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>在文本化行为序列中插入携带时间距离的 [SINK] 锚点并配合两阶段训练与 sink 间注意力增强，缓解语义碎片化，AUC 稳定提升 0.2–0.6%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/expand_more_shrink_less_shaping_effective_rank_dynamics_for_dense_scaling_i/">Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation</a></strong></dt>
<dd>
<p>从有效秩视角诊断 RankMixer 的阻尼振荡塌缩，提出参数化全混合与 GLU-PFFN 实现“多扩张少收缩”，Criteo/Avazu 上 AUC 提升超 0.001。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/from_scaling_to_structured_expressivity_rethinking_transformers_for_ctr_pre/">From Scaling to Structured Expressivity: Rethinking Transformers for CTR Prediction</a></strong></dt>
<dd>
<p>用域感知参数重构 Transformer 块，将复杂度从词表规模 n 转向域数 F，配合基组合超网络实现可预测幂律缩放，离线 AUC 最高 +4.38%，线上 CTR +2.33%。</p>
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
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/heterogeneous_multi_treatment_uplift_modeling_for_trade_off_optimization_in/">Heterogeneous Multi-treatment Uplift Modeling for Trade-off Optimization in Short-Video Recommendation</a></strong></dt>
<dd>
<p>用异构多处理增益建模+在线动态权重，个性化权衡短视频消费与商业化指标，快手全量部署</p>
</dd>
<dt><strong><a href="rl_bandit_rec/hoba_hierarchical_on_policy_bidding_agents_for_adaptive_online_advertising/">HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising</a></strong></dt>
<dd>
<p>三层时间尺度解耦出价：LLM 小时级推理超参、SARSA 分钟级因果去偏选专家、专家池逐次拍卖执行，AuctionNet 与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/learning_peer_influence_probabilities_with_linear_contextual_bandits/">Learning Peer Influence Probabilities with Linear Contextual Bandits</a></strong></dt>
<dd>
<p>将同伴影响力概率学习建模为双目标线性上下文 Bandit，刻画 Regret–RMSE 帕累托前沿并提出 InfluenceCB 以单参数 β 导航权衡。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/uncertainty_calibrated_recommendations_for_low_active_users/">Uncertainty-Aware Adaptive Recommendation across User Lifecycle</a></strong></dt>
<dd>
<p>用 EPE Critic 与经验贝叶斯先验方差量化逐样本认知不确定性，对低活跃用户风险规避降权、高活跃用户 UCB 探索加成，14 天线上 A/B 验证留存与多样性双收益。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/bradley_terry_rankings_for_recommender_systems_across_dataset_taxonomies/">Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies</a></strong></dt>
<dd>
<p>用 Bradley-Terry 概率排序替代指标均值聚合，在 14 算法×89 数据集上给出数据集感知的推荐算法排名与免运行预测。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/canniuplift_a_holistic_framework_for_mitigating_seller_and_incentive_cannib/">CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling</a></strong></dt>
<dd>
<p>用平台级GMV一致性约束与核销路径分解，缓解多卖家电商uplift建模中的卖家级与激励级蚕食，线上ΔGMV相对提升4.08%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/deep_learning_causal_retrieval_optimization_for_efficient_e_commerce_distri/">Deep-learning Causal Retrieval Optimization for Efficient e-commerce Distribution in Pinterest</a></strong></dt>
<dd>
<p>将购物候选生成器触发建模为因果决策，用 DR 多任务提升学习+随机 Holdout+线性回放，线上触发率降 85% 而购物会话持平。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/llm_as_a_judge_for_reliable_and_explainable_offline_evaluation_in_top_k_rec/">LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation</a></strong></dt>
<dd>
<p>用 LLM 从用户文本行为构建语义偏好代理替代 holdout ID 匹配，以 reasoning-then-scoring 生成可解释 Top-K 评测，模型排序一致性较常规测试提升 40–150%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/lower_bias_higher_welfare_how_creator_competition_reshapes_bias_variance_tr/">Lower Bias, Higher Welfare: How Creator Competition Reshapes Bias-Variance Tradeoff in Recommendation Platforms?</a></strong></dt>
<dd>
<p>博弈论建模创作者竞争如何重塑平台正则化选择，理论与实验一致表明策略环境下应弱正则（低偏差）以最大化用户福利。</p>
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

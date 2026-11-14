---
hide:
  - toc
---

# 🏛️ SIGIR 2026 推荐系统论文解读

116 篇 SIGIR 2026 论文解读，覆盖 14 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (6)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong></dt>
<dd>
<p>以塔内自校正、跨塔EMA同步与级联模型共享三机制增强双塔召回，离线三数据集AUC一致提升，线上广告收入+8.36%且QPS仅降0.59%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong></dt>
<dd>
<p>以原型意图与分布意图重构表征，用对齐-均匀性+粗细粒度匹配联合优化，Amazon-book R@20 相对提升 10.6%</p>
</dd>
<dt><strong><a href="collaborative_filtering/ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong></dt>
<dd>
<p>在 Item-ID 投影层上施加 Stiefel 流形正交约束，抑制稀疏缩放下的嵌入坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom Index 与 Int8 融合 ANN 协同设计，吞吐提升 23.7×、成本效率提升 13.35×。</p>
</dd>
<dt><strong><a href="collaborative_filtering/unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动为相关项内偏好信号，融合交叉编码器分级标签、多通道检索先验与去偏互动构建统一双塔训练目标，离线 P@25 +10.5%、NDCG@25 +5.7%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong></dt>
<dd>
<p>零权重免训练新闻推荐框架，以时间衰减×表征相似度打分，离线全库排序超越 NRMS，线上 CTR 逼近 SOTA 且推理快 600 倍。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl>
<dt><strong><a href="graph_rec/aspect_aware_content_based_recommendations_for_mathematical_research_papers/">Aspect-Aware Content-Based Recommendations for Mathematical Research Papers</a></strong></dt>
<dd>
<p>面向数学论文的 aspect 条件异构图推荐，构建 GoldRiM/SilverRiM 数据集，AchGNN 在 Full-Ranking 下 MRR 相对 GraphCL 提升 26%</p>
</dd>
<dt><strong><a href="graph_rec/dcgl_dual_channel_graph_learning_with_large_language_models_for_knowledge_a/">DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation</a></strong></dt>
<dd>
<p>双通道解耦 LLM 语义与 ID 行为信号，多级对比对齐 + 频率感知门控融合，四数据集全库排序全面超越 SOTA。</p>
</dd>
<dt><strong><a href="graph_rec/qagcf_graph_collaborative_filtering_for_q_a_recommendation/">QAGCF: Graph Collaborative Filtering for Q&amp;A Recommendation</a></strong></dt>
<dd>
<p>将问答对解耦为用户-问题/用户-答案双协同图与问题-答案/问题-问题双语义图，融合为全局异质图后用 Jacobi 多项式图滤波器分离高低频信号，配合噪声对比学习，在知乎与商业数据集 Full-Ranking 下 Recall@10 提升 5.16%–5.34%。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 学习用户自适应增强，四数据集全排序下 HR/NDCG 平均提升约 7-8%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong></dt>
<dd>
<p>识别地理约束下 in-batch 负样本中的“不可能负样本”问题，提出区域同质批采样 RCBS，仅改批构造即显著提升用户表征与线上指标。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (15)</h2>
<dl>
<dt><strong><a href="sequential_rec/beyond_item_ids_scaling_short_form_video_recommendation_via_semantic_native/">Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling</a></strong></dt>
<dd>
<p>用深度截断 Bi-gram 语义 ID 替代正交 Video ID，并以无参时间折叠压缩 Transformer 将序列扩至 2000，线上满意观看时长 +1.42%。</p>
</dd>
<dt><strong><a href="sequential_rec/case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong></dt>
<dd>
<p>以日历时间二值信号+共享多尺度时序卷积建模物品复购节奏，再用 ISAB 建模跨物品依赖，在四个杂货数据集与千万级生产系统上取得显著提升。</p>
</dd>
<dt><strong><a href="sequential_rec/cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong></dt>
<dd>
<p>提出可学习多序列构造与线性注意力，缓解单序列上下文污染，Meta 四场景 NE 降 0.3%–0.6%。</p>
</dd>
<dt><strong><a href="sequential_rec/divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong></dt>
<dd>
<p>借鉴 ZPD 理论构建 Teacher-Peer-Self 多源负采样，用分歧重排提升多样性、共识蒸馏复用打分开销，六数据集全库排序下 Recall@20 最高提升 27.3%。</p>
</dd>
<dt><strong><a href="sequential_rec/fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>以目标条件化频谱熵观察为动机，用目标感知频谱滤波+时域双分支做CTR预估，三数据集GAUC/AUC全面SOTA。</p>
</dd>
<dt><strong><a href="sequential_rec/flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结+可学习双网络的子模块组合模拟指数级集成多样性，推理仅用单网络，NDCG@20 最高提升 9.70% 且收敛快 7.69 倍。</p>
</dd>
<dt><strong><a href="sequential_rec/from_time_and_place_to_preference_llm_driven_geo_temporal_context_in_recomm/">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></strong></dt>
<dd>
<p>用 LLM 从时间戳+粗粒度位置生成地理-时间上下文嵌入，以特征融合或辅助对齐损失注入 SASRec，并提出轻量信息量诊断测试。</p>
</dd>
<dt><strong><a href="sequential_rec/fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习消除空间结构冲突，全库排序下尾部长尾 H@10 最高提升 427%。</p>
</dd>
<dt><strong><a href="sequential_rec/learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱食门控+TTNP自监督学习商品复购周期，PPRR降超60%</p>
</dd>
<dt><strong><a href="sequential_rec/pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致模型评估失真，系统分析SSS生效条件与数据分布重塑机理。</p>
</dd>
</dl>
<p><a href="sequential_rec/">查看全部15篇「序列与会话推荐」论文 →</a></p>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (19)</h2>
<dl>
<dt><strong><a href="generative_rec/a_cascaded_generative_approach_for_e_commerce_recommendations/">A Cascaded Generative Approach for e-Commerce Recommendations</a></strong></dt>
<dd>
<p>提出级联生成式货架构建框架：LLM1 生成投放主题、LLM2 经 RAG 约束生成检索关键词，教师-学生蒸馏降本，线上购物车添加 +2.7%。</p>
</dd>
<dt><strong><a href="generative_rec/bi_level_optimization_for_generative_recommendation_bridging_tokenization_a/">Bi-Level Optimization for Generative Recommendation: Bridging Tokenization and Generation</a></strong></dt>
<dd>
<p>将分词器与推荐器建模为双层优化，用元学习+梯度手术让推荐损失直接监督RQ-VAE码本，Full-Ranking下全面超越TIGER/LETTER/ETEGRec。</p>
</dd>
<dt><strong><a href="generative_rec/card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recomm/">CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation</a></strong></dt>
<dd>
<p>将文本/图像/协同信号渲染为统一视觉语义单元，并用可逆非均匀变换校正偏斜分布，提升码本利用率与生成式推荐精度。</p>
</dd>
<dt><strong><a href="generative_rec/differentiable_semantic_id_for_generative_recommendation/">Differentiable Semantic ID for Generative Recommendation</a></strong></dt>
<dd>
<p>用 Gumbel 噪声+不确定性衰减实现可微语义ID联合训练，Leave-One-Out 全库排序下 R@10 相对 TIGER 提升约 12%。</p>
</dd>
<dt><strong><a href="generative_rec/discrete_preference_learning_for_personalized_multimodal_generation/">Discrete Preference Learning for Personalized Multimodal Generation</a></strong></dt>
<dd>
<p>用模态专属LightGCN+RQ-VAE把协同偏好量化成离散token注入LLM与扩散模型，配合跨模态一致性奖励实现个性化图文生成。</p>
</dd>
<dt><strong><a href="generative_rec/dual_diffusional_generative_fashion_recommendation/">Dual-Diffusional Generative Fashion Recommendation</a></strong></dt>
<dd>
<p>双分支扩散 Transformer 联合生成时尚单品图像与属性级文本，以结构化偏好加权采样建模用户，PFITB/GOR 双任务个性化指标显著领先且训练提速约 7 倍。</p>
</dd>
<dt><strong><a href="generative_rec/fave_flow_based_average_velocity_establishment_for_sequential_recommendatio/">FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义锚先验替代高斯噪声、以平均速度场实现单步流匹配生成推荐，全库排序下 N@20 相对提升 9.9%，推理提速 20 倍以上。</p>
</dd>
<dt><strong><a href="generative_rec/gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong></dt>
<dd>
<p>用 Conv1D 局部混合、序列级 Key 门控软遗忘与门控 RMSNorm 将核化线性注意力扩展为双向线性时间算子，1:2 混合编码器在 Yandex Music 上追平双向自注意力，32768 长度单层提速 8.2×。</p>
</dd>
<dt><strong><a href="generative_rec/genrec_a_preference_oriented_generative_framework_for_large_scale_recommend/">GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>以 Page-wise NTP 解一对多歧义、线性 Token Merger 压缩 2× 输入、GRPO-SR 混合奖励对齐偏好，京东线上点击 +9.5%、成交 +8.7%。</p>
</dd>
<dt><strong><a href="generative_rec/gflowgr_fine_tuning_generative_recommendation_frameworks_with_generative_fl/">GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks</a></strong></dt>
<dd>
<p>将生成式推荐重构为集合级序列生成，用 GFlowNets 微调使生成概率正比于行为感知奖励，缓解曝光偏差，淘宝线上年收入相对提升 0.4%。</p>
</dd>
</dl>
<p><a href="generative_rec/">查看全部19篇「生成式推荐与离散语义 ID」论文 →</a></p>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (22)</h2>
<dl>
<dt><strong><a href="llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在保持语义层级的同时缓解几何失衡，Full-Ranking 下 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="llm_rec/agentic_multi_source_grounding_for_enhanced_query_intent_understanding_a_do/">Agentic Multi-Source Grounding for Enhanced Query Intent Understanding: A DoorDash Case Study</a></strong></dt>
<dd>
<p>用目录实体检索+Agentic 网页搜索双源 grounding 替代 winner-takes-all 分类，输出有序双意图并由可插拔消歧层裁决，长尾意图准确率 +13.0pp。</p>
</dd>
<dt><strong><a href="llm_rec/bear_towards_beam_search_aware_optimization_for_recommendation_with_large_l/">BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>揭示 SFT 训练与 beam search 推理的剪枝不一致，提出 token 级 top-B 必要条件的可微正则 BEAR，平均提升 12.50% 且训练开销几乎不变。</p>
</dd>
<dt><strong><a href="llm_rec/beyond_static_best_of_n_bayesian_list_wise_alignment_for_llm_based_recommen/">Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用贝叶斯后验融合静态先验与在线 rollout 证据，构建自演化 BoN 目标，突破静态对齐的性能上界，NDCG@5 相对提升 23.1%。</p>
</dd>
<dt><strong><a href="llm_rec/every_preference_has_its_strength_injecting_ordinal_semantics_into_llm_base/">Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders</a></strong></dt>
<dd>
<p>将显式评分建模为LLM词表中的序数语义锚点，以强度感知对齐交互表征，在序列推荐Hit@1与成对偏好准确率上显著超越CF-LLM混合基线。</p>
</dd>
<dt><strong><a href="llm_rec/factorized_latent_reasoning_for_llm_based_recommendation/">Factorized Latent Reasoning for LLM-based Recommendation</a></strong></dt>
<dd>
<p>将 LLM 隐式推理拆为 K 个解耦偏好因子并配 GRPO 对齐，全库约束解码下 N@5 最高提升 10.26%</p>
</dd>
<dt><strong><a href="llm_rec/filling_the_gaps_selective_knowledge_augmentation_for_llm_recommenders/">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></strong></dt>
<dd>
<p>提出 CKP 比较式知识探测量化 LLM 对物品的协同知识缺口，仅对知识贫乏物品选择性注入属性与参考物品，免训练提升 Recall@1 与长尾覆盖。</p>
</dd>
<dt><strong><a href="llm_rec/genfacet_end_to_end_generative_faceted_search_via_multi_task_preference_ali/">GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment in E-Commerce</a></strong></dt>
<dd>
<p>将电商分面搜索重构为分面生成与查询改写两个耦合生成任务，经蒸馏+SFT+GRPO 对齐搜索效用，线上分面 CTR 相对提升 42.0%。</p>
</dd>
<dt><strong><a href="llm_rec/itemrag_item_based_retrieval_augmented_generation_for_llm_based_recommendat/">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 RAG 从粗粒度用户历史检索转向细粒度物品级检索，融合共购与语义信息，标准与冷启动场景全面超越用户级 RAG 基线。</p>
</dd>
<dt><strong><a href="llm_rec/l2rec_towards_dual_view_understanding_of_llms_for_personalized_recommendati/">L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation</a></strong></dt>
<dd>
<p>以参数级双视图 LoRA-MoE 替代输入/输出级融合，个性化路由对齐行为与语义视图，四数据集 N@10 提升 3.87%–8.02%，线上 CTR +9.24%。</p>
</dd>
</dl>
<p><a href="llm_rec/">查看全部22篇「大语言模型推荐 (LLM4Rec)」论文 →</a></p>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (8)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/a_standardized_re_evaluation_of_conversational_recommender_systems_on_the_r/">A Standardized Re-evaluation of Conversational Recommender Systems on the ReDial Dataset</a></strong></dt>
<dd>
<p>统一复现 7 个 CRS 模型，揭示 Recall@1 复现粒度鸿沟、约 50% 精度来自重复捷径、增益多源于 LLM 骨干而非架构创新。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/bridging_passive_and_active_enhancing_conversation_starter_recommendation_v/">Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling</a></strong></dt>
<dd>
<p>用对抗分布对齐+RQ-VAE语义离散化把用户主动查询引入对话开场白推荐，打破曝光-点击回音室，线上活跃天数+0.04%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/conv_finre_a_conversational_and_longitudinal_benchmark_for_utility_grounded/">Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation</a></strong></dt>
<dd>
<p>用逆优化从纵向决策轨迹反推用户风险偏好，构建四视图多视角金融推荐基准，揭示 LLM 效用对齐与行为模仿的根本张力。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/meituan_merchant_business_diagnosis_via_policy_guided_dual_process_user_sim/">Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation</a></strong></dt>
<dd>
<p>从行为轨迹挖掘决策策略作为共享对齐层，锚定 LLM 推理与 GBDT 拟合双过程分支，群体仿真误差降至 8.80%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/recquest_towards_estimating_user_domain_knowledge_in_conversational_recomme/">RecQuest: Towards Estimating User Domain Knowledge in Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出游戏化数据采集协议 RecQuest 与知识估计新任务，用客观问卷替代自评建立 ground truth，并给出 LLM 零/少样本与微调基线。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/task_aware_automated_user_profile_generation_for_recommendation_simulation_/">Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models</a></strong></dt>
<dd>
<p>提出免训练三阶段画像生成框架 APG4RecSim，用语义合并+反事实因果映射将交互历史转为任务对齐画像，在判别/排序/评分三任务上取得最佳综合表现。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/trace_a_conversational_framework_for_sustainable_tourism_recommendation_wit/">TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations</a></strong></dt>
<dd>
<p>多智能体 LLM 框架以澄清提问+反事实解释实现可持续旅游推荐的非强制绿色引导，24 人用户研究验证。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 分层 MoE 架构，显式建模“何时问、怎么问”的阶段感知偏好引导策略，Recall@10 相对最优基线提升 14.6%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (8)</h2>
<dl>
<dt><strong><a href="multimodal_rec/a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 1.5B LLaMA2+BLIP-2 将图像转字幕并 token 化为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%，延迟中性。</p>
</dd>
<dt><strong><a href="multimodal_rec/from_raw_features_to_effective_embeddings_a_three_stage_approach_for_multim/">From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation</a></strong></dt>
<dd>
<p>用 VLM/LLM 摘要 + LightGCN 传播 + 跨视图对比学习三阶段精炼食谱多模态特征，Full-Ranking 下 Recall@10 提升 7–15%。</p>
</dd>
<dt><strong><a href="multimodal_rec/mllmrec_a_preference_reasoning_paradigm_with_graph_refinement_for_multimoda/">MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 推理用户偏好画像替代噪声聚合初始化，并以阈值去噪+共现增强精修物品图，三数据集 11/12 指标 SOTA。</p>
</dd>
<dt><strong><a href="multimodal_rec/multi_modal_relational_item_representation_learning_for_inferring_substitut/">Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items</a></strong></dt>
<dd>
<p>融合多模态基础模型与元路径行为编码，以自监督对比去噪与 LLM 校验弱监督推断可替代/互补关系，M@10 平均提升 26.1%/39.2%。</p>
</dd>
<dt><strong><a href="multimodal_rec/robust_multimodal_recommendation_via_graph_retrieval_enhanced_modality_comp/">Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion</a></strong></dt>
<dd>
<p>用模态感知子图检索替代 K 跳邻居聚合，配合图 Transformer 与稀疏路由码本补全缺失模态，40% 缺失率下 Recall@20 相对提升 4.2%–5.1%。</p>
</dd>
<dt><strong><a href="multimodal_rec/structural_and_disentangled_adaptation_of_large_vision_language_models_for_/">Structural and Disentangled Adaptation of Large Vision Language Models for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用模态内结构软教师对齐 LVLM 跨模态表征，并以专家化门控 LoRA 解耦梯度流，即插即用提升多模态与序列推荐，长尾增益最高 18.70%。</p>
</dd>
<dt><strong><a href="multimodal_rec/timemm_time_as_operator_spectral_filtering_for_dynamic_multimodal_recommend/">TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation</a></strong></dt>
<dd>
<p>将交互新近度映射为时间核加权图算子族，隐式实现免特征分解的多尺度谱滤波，配合自适应算子混合与模态路由，全库排序下 R@20 相对提升 3%–10%。</p>
</dd>
<dt><strong><a href="multimodal_rec/well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用交互物品模态特征+聚类全局语义免训练初始化用户表征，弥合用户-物品语义鸿沟，六模型四数据集Recall@5最高提升36.4%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl>
<dt><strong><a href="cold_start_rec/genrecedit_adapting_model_editing_for_generative_recommendation_with_cold_s/">GenRecEdit: Adapting Model Editing for Generative Recommendation with Cold-Start Items</a></strong></dt>
<dd>
<p>将 NLP 模型编辑迁移至生成式推荐，用逐位置定位-编辑+One-One 触发注入冷启动物品 SID 模式，仅需重训 9.5% 时间即大幅缓解冷启动坍缩。</p>
</dd>
<dt><strong><a href="cold_start_rec/sparse_contrastive_learning_for_content_based_cold_item_recommendation/">Sparse Contrastive Learning for Content-Based Cold Item Recommendation</a></strong></dt>
<dd>
<p>抛弃 CF 对齐、纯内容空间 item-item 相似度建模，用 α-entmax 稀疏化 sampled softmax 并支持内容空间蒸馏，冷物品排序与曝光公平性双赢。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (4)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/bridging_behavior_and_semantics_for_time_aware_cross_domain_sequential_reco/">Bridging Behavior and Semantics for Time-aware Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用神经 ODE 解耦长短期行为、LLM 离散时间 token 反事实增强语义、门控自适应域迁移，1:999 采样排序下 MRR 最高提升 30.5%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/fedmm_federated_collaborative_signal_quantization_for_multi_market_ctr_pred/">FedMM: Federated Collaborative Signal Quantization for Multi-Market CTR Prediction</a></strong></dt>
<dd>
<p>用双层 RQ-VAE 码本作联邦传输介质，对齐跨市场不相交 ID 空间，AUC 全面超越本地/联邦基线且通信量降 20×。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong></dt>
<dd>
<p>用可迁移物品增广+双阶段训练+域感知画像，缓解CDSR域不平衡与域迁移难题，全库排序下H@10最高提升26.96%</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/reinforcing_user_interest_evolution_in_multi_scenario_learning_for_recommen/">Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems</a></strong></dt>
<dd>
<p>用 Double Q-learning 估计场景级行为置信度作为门控，加权三元组对比损失以抑制多场景负迁移，KuaiSAR 全库排序 N@20 提升约 3.3 倍。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/rest_a_plug_and_play_spatially_constrained_representation_enhancement_frame/">ReST: A Plug-and-Play Spatially-Constrained Representation Enhancement Framework for Local-Life Recommendation</a></strong></dt>
<dd>
<p>以物品为中心，用属性语义预热+空间约束难负采样对比学习增强长尾物品ID表征，离线AUC最高+5.90%，线上GMV+2.804%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (9)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>揭示稠密MLP隐式权重稀疏现象，提出 filter-then-fuse 多视图显式稀疏框架 SSR，在十亿级工业数据与三个公开集上以更低算力超越 RankMixer。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/cheaper_is_better_a_discount_aware_network_for_conversion_rate_prediction_i/">Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System</a></strong></dt>
<dd>
<p>用 FFT 时频分解 + 用户/上下文双重去偏 + 折扣回归辅助任务建模商品折扣率时序，离线 AUC +1.61%、线上 pCVR +3.63%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/counterfactual_multi_task_learning_for_delayed_conversion_modeling_in_e_com/">Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion</a></strong></dt>
<dd>
<p>面向大促预热期“集中式延迟转化”的因果多任务 CVR 模型，联合直接/延迟转化建模、个性化门控迁移与 ATC 反事实正则，离线 AUC 显著超越延迟反馈基线并落地线上 A/B。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/exploring_test_time_scaling_via_prediction_merging_on_large_scale_recommend/">Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出预测合并式测试时缩放，用异构架构与同构随机种子生成多样 logit 并平均融合，三数据集八骨干上稳定超越参数缩放。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/follow_the_trace_exploiting_post_click_trajectories_for_online_delayed_conv/">Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将点击后行为演化建模为反馈轨迹，用贝叶斯分解+可靠性门控回溯补全动态修正未转化样本后验，Criteo AUC 0.8382 超越 DEFUSE。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/generative_bid_shading_in_real_time_bidding_advertising/">Generative Bid Shading in Real-Time Bidding Advertising</a></strong></dt>
<dd>
<p>用自回归残差生成+CHNet奖励模型+GRPO偏好对齐替代两阶段出价调价，离线SR达41.74%、线上ROI+3.4%且推理耗时降37.6%</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/hyformer_revisiting_the_roles_of_sequence_modeling_and_feature_interaction_/">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></strong></dt>
<dd>
<p>用 Global Tokens 交替执行 Query Decoding 与 Query Boosting，统一长序列建模与特征交叉，AUC 提升 0.17% 且 FLOPs 仅 3.9T。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/r_f_inventory_a_large_scale_dataset_for_monotonic_inventory_estimation_in_r/">R&amp;F-Inventory: A Large-Scale Dataset for Monotonic Inventory Estimation in Reach and Frequency Advertising</a></strong></dt>
<dd>
<p>首个公开的 R&amp;F 合约库存预估数据集，以「定向-排期-频控」为上下文提供完整预算-效果曲线，定义单点预测与多维单调曲线重建两大基准任务。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/understanding_dnns_in_feature_interaction_models_a_dimensional_collapse_per/">Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective</a></strong></dt>
<dd>
<p>从嵌入维度坍缩视角重新解释 DNN 在特征交叉模型中的作用：并行/堆叠 DNN 均能平坦化奇异值谱、提升 RankMe，并给出基于梯度的机理分析。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (4)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong></dt>
<dd>
<p>用残差量化码本近似 Target Attention，将交互式建模引入粗排，AUC 0.799、线上 GMV +7.57%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层消除候选顺序依赖，单次前向实现近完美排列不变性，nDCG 仅微降。</p>
</dd>
<dt><strong><a href="multi_task_reranking/towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong></dt>
<dd>
<p>用 LoRA+MoE 独立参数通路隔离稀疏问卷满意度信号，并以在线 DPO 对齐主排序模型，离线 NDCG@5 提升约 3%，线上 Q-Satisfied 提升 0.5–0.9%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong></dt>
<dd>
<p>用后验价值对齐把抽象混合分映射到有效完播率语义空间，再以独立线性 boosting 解耦多业务加权，线上 A/B 全指标正向。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (13)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/adaptive_autoguidance_for_item_side_fairness_in_diffusion_recommender_syste/">Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems</a></strong></dt>
<dd>
<p>用可学习逐步引导权重替代固定自引导，配合流行度正则，在扩散推荐中实现物品侧公平与精度近无损的平衡。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/additive_control_variates_dominate_self_normalisation_in_off_policy_evaluat/">Additive Control Variates Dominate Self-Normalisation in Off-Policy Evaluation</a></strong></dt>
<dd>
<p>证明最优加性基线 β*-IPS 在 MSE 上渐近严格优于 SNIPS，并给出精确方差差解析式与排序场景 β*⊥⊥-IPM 扩展。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/equity_vs_equality_optimizing_ranking_fairness_for_tailored_provider_needs/">Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs</a></strong></dt>
<dd>
<p>提出以增益为基的公平框架与 EquityRank 梯度重排算法，按提供者异质偏好分配曝光与销量收益，离线/在线仿真均取得更优效果-公平权衡。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/erase_a_real_world_aligned_benchmark_for_unlearning_in_recommender_systems/">ERASE -- A Real-World Aligned Benchmark for Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>面向真实场景的推荐系统机器遗忘基准，覆盖 CF/SBR/NBR 三任务、七种遗忘算法、九数据集九模型，揭示近似遗忘的鲁棒性与效率短板。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/following_the_eye_tracking_evidence_established_web_search_assumptions_fail/">Following the Eye-Tracking Evidence: Established Web-Search Assumptions Fail in Carousel Interfaces</a></strong></dt>
<dd>
<p>用 RecGaze 眼动数据系统证伪 F 型注意力、层级点击与标题导航等四条 Web 搜索假设，提出轮播界面特有的 L 型注视模式。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/from_top_1_to_top_k_a_reproducibility_study_and_benchmarking_of_counterfact/">From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems</a></strong></dt>
<dd>
<p>统一复现11种反事实解释器，首次提出列表级Top-K评估协议与PN-R位置敏感指标，揭示显式格式下有效性-稀疏性权衡与图解释器可扩展性瓶颈。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/kuailive_a_real_time_interactive_dataset_for_live_streaming_recommendation/">KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation</a></strong></dt>
<dd>
<p>首个含直播间生命周期与多行为实时交互的快手直播推荐数据集，构建 Top-K/CTR 基准并揭示主播长尾公平性挑战。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/post_hoc_provider_fairness_adaptation_via_hierarchical_exposure_alignment/">Post-hoc Provider Fairness Adaptation via Hierarchical Exposure Alignment</a></strong></dt>
<dd>
<p>冻结基座+轻量公平适配器，用分层曝光对齐(HEFA)与可微NDCG联合优化，实现即插即用的提供方曝光公平。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/recnexteval_a_reference_implementation_for_temporal_next_batch_recommendati/">RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation</a></strong></dt>
<dd>
<p>提出滑动时间窗 next-batch 评估框架，强制全局时间线防泄漏并支持增量学习，含 Python 库与 Web UI。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/solaris_speculative_offloading_of_latent_based_representation_for_inference/">SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling</a></strong></dt>
<dd>
<p>借鉴投机解码思想，异步预计算基础模型用户-物品隐表征并分层补全覆盖，实现推理时知识蒸馏，Meta 广告线上收入 +0.67%。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部13篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</div>

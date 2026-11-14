---
hide:
  - toc
---

# 🏛️ SIGIR 2025 推荐系统论文解读

86 篇 SIGIR 2025 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/a_learnable_fully_interacted_two_tower_model_for_pre_ranking_system/">A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System</a></strong></dt>
<dd>
<p>用可学习 Item Meta Matrix 实现显式早期交互、行/列 FC 相似度矩阵实现晚期交互，在保持双塔解耦推理效率下 AUC 相对提升 4.6%–14.3%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/graph_spectral_filtering_with_chebyshev_interpolation_for_recommendation/">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a></strong></dt>
<dd>
<p>用切比雪夫插值逼近非线性图滤波器，直接以原始交互为信号，突破 LightGCN 的截断线性低通瓶颈，Full-Ranking 下全面 SOTA 且推理快 8.1 倍。</p>
</dd>
<dt><strong><a href="collaborative_filtering/ira_adaptive_interest_aware_representation_and_alignment_for_personalized_m/">IRA: Adaptive Interest-aware Representation and Alignment for Personalized Multi-interest Retrieval</a></strong></dt>
<dd>
<p>用可累积/剪枝的文本化 Interest Unit 表征多兴趣，语义对齐替代点击信号做 ANN 召回，离线 H@20 提升 20%，线上点击 +5.4%。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl>
<dt><strong><a href="graph_rec/benchmarking_recommendation_classification_and_tracing_based_on_hugging_fac/">Benchmarking Recommendation, Classification, and Tracing Based on Hugging Face Knowledge Graph</a></strong></dt>
<dd>
<p>构建 260 万节点 HuggingKG 与三任务 HuggingBench，揭示稀疏领域 KG 上 KGRec 类方法失效、TransE 反超复杂模型的关键现象。</p>
</dd>
<dt><strong><a href="graph_rec/comprehending_knowledge_graphs_with_large_language_models_for_recommender_s/">Comprehending Knowledge Graphs with Large Language Models for Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 理解 KG 局部子图并构建语义物品图做全局检索增强，离线预计算语义嵌入后接入 LightGCN，全库排序下全面超越 12 个基线。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/squeeze_and_excitation_a_weighted_graph_contrastive_learning_for_collaborat/">Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>仅扰动末层GCN视图并用SENet动态加权特征，在三个数据集上实现最高24.45%的NDCG@20提升。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (12)</h2>
<dl>
<dt><strong><a href="sequential_rec/ai_guided_accelerator_for_search_experience/">AI Guided Accelerator For Search Experience</a></strong></dt>
<dd>
<p>挖掘电商查询转移链并做意图一致性过滤，再用 Solar-10B 指令微调生成多样化替代收敛查询，线上 CTR +32.2%、转化 +38.3%。</p>
</dd>
<dt><strong><a href="sequential_rec/alphafuse_learn_id_embeddings_for_sequential_recommendation_in_null_space_o/">AlphaFuse: Learn ID Embeddings for Sequential Recommendation in Null Space of Language Embeddings</a></strong></dt>
<dd>
<p>将语言嵌入 SVD 分解为语义丰富行空间与语义稀疏零空间，在裁剪后的零空间内学习 ID 嵌入，无需适配器即插即用地融合语义与协同信号。</p>
</dd>
<dt><strong><a href="sequential_rec/csrec_rethinking_sequential_recommendation_from_a_causal_perspective/">CSRec: Rethinking Sequential Recommendation from A Causal Perspective</a></strong></dt>
<dd>
<p>将序列推荐重构为因果干预下的决策接受概率建模，用递归约束隔离系统影响，AHR@0.2 提升约 1.5–3 倍。</p>
</dd>
<dt><strong><a href="sequential_rec/data_augmentation_as_free_lunch_exploring_the_test_time_augmentation_for_se/">Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation</a></strong></dt>
<dd>
<p>将数据增强从训练期迁移到推理期，提出 TNoise 与 TMask 两个免训练、免参数的测试时增强算子，在 Full-Ranking 协议下平均提升 14%–30%。</p>
</dd>
<dt><strong><a href="sequential_rec/diff_dual_side_information_filtering_and_fusion_for_sequential_recommendati/">DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation</a></strong></dt>
<dd>
<p>频域滤波去噪 + 早期/中间双重融合的侧信息序列推荐，Full-Ranking 下 R@20 最高提升 14.1%</p>
</dd>
<dt><strong><a href="sequential_rec/fim_frequency_aware_multi_view_interest_modeling_for_local_life_service_rec/">FIM: Frequency-Aware Multi-View Interest Modeling for Local-Life Service Recommendation</a></strong></dt>
<dd>
<p>多视图搜索分离本地生活多周期需求，FFT 频域感知演化建模周期兴趣，Kuaishou 线上 GMV +8.46%。</p>
</dd>
<dt><strong><a href="sequential_rec/heterrec_heterogeneous_information_transformer_for_scalable_sequential_reco/">HeterRec: Heterogeneous Information Transformer for Scalable Sequential Recommendation</a></strong></dt>
<dd>
<p>将物品拆解为异构特征 token 序列并设计层级因果 Transformer 与列表级多步预测损失，在工业与 Amazon 数据上 Recall@100 提升 3.8%–4.2%。</p>
</dd>
<dt><strong><a href="sequential_rec/improving_sequential_recommenders_through_counterfactual_augmentation_of_sy/">Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure</a></strong></dt>
<dd>
<p>用 Decision Transformer 建模全量曝光序列，并以随机/自改进反事实增广+用户模拟器挖掘未曝光兴趣，KuaiRand 上 Recall@20 相对提升约 87%。</p>
</dd>
<dt><strong><a href="sequential_rec/intent_aware_diffusion_with_contrastive_learning_for_sequential_recommendat/">Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 K-means 意图原型引导条件扩散生成意图对齐正视图替代随机增强，在 5 个数据集 Full-Ranking 下 NDCG 平均提升 20.68%。</p>
</dd>
<dt><strong><a href="sequential_rec/linear_item_item_model_with_neural_knowledge_for_session_based_recommendati/">Linear Item-Item Model with Neural Knowledge for Session-based Recommendation</a></strong></dt>
<dd>
<p>将神经会话模型的转移知识蒸馏为物品-物品矩阵正则项，与线性自蒸馏相似度统一闭式求解，R@20 最高提升 14.78% 且推理 FLOPs 降低 813 倍。</p>
</dd>
</dl>
<p><a href="sequential_rec/">查看全部12篇「序列与会话推荐」论文 →</a></p>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (4)</h2>
<dl>
<dt><strong><a href="generative_rec/addressing_missing_data_issue_for_diffusion_based_recommendation/">Addressing Missing Data Issue for Diffusion-based Recommendation</a></strong></dt>
<dd>
<p>用双侧 Thompson 采样在引导序列中模拟缺失数据，以一致性正则化让扩散推荐模型外推抵抗真实缺失，YooChoose NDCG@20 提升 26.95%。</p>
</dd>
<dt><strong><a href="generative_rec/generative_recommender_with_end_to_end_learnable_item_tokenization/">Generative Recommender with End-to-End Learnable Item Tokenization</a></strong></dt>
<dd>
<p>提出 ETEGRec 双编码器-解码器框架，用序列-物品对齐与偏好-语义对齐将 RQ-VAE 分词器与 T5 推荐器端到端联合优化，全库排序下 Recall@10 相对 LETTER 提升约 5%。</p>
</dd>
<dt><strong><a href="generative_rec/killing_two_birds_with_one_stone_unifying_retrieval_and_ranking_with_a_sing/">Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model</a></strong></dt>
<dd>
<p>将召回与排序统一为单生成模型的位置级序列生成任务，配合排序驱动增强器与梯度自适应加权实现协同与同步优化，全库排序下召回排序双提升。</p>
</dd>
<dt><strong><a href="generative_rec/order_agnostic_identifier_for_large_language_model_based_generative_recomme/">Order-agnostic Identifier for Large Language Model-based Generative Recommendation</a></strong></dt>
<dd>
<p>提出无序集合标识符范式，用CF+语义多token集合表示物品并同时生成，T5上Recall@10平均提升约50%，推理提速8-18倍。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (14)</h2>
<dl>
<dt><strong><a href="llm_rec/adore_autonomous_domain_oriented_relevance_engine_for_e_commerce/">ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce</a></strong></dt>
<dd>
<p>用 CoT-LLM 自动挖掘难样本、KTO 对齐用户行为、错误类型感知合成对抗样本，再以关键属性显式蒸馏到轻量 BERT，离线 F1 85.45%、线上 CTR +1.24%。</p>
</dd>
<dt><strong><a href="llm_rec/corona_a_coarse_to_fine_framework_for_graph_based_recommendation_with_large/">CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 做偏好/意图推理在交互图上逐级缩小候选子图，再由 GNN 精排，全库排序下 Recall 平均相对提升 18.6%。</p>
</dd>
<dt><strong><a href="llm_rec/do_llms_memorize_recommendation_datasets_a_preliminary_study_on_movielens_1/">Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M</a></strong></dt>
<dd>
<p>用少样本提示探测 GPT/Llama 对 MovieLens-1M 的物品、用户与交互记忆覆盖率，揭示记忆量与推荐性能强相关且放大流行度偏置。</p>
</dd>
<dt><strong><a href="llm_rec/efficiency_unleashed_inference_acceleration_for_llm_based_recommender_syste/">Efficiency Unleashed: Inference Acceleration for LLM-based Recommender Systems with Speculative Decoding</a></strong></dt>
<dd>
<p>将检索式推测解码引入 LLM 离线知识生成，用定制检索池与松弛验证实现 3-5× 无损加速，线上省 67% 算力。</p>
</dd>
<dt><strong><a href="llm_rec/improving_llm_powered_recommendations_with_personalized_information/">Improving LLM-powered Recommendations with Personalized Information</a></strong></dt>
<dd>
<p>将用户偏好与物品感知两条 CoT 显式抽取并注入 CRM 召回与 LLM 重排，Leave-One-Out 全库排序下 Hit@10 最高提升约 11%，并显著降低位置偏差。</p>
</dd>
<dt><strong><a href="llm_rec/intent_representation_learning_with_large_language_model_for_recommendation/">Intent Representation Learning with Large Language Model for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 思维链抽取文本意图，与交互意图双塔对齐+动量蒸馏匹配，作为模型无关插件在三个数据集上稳定提升 Recall/NDCG。</p>
</dd>
<dt><strong><a href="llm_rec/large_language_models_enhanced_hyperbolic_space_recommender_systems/">Large Language Models Enhanced Hyperbolic Space Recommender Systems</a></strong></dt>
<dd>
<p>HyperLLM 用 LLM 生成层级标签与偏好摘要，经元优化两阶段训练将结构/语义层级信息注入双曲推荐，Recall@20 最高提升 41.7%。</p>
</dd>
<dt><strong><a href="llm_rec/msl_not_all_tokens_are_what_you_need_for_tuning_llm_as_a_recommender/">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></strong></dt>
<dd>
<p>提出掩码 Softmax 损失屏蔽虚构物品 token，并配自适应温度策略，四数据集 NDCG@10 平均提升 42.24%。</p>
</dd>
<dt><strong><a href="llm_rec/multi_grained_patch_training_for_efficient_llm_based_recommendation/">Multi-Grained Patch Training for Efficient LLM-based Recommendation</a></strong></dt>
<dd>
<p>用层次化 item/session patch 压缩长交互历史，两阶段训练让 LLM 理解压缩表征，同 token 预算下 HR@20 最高提升 72%。</p>
</dd>
<dt><strong><a href="llm_rec/pre_train_align_and_disentangle_empowering_sequential_recommendation_with_l/">Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 MK-MMD 特征核做推荐锚定对齐，再以三专家频率门控解耦，Full-Ranking 下冷物品 HR@10 提升最高 462%。</p>
</dd>
</dl>
<p><a href="llm_rec/">查看全部14篇「大语言模型推荐 (LLM4Rec)」论文 →</a></p>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (7)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/agentcf_memory_enhanced_llm_based_agents_for_popularity_aware_cross_domain_/">AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations</a></strong></dt>
<dd>
<p>以双层记忆+两步融合+兴趣组共享记忆，令 LLM 用户智能体在跨域场景显式建模流行度影响，采样排序 MRR 平均超 AgentCF 14.6%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/agentic_feedback_loop_modeling_improves_recommendation_and_user_simulation/">Agentic Feedback Loop Modeling Improves Recommendation and User Simulation</a></strong></dt>
<dd>
<p>用记忆驱动的推荐智能体-用户智能体多轮反馈闭环，同时提升下一物品推荐与用户行为模拟，并验证不放大流行度/位置偏差。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/mscrs_multi_modal_semantic_graph_prompt_learning_framework_for_conversation/">MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>构建协同/文本/图像三模态语义图并以共享嵌入融合，图提示注入 DialoGPT，ReDial 与 INSPIRED 全指标 SOTA。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/pscon_product_search_through_conversations/">PSCon: Product Search Through Conversations</a></strong></dt>
<dd>
<p>提出人-人教练式采集协议构建双市场双语真实 CPS 数据集 PSCon，定义六子任务流水线并给出 Transformer 统一基准模型。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/pub_an_llm_enhanced_personality_driven_user_behaviour_simulator_for_recomme/">PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation</a></strong></dt>
<dd>
<p>用大五人格驱动 LLM Agent 生成合成交互日志，Jaccard 相似度 0.31 逼近真实序列，并复现 7 种推荐算法的 nDCG 排序趋势。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/search_based_interaction_for_conversation_recommendation_via_generative_rew/">Search-Based Interaction For Conversation Recommendation via Generative Reward Model Based Simulated User</a></strong></dt>
<dd>
<p>用生成式奖励模型构建无标签模拟用户，通过属性评论与生成式打分驱动束搜索交互，INSPIRED 上 Recall@10 相对最优基线提升 83%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/simulating_before_planning_constructing_intrinsic_user_world_model_for_user/">Simulating Before Planning: Constructing Intrinsic User World Model for User-Tailored Dialogue Policy Planning</a></strong></dt>
<dd>
<p>用扩散模型刻画用户画像、布朗桥预判用户反馈，构建内在用户世界模型以定制对话策略，P4G/ESConv 上 SR 显著提升。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (6)</h2>
<dl>
<dt><strong><a href="multimodal_rec/cohesion_composite_graph_convolutional_network_with_dual_stage_fusion_for_m/">COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用行为模态早期精炼+晚期注意力融合的双阶段融合，配合异质/同质复合图与自适应BPR，在三个 Amazon 数据集上 R@20 相对提升 4.4%–6.1%。</p>
</dd>
<dt><strong><a href="multimodal_rec/diffusion_based_multi_modal_synergy_interest_network_for_click_through_rate/">Diffusion-based Multi-modal Synergy Interest Network for Click-through Rate Prediction</a></strong></dt>
<dd>
<p>用 PLE 式共享/独有专家解耦模态知识、以扩散式多步跨模态去噪捕捉协同信号，FDAF 非侵入融合，四数据集 AUC 全面领先。</p>
</dd>
<dt><strong><a href="multimodal_rec/disentangling_and_generating_modalities_for_recommendation_in_missing_modal/">Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios</a></strong></dt>
<dd>
<p>将模态特征解耦为通用/特有分量并用互信息损失分离，再以跨模态对齐特征与用户模态偏好生成缺失模态，缺失场景全库排序 R@20 相对提升 4.7%–8.2%。</p>
</dd>
<dt><strong><a href="multimodal_rec/multi_modal_multi_behavior_sequential_recommendation_with_conditional_diffu/">Multi-Modal Multi-Behavior Sequential Recommendation with Conditional Diffusion-Based Feature Denoising</a></strong></dt>
<dd>
<p>用条件扩散分别对模态与行为特征去噪，并以多专家网络解耦共性/特性兴趣，在 Rec-Tmal/Kuaishou 上 HR@10 提升约 4-8%。</p>
</dd>
<dt><strong><a href="multimodal_rec/short_video_segment_level_user_dynamic_interests_modeling_in_personalized_r/">Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation</a></strong></dt>
<dd>
<p>将短视频切分为5秒片段，用CLIP视觉+ID双模态跨注意力建模片段级动态兴趣，以段内排序损失解决标签缺失，在跳看预测与视频推荐两任务上大幅超越基线。</p>
</dd>
<dt><strong><a href="multimodal_rec/u_sticker_a_large_scale_multi_domain_user_sticker_dataset_for_retrieval_and/">U-Sticker: A Large-Scale Multi-Domain User Sticker Dataset for Retrieval and Personalization</a></strong></dt>
<dd>
<p>构建含 22K 用户、370K 贴纸、8.3M 消息的多领域多语言贴纸数据集，支撑个性化贴纸检索与用户建模。</p>
</dd>
</dl>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl>
<dt><strong><a href="social_group_rec/frog_effective_friend_recommendation_in_online_games_via_modality_aware_use/">FROG: Effective Friend Recommendation in Online Games via Modality-aware User Preferences</a></strong></dt>
<dd>
<p>面向在线游戏好友推荐，用模态感知双向匹配+局部/全局双路注意力建模用户偏好，离线 HR/NDCG 最高提升 15.82%/14.59%，线上 accRate 提升 33.3%。</p>
</dd>
<dt><strong><a href="social_group_rec/invariance_matters_empowering_social_recommendation_via_graph_invariant_lea/">Invariance Matters: Empowering Social Recommendation via Graph Invariant Learning</a></strong></dt>
<dd>
<p>用图不变学习替代社交图硬过滤：多环境生成器+不变风险最小化+对抗环境探索，Full-Ranking 下 NDCG@20 较 GBSR 提升 5.4%–8.7%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (6)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/adaptive_domain_scaling_for_personalized_sequential_modeling_in_recommender/">Adaptive Domain Scaling for Personalized Sequential Modeling in Recommenders</a></strong></dt>
<dd>
<p>用域特征动态生成序列物品的个性化MLP权重与候选物品的多查询表示，改造目标注意力实现多域序列建模，抖音双场景线上GMV+0.79%、ADVV+1.00%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/adaptive_graph_integration_for_cross_domain_recommendation_via_heterogeneou/">Adaptive Graph Integration for Cross-Domain Recommendation via Heterogeneous Graph Coordinators</a></strong></dt>
<dd>
<p>用异质自适应图协调器把多域图拼成统一结构做多域预训练，再以图提示迁移到目标域，Full-Ranking 下全面超越 9 个基线。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/bridge_the_domains_large_language_models_enhanced_cross_domain_sequential_r/">Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义嵌入+分层画像桥接跨域物品与用户偏好，破解重叠困境与转移复杂性，三数据集 H@10 最高提升 31.7%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/cdc_causal_domain_clustering_for_multi_domain_recommendation/">CDC: Causal Domain Clustering for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用孤立/混合双亲和矩阵+因果发现系数自适应融合，联合优化目标域簇与训练源域集，50+域上DomainAUC提升4.14‰/6.76‰，线上eCPM+4.9%</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/learning_universal_user_representations_leveraging_cross_domain_user_intent/">Learning Universal User Representations Leveraging Cross-domain User Intent at Snapchat</a></strong></dt>
<dd>
<p>将 Snapchat 多业务面事件拼接为统一序列训练 Transformer 用户塔，产出跨域通用用户表征，线上多场景 A/B 全面正向。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/x_cross_dynamic_integration_of_language_models_for_cross_domain_sequential_/">X-Cross: Dynamic Integration of Language Models for Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>逐层动态集成多个 LoRA 源域语言模型，仅用 25% 参数与 50-75% 更少微调数据即达/超 LoRA 跨域序列推荐效果。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/comapoi_a_collaborative_multi_agent_framework_for_next_poi_prediction_bridg/">CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction Bridging the Gap Between Trajectory and Language</a></strong></dt>
<dd>
<p>用 Profiler/Forecaster/Predictor 三智能体把数值轨迹转语言画像、动态收缩候选集并反向推理微调，三数据集全指标提升 4%–12%。</p>
</dd>
<dt><strong><a href="poi_spatiotemporal_rec/sep_gcn_leveraging_similar_edge_pairs_with_temporal_and_spatial_contexts_fo/">SEP-GCN: Leveraging Similar Edge Pairs with Temporal and Spatial Contexts for Location-Based Recommender Systems</a></strong></dt>
<dd>
<p>以时空相似交互边对构建边级相似图并注入LightGCN传播，在NYC/Gowalla/Brightkite全库排序上Recall@20最高提升14.89%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (8)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/deep_multiple_quantization_network_on_long_behavior_sequence_for_click_thro/">Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>用多码本量化将超长行为序列端到端压缩为约百个兴趣簇码字，配合 HSTU 簇间交互与候选感知注意力，消除两阶段检索的相关性分布偏差，线上 CTR +3.5%、RPM +2.0%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/dlf_enhancing_explicit_implicit_interaction_via_dynamic_low_order_aware_fus/">DLF: Enhancing Explicit-Implicit Interaction via Dynamic Low-Order-Aware Fusion for CTR Prediction</a></strong></dt>
<dd>
<p>以残差感知低阶交互网络与逐层注意力融合，动态平衡显隐式特征交叉，四数据集AUC全面SOTA。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/elec_efficient_large_language_model_empowered_click_through_rate_prediction/">ELEC: Efficient Large Language Model-Empowered Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>伪孪生增益/朴素双网络+分数级CLID与表示级MSE双蒸馏，在线仅跑表格DCNv2，AUC较FLIP提升0.0020/0.0034。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/hybrid_advertising_in_the_sponsored_search/">Hybrid Advertising in the Sponsored Search</a></strong></dt>
<dd>
<p>提出混合广告拍卖模型与 HRegNet 神经网络机制，在近 DSIC/IR 约束下联合优化独立店铺与店铺-品牌捆绑包的广告位分配与定价，真实数据上收益显著超越 VCG 与 JRegNet。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/interest_changes_considering_user_interest_life_cycle_in_recommendation_sys/">Interest Changes: Considering User Interest Life Cycle in Recommendation System</a></strong></dt>
<dd>
<p>用曝光/点击/交互直方图+VQ聚类刻画兴趣生命周期，经特征重校准与专家网络融合注入MMOE精排，Lofter线上CTR+0.38%、CVR+1.04%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/lrea_low_rank_efficient_attention_on_modeling_long_term_user_behaviors_for_/">LREA: Low-Rank Efficient Attention on Modeling Long-Term User Behaviors for CTR Prediction</a></strong></dt>
<dd>
<p>用低秩分解+矩阵吸收把万级长行为序列压到秩 r 维做目标注意力，配非负正则保非线性，线上仅增 3ms 延迟、CTR +5.88%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/multi_objective_aligned_bidword_generation_model_for_e_commerce_search_adve/">Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising</a></strong></dt>
<dd>
<p>用判别器提供相关性/真实性/CPM三维奖励，扩展MODPO做多目标偏好对齐，驱动LLM生成高商业价值竞价词，线上广告收入+2.13%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/nam_a_normalization_attention_model_for_personalized_product_search_in_flig/">NAM: A Normalization Attention Model for Personalized Product Search In Fliggy</a></strong></dt>
<dd>
<p>用逆物品频率(IIF)对注意力做全局归一化并门控个性化强度，解决热门物品高估/长尾低估，离线CTCVR AUC+0.001、线上CTCVR+0.8%。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (4)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/comprehensive_list_generation_for_multi_generator_reranking/">Comprehensive List Generation for Multi-Generator Reranking</a></strong></dt>
<dd>
<p>提出多生成器-评估器重排框架与列表全面性指标，用 CLIG 自动学习互补生成器，离线 AUC 提升 0.9%、线上观看时长 +0.48%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/csmf_cascaded_selective_mask_fine_tuning_for_multi_objective_embedding_base/">CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval</a></strong></dt>
<dd>
<p>借鉴 PEFT 选择性掩码思想，按曝光→点击→转化级联顺序分阶段微调双塔召回，零参数增量实现多目标融合检索。</p>
</dd>
<dt><strong><a href="multi_task_reranking/nr4der_neural_re_ranking_for_diversified_exercise_recommendation/">NR4DER: Neural Re-ranking for Diversified Exercise Recommendation</a></strong></dt>
<dd>
<p>用 mLSTM 难度过滤 + 活跃学生序列增强 + 神经重排，兼顾长尾学生与个性化学习节奏的习题推荐。</p>
</dd>
<dt><strong><a href="multi_task_reranking/towards_principled_learning_for_re_ranking_in_recommender_systems/">Towards Principled Learning for Re-ranking in Recommender Systems</a></strong></dt>
<dd>
<p>提出收敛一致性与对抗一致性两条重排学习原则，以自洽重排与相邻换位扰动构造额外监督，即插即用地提升多种重排基线。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/darlr_dual_agent_offline_reinforcement_learning_for_recommender_systems_wit/">DARLR: Dual-Agent Offline Reinforcement Learning for Recommender Systems with Dynamic Reward</a></strong></dt>
<dd>
<p>双智能体离线 RL 用选择器聚合相似用户动态重塑奖励与不确定性惩罚，四数据集累积奖励全面超越 ROLeR/DORL。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/enhancing_new_item_fairness_in_dynamic_recommender_systems/">Enhancing New-item Fairness in Dynamic Recommender Systems</a></strong></dt>
<dd>
<p>用 DQN 智能体 + 三重奖励（探索/公平/准确）在动态推荐系统中提升新物品曝光公平性，KuaiRec 上 TGF 降 86%+ 且 HR 反升 39%。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (13)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/a_worrying_reproducibility_study_of_intent_aware_recommendation_models/">A Worrying Reproducibility Study of Intent-Aware Recommendation Models</a></strong></dt>
<dd>
<p>复现 5 个意图感知推荐模型，发现 2 个不可复现、全部被调优后的传统非神经基线超越，揭示方法学顽疾。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/datarec_a_python_library_for_standardized_and_reproducible_data_management_/">DataRec: A Python Library for Standardized and Reproducible Data Management in Recommender Systems</a></strong></dt>
<dd>
<p>基于 55 篇顶会论文调研构建的统一数据管理库，提供可复现的过滤/划分/版本校验与跨框架导出，覆盖 18 个内置数据集。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/diversity_aware_dual_promotion_poisoning_attack_on_sequential_recommendatio/">Diversity-aware Dual-promotion Poisoning Attack on Sequential Recommendation</a></strong></dt>
<dd>
<p>揭示攻击与推荐目标梯度冲突，提出双提升MSE攻击目标与多样性重排序列生成，在保推荐精度的同时显著提升目标物品曝光。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/exploring_the_escalation_of_source_bias_in_user_data_and_recommender_system/">Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop</a></strong></dt>
<dd>
<p>揭示 AIGC 在用户-数据-系统反馈回路中被逐步放大的来源偏见，并提出基于 L1 损失的黑盒去偏方法维持内容生态平衡。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/extending_movielens_32m_to_provide_new_evaluation_objectives/">Extending MovieLens-32M to Provide New Evaluation Objectives</a></strong></dt>
<dd>
<p>用 TREC 池化法为 ML-32M 构建 51 用户偏好型 qrels，揭示传统 train/test 评测的流行度偏置并给出新评测目标。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/id_free_not_risk_free_llm_powered_agents_unveil_risks_in_id_free_recommende/">ID-Free Not Risk-Free: LLM-Powered Agents Unveil Risks in ID-Free Recommender Systems</a></strong></dt>
<dd>
<p>揭示无ID推荐器可被LLM多智能体改写文本攻击，提出TextSimu攻击与RewriteDetection检测，Hit@50最高提升数倍。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/llm_generated_fake_news_induces_truth_decay_in_news_ecosystem_a_case_study_/">LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation</a></strong></dt>
<dd>
<p>构建约 7.5 万条人机混合新闻推荐数据集，模拟 LLM 生成假新闻在候选、历史、训练集三阶段渗透，揭示真实新闻排序优势被侵蚀的 Truth Decay 现象并以困惑度给出解释。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/pre_training_for_recommendation_unlearning/">Pre-training for Recommendation Unlearning</a></strong></dt>
<dd>
<p>提出预训练式影响编码器 UnlearnRec，将遗忘请求映射为嵌入漂移，免重训实现高效图推荐遗忘，10x 加速且保持精度。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/recgaze_the_first_eye_tracking_and_user_interaction_dataset_for_carousel_in/">RecGaze: The First Eye Tracking and User Interaction Dataset for Carousel Interfaces</a></strong></dt>
<dd>
<p>首个轮播界面眼动+点击+光标+选择解释数据集，87用户3477次交互，揭示F型浏览与位置偏置。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/revisiting_algorithmic_audits_of_tiktok_poor_reproducibility_and_short_term/">Revisiting Algorithmic Audits of TikTok: Poor Reproducibility and Short-term Validity of Findings</a></strong></dt>
<dd>
<p>复现 TikTok 推荐审计发现可复现性极差、结论仅短期有效，观看行为才是最强个性化信号，且结论高度依赖度量选择。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部13篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</div>

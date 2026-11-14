---
hide:
  - toc
---

# 🏛️ KDD 2025 推荐系统论文解读

73 篇 KDD 2025 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (7)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/embed_progressive_implicit_preference_in_unified_space_for_deep_collaborati/">Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering</a></strong></dt>
<dd>
<p>将多类隐式反馈按稀疏度映射为有序类别，用嵌套编码+嵌套OLR把渐进偏好压入统一空间，实现单次NNS一阶段排序。</p>
</dd>
<dt><strong><a href="collaborative_filtering/high_precision_audience_expansion_via_extreme_classification_in_a_two_sided/">High Precision Audience Expansion via Extreme Classification in a Two-Sided Marketplace</a></strong></dt>
<dd>
<p>将 Airbnb 位置召回从矩形边界重构为 25M 级 S2 单元极端多分类，离线精度 +11.01%、线上未取消预订者 +0.17%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/iu4rec_interest_unit_based_product_organization_and_recommendation_for_e_co/">IU4Rec: Interest Unit-Based Product Organization and Recommendation for E-Commerce Platform</a></strong></dt>
<dd>
<p>闲鱼C2C限库存场景下以兴趣单元(IU)重构商品组织与两阶段推荐，IU-Boosted网络离线AUC+1.91%、线上CTR+2.90%</p>
</dd>
<dt><strong><a href="collaborative_filtering/on_the_role_of_weight_decay_in_collaborative_filtering_a_popularity_perspec/">On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective</a></strong></dt>
<dd>
<p>揭示权重衰减本质是将流行度编码进嵌入模长，提出 PRISM 初始化策略替代权重衰减，NDCG 最高提升 4.77% 且训练加速 38.48%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/optimizing_recall_or_relevance_a_multi_task_multi_head_approach_for_item_to/">Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for Item-to-Item Retrieval in Recommendation</a></strong></dt>
<dd>
<p>用多任务蒸馏损失+多头I2I架构联合优化共互动召回与语义相关性，离线召回@500提升14.4%、L2主题相关性提升56.5%，并完成线上A/B验证。</p>
</dd>
<dt><strong><a href="collaborative_filtering/real_time_indexing_for_large_scale_recommendation_by_streaming_vector_quant/">Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever</a></strong></dt>
<dd>
<p>用流式 VQ 码本替代 HNSW/DR 静态索引，实现实时挂载、索引均衡与可修复，抖音全量上线显著提升用户参与。</p>
</dd>
<dt><strong><a href="collaborative_filtering/synergizing_implicit_and_explicit_user_interests_a_multi_embedding_retrieva/">Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest</a></strong></dt>
<dd>
<p>Pinterest 工业级多嵌入召回框架，用可微聚类建模隐式兴趣、条件召回建模显式兴趣，线上 HF Repins +1.09%、A-Pincepts +0.81%。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl>
<dt><strong><a href="graph_rec/an_automatic_graph_construction_framework_based_on_large_language_models_fo/">An Automatic Graph Construction Framework based on Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 逐点生成语义向量 + 残差量化抽取隐因子作额外节点，O(N) 构图并设计元路径聚合，Full-Ranking 下 NDCG@10 最高提升 43.57%，华为广告 A/B 提升 RPM 2.69%。</p>
</dd>
<dt><strong><a href="graph_rec/collaborative_filtering_meets_spectrum_shift_connecting_user_item_interacti/">Collaborative Filtering Meets Spectrum Shift: Connecting User-Item Interaction with Graph-Structured Side Information</a></strong></dt>
<dd>
<p>揭示增广图频谱右移现象，提出仅含平移/缩放两因子的 SSC 校正，即插即用提升 LightGCN/JGCF 最高 23%，零额外开销。</p>
</dd>
<dt><strong><a href="graph_rec/lightkg_efficient_knowledge_aware_recommendations_with_simplified_gnn_archi/">LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture</a></strong></dt>
<dd>
<p>用标量对编码关系+线性聚合+免子图对比层，在稀疏场景超越12个KGRS基线5.8%并省84.3%训练时间</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/enhancing_graph_contrastive_learning_with_reliable_and_informative_augmenta/">Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation</a></strong></dt>
<dd>
<p>用端到端多级向量量化学习离散码，将码作为虚拟邻居与语义正样本构建三重对比视图，Full-Ranking 下 NDCG@10 最高提升 17.5%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/omnisage_large_scale_multi_entity_heterogeneous_graph_representation_learni/">OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning</a></strong></dt>
<dd>
<p>统一异构图 GNN、内容特征与用户序列的多任务对比学习框架，支撑 Pinterest 十亿级实体通用表征，线上全站 Repin 提升约 2.5%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/shapley_value_driven_data_pruning_for_recommender_systems/">Shapley Value-driven Data Pruning for Recommender Systems</a></strong></dt>
<dd>
<p>用 FastSHAP 实时估计交互对训练损失的边际贡献，按 Shapley 值剪枝低效用交互并重训，四数据集 Recall/NDCG 平均提升 3.72%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (7)</h2>
<dl>
<dt><strong><a href="sequential_rec/bridging_textual_collaborative_gap_through_semantic_codes_for_sequential_re/">Bridging Textual-Collaborative Gap through Semantic Codes for Sequential Recommendation</a></strong></dt>
<dd>
<p>用多视图文本量化语义码作桥梁，以码为 Query 交叉注意力融合文本，配合双重掩码自监督，Full-Ranking 下 R@10 最高提升 20.1%。</p>
</dd>
<dt><strong><a href="sequential_rec/dv365_extremely_long_user_history_modeling_at_instagram/">DV365: Extremely Long User History Modeling at Instagram</a></strong></dt>
<dd>
<p>以离线用户嵌入替代端到端长序列建模，用多切片+漏斗摘要将 4 万~7 万条历史压缩为 58×256 嵌入，15 个下游模型 NE 平均提升 0.4%+。</p>
</dd>
<dt><strong><a href="sequential_rec/glint_ru_gated_lightweight_intelligent_recurrent_units_for_sequential_recom/">GLINT-RU: Gated Lightweight Intelligent Recurrent Units for Sequential Recommender Systems</a></strong></dt>
<dd>
<p>单层并行专家混合（密集选择GRU+线性注意力）+门控MLP的高效序列推荐，Leave-One-Out全库排序下精度与推理速度双超SOTA。</p>
</dd>
<dt><strong><a href="sequential_rec/multi_grained_preference_enhanced_transformer_for_multi_behavior_sequential/">Multi-Grained Preference Enhanced Transformer for Multi-Behavior Sequential Recommendation</a></strong></dt>
<dd>
<p>M-GPT 用交互级图卷积建模跨类型行为依赖，并以多粒度多尺度注意力抽取行为感知偏好，Leave-One-Out 全库排序下 HR@5 相对提升 12.2%。</p>
</dd>
<dt><strong><a href="sequential_rec/time_matters_enhancing_sequential_recommendations_with_time_guided_graph_ne/">Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs</a></strong></dt>
<dd>
<p>用时间引导扩散补全稀疏用户时间图、图神经ODE对齐用户兴趣与物品演化，五数据集Recall@20最高提升46%。</p>
</dd>
<dt><strong><a href="sequential_rec/unlocking_the_power_of_diffusion_models_in_sequential_recommendation_a_simp/">Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach</a></strong></dt>
<dd>
<p>揭示扩散序列推荐中的嵌入坍缩问题，提出逐 token 独立扩散+三阶段训练，六数据集 HR@20 平均提升 15.45%。</p>
</dd>
<dt><strong><a href="sequential_rec/why_not_together_a_multiple_round_recommender_system_for_queries_and_items/">Why Not Together? A Multiple-Round Recommender System for Queries and Items</a></strong></dt>
<dd>
<p>提出 MAGUS 多轮查询-物品联合推荐框架，以词组合关系图桥接查询与物品，用标签传播在线吸收反馈，可插拔接入 12 种基座模型。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (4)</h2>
<dl>
<dt><strong><a href="generative_rec/controlling_diversity_at_inference_guiding_diffusion_recommender_models_wit/">Controlling Diversity at Inference: Guiding Diffusion Recommender Models with Targeted Category Preferences</a></strong></dt>
<dd>
<p>提出解耦扩散模型 D3Rec，通过类别偏好引导的反向去噪过程，在推理阶段灵活控制推荐准确率-多样性权衡。</p>
</dd>
<dt><strong><a href="generative_rec/flow_matching_for_collaborative_filtering/">Flow Matching for Collaborative Filtering</a></strong></dt>
<dd>
<p>用行为引导先验与离散流匹配重构生成式CF，全库排序下精度SOTA且仅需2步推理。</p>
</dd>
<dt><strong><a href="generative_rec/generating_long_semantic_ids_in_parallel_for_recommendation/">Generating Long Semantic IDs in Parallel for Recommendation</a></strong></dt>
<dd>
<p>提出 RPG 用多 token 并行预测 + 图约束解码，将语义 ID 长度扩至 64，NDCG@10 平均超生成式基线 12.6% 且推理更省。</p>
</dd>
<dt><strong><a href="generative_rec/great_guiding_query_generation_with_a_trie_for_recommending_related_search_/">GREAT: Guiding Query Generation with a Trie for Recommending Related Search about Video at Kuaishou</a></strong></dt>
<dd>
<p>用高质量查询构建 Trie 约束 LLM 训练与解码，实现短视频到查询的 I2Q 推荐，离线 Edit@1 最优、线上 CTR 与相关性双升。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (10)</h2>
<dl>
<dt><strong><a href="llm_rec/agent0_leveraging_llm_agents_to_discover_multi_value_features_from_text_for/">Agent0: Leveraging LLM Agents to Discover Multi-value Features from Text for Enhanced Recommendations</a></strong></dt>
<dd>
<p>用 Architect-Sentinel-Oracle 三角色 LLM 智能体闭环自动挖掘文本多值类别特征，以 DCNv2+RIG 为 Oracle 在真实广告生产数据上取得 0.005+ RIG 提升并提速 2-3 倍。</p>
</dd>
<dt><strong><a href="llm_rec/collaboration_of_large_language_models_and_small_recommendation_models_for_/">Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation</a></strong></dt>
<dd>
<p>提出端云协同 LSC4Rec：云端 LLM 召回+设备端 SRM 实时重排，配合协同训练、分数融合与不一致度触发请求，Beauty 平均提升 16.18%。</p>
</dd>
<dt><strong><a href="llm_rec/earn_efficient_inference_acceleration_for_llm_based_generative_recommendati/">EARN: Efficient Inference Acceleration for LLM-based Generative Recommendation by Register Tokens</a></strong></dt>
<dd>
<p>基于 LLMRec 注意力层间稀疏反转与双注意力汇现象，用首尾寄存器 token 压缩历史交互并在第 k 层后剪除 prompt，实现 3.79× 加速与 80.8% KV Cache 缩减且精度反超微调。</p>
</dd>
<dt><strong><a href="llm_rec/exploring_feature_based_knowledge_distillation_for_recommender_system_a_fre/">Exploring Feature-based Knowledge Distillation for Recommender System: A Frequency Perspective</a></strong></dt>
<dd>
<p>从频域视角揭示特征蒸馏等权缺陷，提出轻量图滤波重加权 FreqD，全库排序下 Recall@20 最高提升 18.2%。</p>
</dd>
<dt><strong><a href="llm_rec/goracs_group_level_optimal_transport_guided_coreset_selection_for_llm_based/">GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>用最优传输+梯度范数构造测试损失上界代理目标，配合贪心初始化-交换精炼两阶段算法做组级核心集选择，20%训练开销超越全量微调。</p>
</dd>
<dt><strong><a href="llm_rec/lettingo_explore_user_profile_generation_for_recommendation_system/">LettinGo: Explore User Profile Generation for Recommendation System</a></strong></dt>
<dd>
<p>用多LLM探索开放式用户画像、以下游推荐成败构造偏好对、DPO对齐画像生成器，Amazon Books Acc 66.3%→70.4%。</p>
</dd>
<dt><strong><a href="llm_rec/llm2rec_large_language_models_are_powerful_embedding_models_for_sequential_/">LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation</a></strong></dt>
<dd>
<p>两阶段将 LLM 改造为融合协同信号与语义的物品嵌入模型，Full-Ranking 下域内平均提升约 15%、跨域仍稳定增益。</p>
</dd>
<dt><strong><a href="llm_rec/lost_in_sequence_do_large_language_models_understand_sequential_recommendat/">Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?</a></strong></dt>
<dd>
<p>揭示现有 LLM4Rec 模型无法捕捉交互序列顺序信息，提出冻结 LLM 与 CF-SRec、仅训练轻量 MLP 的蒸馏框架 LLM-SRec，在 Amazon 四数据集上 NDCG@10 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="llm_rec/the_mental_world_of_large_language_models_in_recommendation_a_benchmark_on_/">The Mental World of Large Language Models in Recommendation: A Benchmark on Association, Personalization, and Knowledgeability</a></strong></dt>
<dd>
<p>提出 LRWorld 基准（38K 样本/23M tokens），从关联、个性化、知识性三尺度十因素 31 任务系统评测数十个 LLM，揭示其在神经嵌入检索上的致命短板。</p>
</dd>
<dt><strong><a href="llm_rec/uqabench_evaluating_user_embedding_for_prompting_llms_in_personalized_quest/">UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering</a></strong></dt>
<dd>
<p>提出首个标准化三阶段（预训练-微调-评测）用户嵌入软提示基准，用序列理解/行为预测/兴趣感知三类任务揭示嵌入式生成推荐与文本式相当且省 8-19× token。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/lavic_adapting_large_vision_language_models_to_visually_aware_conversationa/">LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation</a></strong></dt>
<dd>
<p>用视觉知识自蒸馏把商品图压成5个[CLS] token，再做推荐提示微调，在新建Reddit-Amazon三域上HR@1最高提升54.2%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/leveraging_the_power_of_conversations_optimal_key_term_selection_in_convers/">Leveraging the Power of Conversations: Optimal Key Term Selection in Conversational Contextual Bandits</a></strong></dt>
<dd>
<p>用平滑关键词语境与最小特征值自适应触发对话，将对话式上下文 Bandit 遗憾降至近极小极大最优 O(√(dT logT))，累积遗憾较基线降低至少 14.6%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (3)</h2>
<dl>
<dt><strong><a href="multimodal_rec/findrec_stein_guided_entropic_flow_for_multi_modal_sequential_recommendatio/">FindRec: Stein-Guided Entropic Flow for Multi-Modal Sequential Recommendation</a></strong></dt>
<dd>
<p>用 Stein 核对齐图文与 ID 流分布、MoE 路由动态筛选跨模态信号，Mamba 建模长序列，三数据集 NDCG@5 相对提升 1.4–3.1%。</p>
</dd>
<dt><strong><a href="multimodal_rec/mdvt_enhancing_multimodal_recommendation_with_model_agnostic_multimodal_dri/">MDVT: Enhancing Multimodal Recommendation with Model-Agnostic Multimodal-Driven Virtual Triplets</a></strong></dt>
<dd>
<p>用多模态融合表征动态构造虚拟三元组作为额外监督信号，配合三种预热阈值策略与增强 pairwise 损失，即插即用缓解数据稀疏，稀疏用户组 NDCG@10 最高提升 12.85%。</p>
</dd>
<dt><strong><a href="multimodal_rec/notellm_2_multimodal_large_representation_models_for_recommendation/">NoteLLM-2: Multimodal Large Representation Models for Recommendation</a></strong></dt>
<dd>
<p>端到端微调 LLM+视觉编码器构建高效 MLRM，用 mICL 与门控后期融合缓解视觉忽视，I2I 全库召回 R@100 显著提升并落地线上 A/B。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/online_item_cold_start_recommendation_with_popularity_aware_meta_learning/">Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning</a></strong></dt>
<dd>
<p>按流行度固定切分元学习任务并配冷启动增强器，流式在线场景下冷启动召回相对 SOTA 提升 11%–74%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (5)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/effective_two_stage_knowledge_transfer_for_multi_entity_cross_domain_recomm/">Effective Two-Stage Knowledge Transfer for Multi-Entity Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>提出 MKT 两阶段多实体知识迁移框架，用异构特征对齐与极化分布损失解决异质特征 schema 下的负迁移，线上 CTR +4.13%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/exploring_preference_guided_diffusion_model_for_cross_domain_recommendation/">Exploring Preference-Guided Diffusion Model for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用源域交互历史构造偏好引导信号，逐步注入扩散逆向过程生成目标域冷启动用户表征，三场景 MAE 最高降 32.60%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/measure_domain_s_gap_a_similar_domain_selection_principle_for_multi_domain_/">Measure Domain&#x27;s Gap: A Similar Domain Selection Principle for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用原型距离+bandit动态为每个域挑选相似域子集，解耦共享专家以缓解多域推荐负迁移，三数据集AUC稳定提升。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/perscen_learning_personalized_interaction_pattern_and_scenario_preference_f/">PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching</a></strong></dt>
<dd>
<p>用用户专属特征图 + VQ 码本蒸馏场景感知偏好 + 渐进式 GLU 融合，在 KuaiRand/Alimama 四场景召回上全面超越 ADIN/SASS/M5。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/revisiting_self_attention_for_cross_domain_sequential_recommendation/">Revisiting Self-attention for Cross-domain Sequential Recommendation</a></strong></dt>
<dd>
<p>将跨域序列推荐重构为推荐损失与跨域注意力抑制的双目标帕累托优化，以即插即用方式改造自注意力，平均提升 Recall@10 达 9.8%/16.0%。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/generative_next_poi_recommendation_with_semantic_id/">Generative Next POI Recommendation with Semantic ID</a></strong></dt>
<dd>
<p>用RQ-VAE将POI多源语义量化为层级语义ID并微调LLaMA3-8B生成下一POI，Acc@1最高提升16%，token开销降55%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (8)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/an_audio_centric_multi_task_learning_framework_for_streaming_ads_targeting_/">An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify</a></strong></dt>
<dd>
<p>面向音频主导多模态广告场景的 CAMoE 框架，以模态感知任务分组、自适应损失掩码与 DCN 专家实现近帕累托最优 CTR 预估，线上音频 CTR +14.5%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/beyond_advertising_mechanism_design_for_platform_wide_marketing_service_qua/">Beyond Advertising: Mechanism Design for Platform-Wide Marketing Service &#x27;QuanZhanTui&#x27;</a></strong></dt>
<dd>
<p>提出全站推场景下的库存约束卖家模型与流式支付拍卖 LPA，在合成与真实数据上同时提升流式福利、用户体验与平台收入。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/bid2x_revealing_dynamics_of_bidding_environment_in_online_advertising_from_/">Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens</a></strong></dt>
<dd>
<p>将自动出价环境建模重构为跨场景基础模型的下一token预测，用变量/时间双注意力+零膨胀投影，8数据集MAE全面领先，线上GMV+4.65%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/contextual_generative_auction_with_permutation_level_externalities_for_onli/">Contextual Generative Auction with Permutation-level Externalities for Online Advertising</a></strong></dt>
<dd>
<p>用自回归生成器直接生成广告位序列并建模排列级外部性，配合可微事后遗憾学习支付规则，淘宝线上 RPM +3.2%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/lightweight_auto_bidding_based_on_traffic_prediction_in_live_advertising/">Lightweight Auto-bidding based on Traffic Prediction in Live Advertising</a></strong></dt>
<dd>
<p>直播广告场景下，用LightGBM预测全时段累计成本/点击，结合LP对偶最优出价公式与投影梯度下降求解对偶变量，实现秒级轻量自动出价，线上GMV+4.28%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/scaling_transformers_for_discriminative_recommendation_via_generative_pretr/">Scaling Transformers for Discriminative Recommendation via Generative Pretraining</a></strong></dt>
<dd>
<p>用生成式预训练初始化判别式排序模型并冻结稀疏Embedding，破解CTR/CVR过拟合，使稠密参数从13K扩到0.3B仍遵循幂律提升。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/swat_statistical_modeling_of_video_watch_time_through_user_behavior_analysi/">SWaT: Statistical Modeling of Video Watch Time through User Behavior Analysis</a></strong></dt>
<dd>
<p>将用户观看行为假设翻译为分桶二项/几何分布白盒统计模型，以软标签 BCE 训练，离线与线上 A/B 均稳定超越 WLR/TPM/CREAD。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/tayfcs_towards_light_feature_combination_selection_for_deep_recommender_sys/">TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems</a></strong></dt>
<dd>
<p>用泰勒展开+IME近似一次性梯度回传估计高阶组合重要性，再用LR滑窗剔除冗余，三数据集AUC显著提升且线上CVR+13.9%</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (4)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/beyond_item_dissimilarities_diversifying_by_intent_in_recommender_systems/">Beyond Item Dissimilarities: Diversifying by Intent in Recommender Systems</a></strong></dt>
<dd>
<p>用实时意图概率驱动整页贪心重排，以贝叶斯后验更新替代物品相似度，YouTube 线上 DAU +0.05%、用户享受度 +0.09%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/breaking_the_top_k_barrier_advancing_top_k_ranking_metrics_optimization_in_/">Breaking the Top-$K$Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems</a></strong></dt>
<dd>
<p>用量化分位数把 Top-K 截断转成阈值比较，推导 NDCG@K 的紧致平滑上界 SL@K，四数据集三骨干平均提升 6.03%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/combinatorial_optimization_perspective_based_framework_for_multi_behavior_r/">Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation</a></strong></dt>
<dd>
<p>将多行为融合视为组合优化问题施加三阶段约束，并用分布拟合多专家网络抑制负迁移，三数据集 HR@10 最高提升 49.91%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/mopi_hfrs_a_multi_objective_personalized_health_aware_food_recommendation_s/">MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation</a></strong></dt>
<dd>
<p>构建首个融合 NHANES 医疗数据的健康感知食物推荐基准，用符号图结构学习 + Pareto 多目标优化平衡偏好/个性化健康/营养多样性，并以知识注入 LLM 生成可解释理由。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (5)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/a_unified_online_offline_framework_for_co_branding_campaign_recommendations/">A Unified Online-Offline Framework for Co-Branding Campaign Recommendations</a></strong></dt>
<dd>
<p>首个联名营销推荐系统性研究：二分图建模+方差感知CUCB在线学习+部分枚举离线预算分配，理论保证近优regret与1-1/e近似，真实数据收益提升12%-73%。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/an_adaptable_budget_planner_for_enhancing_budget_constrained_auto_bidding_i/">An Adaptable Budget Planner for Enhancing Budget-Constrained Auto-Bidding in Online Advertising</a></strong></dt>
<dd>
<p>将竞价周期切分为多阶段，用 in-context RL 逐 episode 调整预算分配，few-shot 适配新广告主，仿真与线上 A/B 均提升累计价值。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/multi_task_offline_reinforcement_learning_for_online_advertising_in_recomme/">Multi-task Offline Reinforcement Learning for Online Advertising in Recommender Systems</a></strong></dt>
<dd>
<p>用因果状态编码器+因果注意力+多任务解码器把广告渠道推荐与预算分配统一进离线RL，KuaiRand/Criteo 上 F1 提升约 7.5%/16.2%，线上 CTR +0.08。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/revisiting_clustering_of_neural_bandits_selective_reinitialization_for_miti/">Revisiting Clustering of Neural Bandits: Selective Reinitialization for Mitigating Loss of Plasticity</a></strong></dt>
<dd>
<p>针对神经聚类 Bandit 的可塑性丧失问题，提出选择性重初始化框架 SeRe，配合自适应变点检测，在六个推荐数据集上最多降低 12.82% 累积遗憾。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/session_level_dynamic_ad_load_optimization_using_offline_robust_reinforceme/">Session-Level Dynamic Ad Load Optimization using Offline Robust Reinforcement Learning</a></strong></dt>
<dd>
<p>将会话级广告加载建模为鲁棒MDP，用IPM不确定集导出免额外开销的鲁棒对决DQN，离线AUCC提升80%+，线上A/B双位数增益。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (10)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/addressing_correlated_latent_exogenous_variables_in_debiased_recommender_sy/">Addressing Correlated Latent Exogenous Variables in Debiased Recommender Systems</a></strong></dt>
<dd>
<p>放松外生变量独立假设，用双变量正态潜变量建模选择与偏好相关性，蒙特卡洛似然估计实现去偏，Coat/Yahoo!R3/KuaiRec 全指标领先。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/breaker_removing_shortcut_cues_with_user_clustering_for_single_slot_recomme/">Breaker: Removing Shortcut Cues with User Clustering for Single-slot Recommendation System</a></strong></dt>
<dd>
<p>用用户表征聚类辅助任务+多塔簇专属偏好建模，破除单槽推荐中用户固有倾向的捷径偏置，离线Recall@1提升7%并上线美团支付。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/data_watermarking_for_sequential_recommender_systems/">Data Watermarking for Sequential Recommender Systems</a></strong></dt>
<dd>
<p>首次定义序列推荐数据集/单用户双场景数据水印，用感受野约束插入连续物品序列，5 模型 3 数据集上水印 NDCG 近 1.0 且效用无损。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/flexible_generation_of_preference_data_for_recommendation_analysis/">Flexible Generation of Preference Data for Recommendation Analysis</a></strong></dt>
<dd>
<p>HYDRA 以用户-物品匹配、物品流行度、用户活跃度三因子混合分布生成可控合成偏好数据，复现真实基准的度分布与算法排序。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/forward_once_for_all_structural_parameterized_adaptation_for_efficient_clou/">Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation</a></strong></dt>
<dd>
<p>用结构控制器+结构引导映射器为每台设备动态组装子网络，一次前向即完成端侧适配，兼顾隐私、带宽与算力。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/gigl_large_scale_graph_neural_networks_at_snapchat/">GiGL: Large-Scale Graph Neural Networks at Snapchat</a></strong></dt>
<dd>
<p>Snapchat 开源十亿级图学习库 GiGL，以表格化子图采样+PyG 解耦实现 900M 节点/16.8B 边图 12 小时内端到端训练推理，支撑 35+ 线上发布。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/gradients_as_an_action_towards_communication_efficient_federated_recommende/">Gradients as an Action: Towards Communication-Efficient Federated Recommender Systems via Adaptive Action Sharing</a></strong></dt>
<dd>
<p>将物品梯度聚类为少量“动作”共享传输，配合自适应聚类与带宽感知压缩，在 ML-100K/1M/Lastfm-2K 上以最高 96.88% 压缩率不损性能。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/harnessing_the_power_of_interleaving_and_counterfactual_evaluation_for_airb/">Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking</a></strong></dt>
<dd>
<p>提出竞争对式 Team Draft 交错实验与在线反事实评估，用双层随机化与位置差增益估计器将 Airbnb 搜索排序实验灵敏度提升 50–100 倍。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/merrec_a_large_scale_multipurpose_mercari_dataset_for_consumer_to_consumer_/">MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems</a></strong></dt>
<dd>
<p>首个大规模 C2C 推荐数据集 MerRec（5.56M 用户/83M 商品/1.27B 交互），并配套三塔 Mercatran 模型与四任务基准。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/taming_recommendation_bias_with_causal_intervention_on_evolving_personal_po/">Taming Recommendation Bias with Causal Intervention on Evolving Personal Popularity</a></strong></dt>
<dd>
<p>提出演化个人流行度度量与因果去混杂训练，并在推理期用移动平均趋势外推做干预，兼顾用户-物品演化一致性，Full-Ranking 下 Recall@20 最高提升 20.4%。</p>
</dd>
</dl>
</div>

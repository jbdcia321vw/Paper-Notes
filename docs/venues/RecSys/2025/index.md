---
hide:
  - toc
---

# 🏛️ RecSys 2025 推荐系统论文解读

86 篇 RecSys 2025 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (17)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/a_non_parametric_choice_model_that_learns_how_users_choose_between_recommen/">A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options</a></strong></dt>
<dd>
<p>用 sigmoid 核密度估计学习误差分布，无需假设 MNL/ENL 即可同时推断用户偏好与选择机制，抗竞争型曝光偏差。</p>
</dd>
<dt><strong><a href="collaborative_filtering/agentic_personalisation_of_cross_channel_marketing_experiences/">Agentic Personalisation of Cross-Channel Marketing Experiences</a></strong></dt>
<dd>
<p>用 DiD 因果估计 + Thompson 采样构建模块化营销智能体，在 640 万用户线上 RCT 中显著提升漏斗事件与 GMV。</p>
</dd>
<dt><strong><a href="collaborative_filtering/balancing_accuracy_and_novelty_with_sub_item_popularity/">Balancing Accuracy and Novelty with Sub-Item Popularity</a></strong></dt>
<dd>
<p>复用 RecJPQ 子 ID 分解在子码本粒度计算个性化流行度 sPPS，与 item 级 PPS 加权融合，在 NDCG 不降前提下显著提升个性化新颖度。</p>
</dd>
<dt><strong><a href="collaborative_filtering/correcting_the_logq_correction_revisiting_sampled_softmax_for_large_scale_r/">Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval</a></strong></dt>
<dd>
<p>指出标准 logQ 修正误把正样本当作采样项，提出带置信度权重的改进修正，在公开与工业数据上稳定超越标准 logQ。</p>
</dd>
<dt><strong><a href="collaborative_filtering/estimating_quantum_execution_requirements_for_feature_selection_in_recommen/">Estimating Quantum Execution Requirements for Feature Selection in Recommender Systems Using Extreme Value Theory</a></strong></dt>
<dd>
<p>用极值理论对 QAOA 求解 QUBO 特征选择的输出建模，估计达到经典 SA 水平所需的量子执行次数，并在模拟器与真实 QPU 上验证。</p>
</dd>
<dt><strong><a href="collaborative_filtering/hierarchical_graph_information_bottleneck_for_multi_behavior_recommendation/">Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用信息瓶颈原理重构多行为层次推荐，以 InfoNCE 保真、HSIC 压缩、可学习边丢弃去噪，全库排序下 HR@10 最高提升 15.08%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/industry_insights_from_comparing_deep_learning_and_gbdt_models_for_e_commer/">Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank</a></strong></dt>
<dd>
<p>在 OTTO 电商 LTR 场景系统对比 DNN 与生产级 LambdaMART，Two-Tower+Softmax CE 离线 NDCG_c +4.32%，8 周线上点击 +1.86%、营收 +0.56%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/ip2_entity_guided_interest_probing_for_personalized_news_recommendation/">IP2: Entity-Guided Interest Probing for Personalized News Recommendation</a></strong></dt>
<dd>
<p>以实体-标题对比预训练探测新闻内实体焦点，并用跨注意力双塔用户编码器建模新闻间实体引导兴趣，MIND-large AUC 达 72.06%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/kamae_bridging_spark_and_keras_for_seamless_ml_preprocessing/">Kamae: Bridging Spark and Keras for Seamless ML Preprocessing</a></strong></dt>
<dd>
<p>将 PySpark 预处理管线一对一翻译为等价 Keras 层，消除离线/在线特征工程重复实现，Expedia 线上延迟降 61%、成本降 58%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/location_matters_leveraging_multi_resolution_geo_embeddings_for_housing_sea/">Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search</a></strong></dt>
<dd>
<p>将 H3 多分辨率六边形网格与城市嵌入注入双塔对比学习，缓解房源地理稀疏，离线重排模拟租金转化提升 85%。</p>
</dd>
</dl>
<p><a href="collaborative_filtering/">查看全部17篇「协同过滤与双塔召回」论文 →</a></p>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/nlgcl_naturally_existing_neighbor_layers_graph_contrastive_learning_for_rec/">NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用 GNN 相邻层天然异构节点作正样本对替代图增广，免视图构建，全库排序下 NDCG@10 最高提升 12.41% 且训练最快。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/sgcl_unifying_self_supervised_and_supervised_learning_for_graph_recommendat/">SGCL: Unifying Self-Supervised and Supervised Learning for Graph Recommendation</a></strong></dt>
<dd>
<p>将 BPR 与 InfoNCE 合并为单一监督图对比损失，免增广免负采样，精度与训练效率双超 SGL/SimGCL。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl>
<dt><strong><a href="sequential_rec/esasrec_enhancing_transformer_based_recommendations_in_a_modular_fashion/">eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion</a></strong></dt>
<dd>
<p>系统消融 Transformer 序列推荐各模块增强，提出 SASRec+LiGR+SS 组合 eSASRec，在时间切分与覆盖率指标下与 HSTU 同处帕累托前沿。</p>
</dd>
<dt><strong><a href="sequential_rec/more_a_mixture_of_reflectors_framework_for_large_language_model_based_seque/">MoRE: A Mixture of Reflectors Framework for Large Language Model-Based Sequential Recommendation</a></strong></dt>
<dd>
<p>用显式/隐式/协同三视角离线反思 + 元反思器自改进与 Contextual Bandit 在线选视角，LLM 序列推荐全库候选下 NDCG@10 最高提升 10.1%。</p>
</dd>
<dt><strong><a href="sequential_rec/not_just_what_but_when_integrating_irregular_intervals_to_llm_for_sequentia/">Not Just What, But When: Integrating Irregular Intervals to LLM for Sequential Recommendation</a></strong></dt>
<dd>
<p>将不规则时间间隔经 MLP 编码与 Interval-Infused Attention 注入 LLaMA-2，配合选项化提示，在 3 个 Amazon 数据集上 HR@1 平均提升 4.4%，并提出“间隔冷启动”新评测视角。</p>
</dd>
<dt><strong><a href="sequential_rec/time_to_split_exploring_data_splitting_strategies_for_offline_evaluation_of/">Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders</a></strong></dt>
<dd>
<p>系统对比 LOO 与全局时间切分及目标/验证变体，揭示 LOO 时序泄漏与排序失真，推荐 GTS+Last/Random/Successive 协议。</p>
</dd>
<dt><strong><a href="sequential_rec/user_long_term_multi_interest_retrieval_model_for_recommendation/">User Long-Term Multi-Interest Retrieval Model for Recommendation</a></strong></dt>
<dd>
<p>以类别感知分层双兴趣学习与指针生成式类别→物品级联检索，将召回阶段行为序列扩展至千级，淘宝离线HR@2000达16.55%，线上点击+5.54%、订单+11.01%。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (6)</h2>
<dl>
<dt><strong><a href="generative_rec/enhancing_embedding_representation_stability_in_recommendation_systems_with/">Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID</a></strong></dt>
<dd>
<p>提出 Semantic ID prefix-ngram 分层前缀参数化，用语义聚类碰撞替代随机哈希，缓解 ID 漂移与长尾稀疏，Meta 线上 NE 提升 0.15% 且 A/A 方差降 43%</p>
</dd>
<dt><strong><a href="generative_rec/grace_generative_recommendation_via_journey_aware_sparse_attention_on_chain/">GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization</a></strong></dt>
<dd>
<p>用 PKG 属性链式 CoT token 增强语义 ID，并以旅程感知四路稀疏注意力替代全注意力，Home 域 HR@10 相对提升 106.9%，长序列注意力计算降 48%。</p>
</dd>
<dt><strong><a href="generative_rec/how_fair_is_your_diffusion_recommender_model/">How Fair is Your Diffusion Recommender Model?</a></strong></dt>
<dd>
<p>首个针对 DiffRec/L-DiffRec 的公平性实证研究，在 ML1M/FTKY 上以 Full-Ranking 评估效用与消费者/提供者公平，揭示扩散模型放大偏见而 L-DiffRec 可缓解。</p>
</dd>
<dt><strong><a href="generative_rec/prompt_to_slate_diffusion_models_for_prompt_conditioned_slate_generation/">Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation</a></strong></dt>
<dd>
<p>用潜空间扩散 Transformer 从纯文本 Prompt 直接生成整组 Slate，离线 NDCG 提升 17%、线上重复曝光降 13.4%。</p>
</dd>
<dt><strong><a href="generative_rec/semantic_ids_for_joint_generative_search_and_recommendation/">Semantic IDs for Joint Generative Search and Recommendation</a></strong></dt>
<dd>
<p>系统对比任务专属与跨任务语义ID构建策略，发现多任务双塔嵌入+统一RQ-KMeans语义ID在联合搜索推荐中取得最佳权衡。</p>
</dd>
<dt><strong><a href="generative_rec/semantic_ids_for_music_recommendation/">Semantic IDs for Music Recommendation</a></strong></dt>
<dd>
<p>用 RQ-VAE 残差量化生成音乐语义 ID 替代逐曲嵌入，参数量降 75%–99% 而精度不降，线上多样性显著提升。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl>
<dt><strong><a href="llm_rec/balancing_fine_tuning_and_rag_a_hybrid_strategy_for_dynamic_llm_recommendat/">Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates</a></strong></dt>
<dd>
<p>以十亿用户短视频兴趣探索系统为案例，量化对比微调与RAG的更新代价/敏捷性，提出月度微调+周内RAG刷新的混合更新策略，线上A/B显著提升满意度。</p>
</dd>
<dt><strong><a href="llm_rec/biases_in_llm_generated_musical_taste_profiles_for_recommendation/">Biases in LLM-Generated Musical Taste Profiles for Recommendation</a></strong></dt>
<dd>
<p>通过 64 人用户研究揭示 LLM 生成音乐品味画像存在流派/国别系统性偏差，且用户认同度与下游推荐指标仅弱相关。</p>
</dd>
<dt><strong><a href="llm_rec/evaluating_podcast_recommendations_with_profile_aware_llm_as_a_judge/">Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge</a></strong></dt>
<dd>
<p>用90天收听历史蒸馏自然语言用户画像，驱动零样本LLM做剧集级与模型级离线评判，47人用户研究中与人工判断高度一致。</p>
</dd>
<dt><strong><a href="llm_rec/heterogeneous_user_modeling_for_llm_based_recommendation/">Heterogeneous User Modeling for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用压缩提示+[USER]锚点token将跨域异构行为压成单一用户表征，并以域重要性平滑加权缓解域跷跷板，六域全库排序R@10最高提升约30%。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/failure_prediction_in_conversational_recommendation_systems/">Failure Prediction in Conversational Recommendation Systems</a></strong></dt>
<dd>
<p>提出监督式对话性能预测任务，用AE与L1收缩多轮预测器区分系统失败与目录失败，揭示目录缺失场景下预测精度显著下降。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/usb_rec_an_effective_framework_for_improving_conversational_recommendation_/">USB-Rec: An Effective Framework for Improving Conversational Recommendation Capability of Large Language Model</a></strong></dt>
<dd>
<p>用带标签用户模拟器自动构造偏好对做 RL 训练，再在推理期用内部用户模拟器+树搜索自增强，iEval 全面超越 SOTA。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (9)</h2>
<dl>
<dt><strong><a href="multimodal_rec/aps_explorer_navigating_algorithm_performance_spaces_for_informed_dataset_s/">APS Explorer: Navigating Algorithm Performance Spaces for Informed Dataset Selection</a></strong></dt>
<dd>
<p>将 29 算法 × 96 数据集的性能向量投影为可交互 PCA 性能空间，用可视化工具支撑可辩护的数据集选择。</p>
</dd>
<dt><strong><a href="multimodal_rec/beyond_the_past_leveraging_audio_and_human_memory_for_sequential_music_reco/">Beyond the past: Leveraging Audio and Human Memory for Sequential Music Recommendation</a></strong></dt>
<dd>
<p>用音频编码器预测新歌的 ACT-R 类激活值，将记忆式重复建模扩展到未听歌曲，NDCG_Exp 相对 PISA 提升约 40%，RepBias 降至 0.09%。</p>
</dd>
<dt><strong><a href="multimodal_rec/image_seeking_intent_prediction_for_cross_device_product_search/">Image-Seeking Intent Prediction for Cross-Device Product Search</a></strong></dt>
<dd>
<p>用语音查询+商品摘要预测图像意图，以可微精度损失抑制误触发，离线 Precision 达 78.5。</p>
</dd>
<dt><strong><a href="multimodal_rec/improving_visual_recommendation_on_e_commerce_platforms_using_vision_langua/">Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models</a></strong></dt>
<dd>
<p>在 Mercari C2C 平台微调 SigLIP 图像编码器做视觉相似召回，离线 nDCG@5 +9.1%，线上 CTR +50%、CVR +14%。</p>
</dd>
<dt><strong><a href="multimodal_rec/just_ask_for_music_jam_multimodal_and_personalized_natural_language_music_r/">Just Ask for Music (JAM): Multimodal and Personalized Natural Language Music Recommendation</a></strong></dt>
<dd>
<p>将查询建模为用户到物品的向量平移，用交叉注意力聚合音频/歌词/CF 多模态，在 JAMSessions 全库排序上 NDCG@10 相对 TalkRec 提升约 105%</p>
</dd>
<dt><strong><a href="multimodal_rec/llm_powered_nuanced_video_attribute_annotation_for_enhanced_recommendations/">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></strong></dt>
<dd>
<p>用 Gemini 多模态 LLM 做短视频“氛围”属性标注，经金/银集评估与蒸馏扩量，接入个性化受限召回，线上参与度 +0.49%。</p>
</dd>
<dt><strong><a href="multimodal_rec/personalized_image_generation_for_recommendations_beyond_catalogs/">Personalized Image Generation for Recommendations Beyond Catalogs</a></strong></dt>
<dd>
<p>用轻量条件扩散先验从隐式反馈采样用户个性化 CLIP 嵌入，冻结 SD+IP-Adapter 解码，免微调实现可扩展个性化图像生成。</p>
</dd>
<dt><strong><a href="multimodal_rec/vl_clip_enhancing_multimodal_recommendations_via_visual_grounding_and_llm_a/">VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings</a></strong></dt>
<dd>
<p>用 Grounding DINO 裁剪商品主体 + LLM 自反思改写商品描述，再微调 CLIP 双塔，Walmart 线上 CTR +18.6%、GMV +4.0%。</p>
</dd>
<dt><strong><a href="multimodal_rec/yambda_5b_a_large_scale_multi_modal_dataset_for_ranking_and_retrieval/">Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval</a></strong></dt>
<dd>
<p>发布 Yandex Music 4.79B 交互工业级数据集，含音频嵌入与 is_organic 标志，提出 Global Temporal Split 评测协议并给出 6 类基线基准。</p>
</dd>
</dl>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (3)</h2>
<dl>
<dt><strong><a href="social_group_rec/a_multistakeholder_approach_to_value_driven_co_design_of_recommender_system/">A Multistakeholder Approach to Value-Driven Co-Design of Recommender System Evaluation Metrics in Digital Archives</a></strong></dt>
<dd>
<p>通过 5×5 焦点小组将档案领域五类利益相关者价值映射为四阶段研究漏斗评估指标框架，提出研究路径质量、元数据加权相关性等指标方向。</p>
</dd>
<dt><strong><a href="social_group_rec/consistent_explainers_or_unreliable_narrators_understanding_llm_generated_g/">Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations</a></strong></dt>
<dd>
<p>用社会选择聚合策略作多基线对4个LLM的群组推荐与解释做一致性审计，发现推荐近似ADD、解释常称平均但夹带未定义流行度/相似度/多样性。</p>
</dd>
<dt><strong><a href="social_group_rec/you_don_t_bring_me_flowers_mitigating_unwanted_recommendations_through_conf/">You Don&#x27;t Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control</a></strong></dt>
<dd>
<p>用保形风险控制对任意排序器做后处理，以“重复安全内容替换”替代纯删除，在 KuaiRand 上可证明地把不想要内容比例压到用户指定 α 以下且保住 nDCG。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (9)</h2>
<dl>
<dt><strong><a href="cold_start_rec/do_we_really_need_specialization_evaluating_generalist_text_embeddings_for_/">Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search</a></strong></dt>
<dd>
<p>实证通用文本嵌入(GTE)零样本即超越领域微调模型，并以PCA有效维度揭示空间利用率与压缩收益。</p>
</dd>
<dt><strong><a href="cold_start_rec/exploring_the_potential_of_llms_for_serendipity_evaluation_in_recommender_s/">Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 模拟用户对推荐结果做惊喜度（serendipity）打分，在淘宝与 MovieLens 两个真实用户调研数据集上做元评估，最优配置 Pearson 相关达 21.5%，显著超越传统代理指标。</p>
</dd>
<dt><strong><a href="cold_start_rec/item_centric_exploration_for_cold_start_problem/">Item-centric Exploration for Cold Start Problem</a></strong></dt>
<dd>
<p>提出以物品为中心的 Beta 后验过滤控制，在排序后剔除“错配受众”曝光，线上探索曝光降 20% 而满意度升 50%、可推荐语料扩 10%。</p>
</dd>
<dt><strong><a href="cold_start_rec/let_it_go_not_quite_addressing_item_cold_start_in_sequential_recommendation/">Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization</a></strong></dt>
<dd>
<p>冻结内容嵌入+有界可训练 delta 校正，兼顾冷启动物品语义结构与热物品任务适配，三数据集冷启动 NDCG@10 最高提升 20.9%。</p>
</dd>
<dt><strong><a href="cold_start_rec/llm_recg_a_semantic_bias_aware_framework_for_zero_shot_sequential_recommend/">LLM-RecG: A Semantic Bias-Aware Framework for Zero-Shot Sequential Recommendation</a></strong></dt>
<dd>
<p>揭示 LLM 语义嵌入的域语义偏置，用域间紧致+域内多样双熵损失与源域序列模式注意力实现零样本跨域序列推荐，R@10 平均提升超 10%。</p>
</dd>
<dt><strong><a href="cold_start_rec/on_inherited_popularity_bias_in_cold_start_item_recommendation/">On Inherited Popularity Bias in Cold-Start Item Recommendation</a></strong></dt>
<dd>
<p>揭示生成式冷启动模型继承热模型流行度偏置，提出基于向量模长重标定的免训练后处理去偏方法，在三个多媒体数据集上显著提升物品公平性与曝光多样性。</p>
</dd>
<dt><strong><a href="cold_start_rec/reading_between_the_lines_a_study_of_thematic_bias_in_book_recommender_syst/">Reading Between the Lines: A Study of Thematic Bias in Book Recommender Systems</a></strong></dt>
<dd>
<p>用 BERTopic 为 Book-Crossing 打主题标签，构建数据-推荐-用户三阶段主题偏置评估框架，揭示长尾小众读者被主流主题同质化的现象。</p>
</dd>
<dt><strong><a href="cold_start_rec/recommendation_is_a_dish_better_served_warm/">Recommendation Is a Dish Better Served Warm</a></strong></dt>
<dd>
<p>系统量化冷启动阈值：物品侧训练交互扫描+用户侧推理历史截断，揭示 6–15 条交互的冷热分界并批判 p-core 过滤的任意性。</p>
</dd>
<dt><strong><a href="cold_start_rec/socripple_a_two_stage_framework_for_cold_start_video_recommendations/">SocRipple: A Two-Stage Framework for Cold-Start Video Recommendations</a></strong></dt>
<dd>
<p>社交种子+嵌入KNN涟漪扩散的两阶段冷启视频召回框架，Recall@200 提升至 12.8–13.2%，线上冷启曝光 +36% 且互动率持平。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (5)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/affect_aware_cross_domain_recommendation_for_art_therapy_via_music_preferen/">Affect-aware Cross-Domain Recommendation for Art Therapy via Music Preference Elicitation</a></strong></dt>
<dd>
<p>以音乐偏好诱导替代纯视觉诱导，提出 Mozart/Haydn/Salieri 三种情感感知跨域艺术治疗推荐，200 人用户研究验证情绪改善。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/impacts_of_mainstream_driven_algorithms_on_recommendations_for_children_acr/">Impacts of Mainstream-Driven Algorithms on Recommendations for Children Across Domains: A Reproducibility Study</a></strong></dt>
<dd>
<p>跨电影/音乐/图书三域复现儿童与主流用户偏好偏差，并引入流行度偏差指标，揭示推荐算法对儿童群体的域特异性失效。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/leave_no_one_behind_fairness_aware_cross_domain_recommender_systems_for_non/">Leave No One Behind: Fairness-Aware Cross-Domain Recommender Systems for Non-Overlapping Users</a></strong></dt>
<dd>
<p>为无源域数据的非重叠用户生成虚拟源域画像，双注意力+限制器实现模型无关的跨域公平性纠偏，UGF最高降94.13%且整体精度不降。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/rankgraph_unified_heterogeneous_graph_learning_for_cross_domain_recommendat/">RankGraph: Unified Heterogeneous Graph Learning for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>构建跨产品异构异质图，用 GPU 加速 RGCN + 对比学习产出图 token 注入推荐基础模型，线上 CTR +0.92%、CVR +2.82%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/zero_shot_cross_domain_knowledge_distillation_a_case_study_on_youtube_music/">Zero-shot Cross-domain Knowledge Distillation: A Case study on YouTube Music</a></strong></dt>
<dd>
<p>用 YouTube 视频大教师零样本蒸馏到低流量 YouTube Music 双排序模型，离线 AUC 微增但线上新发内容互动 +11.39%。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/exploring_the_effect_of_context_awareness_and_popularity_calibration_on_pop/">Exploring the Effect of Context-Awareness and Popularity Calibration on Popularity Bias in POI Recommendations</a></strong></dt>
<dd>
<p>系统评测上下文感知模型与校准流行度(CP)重排在POI推荐中的去偏效果，发现二者组合可在保精度同时最贴近用户流行度画像。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (7)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/blending_sequential_embeddings_graphs_and_engineered_features_4th_place_sol/">Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025</a></strong></dt>
<dd>
<p>异构因果Transformer序列编码+TwHIN图+300维手工特征经DCN-v2融合，构建跨6任务通用用户画像，获挑战赛第4名。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/counterfactual_inference_under_thompson_sampling/">Counterfactual Inference under Thompson Sampling</a></strong></dt>
<dd>
<p>推导 Thompson Sampling 下动作倾向性的精确可计算闭式解，使 IPS 类离线评估在 TS 策略下首次可行，OBP 仿真验证无偏收敛。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/decoupled_entity_representation_learning_for_pinterest_ads_ranking/">Decoupled Entity Representation Learning for Pinterest Ads Ranking</a></strong></dt>
<dd>
<p>提出解耦式上游-下游实体表征框架 DERM，三塔+对比自监督+滑动平均聚合，离线 AUC 提升 0.19%、线上 CTR +1.38%、CPA -1.61%</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/exploring_scaling_laws_of_ctr_model_for_online_performance_improvement/">Exploring Scaling Laws of CTR Model for Online Performance Improvement</a></strong></dt>
<dd>
<p>提出 SUAN 统一注意力块实现 CTR 模型三级量级缩放律，再经稀疏注意力+并行推理+在线蒸馏落地线上，CTR +2.81%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/identifying_offline_metrics_that_predict_online_impact_a_pragmatic_strategy/">Identifying Offline Metrics that Predict Online Impact: A Pragmatic Strategy for Real-World Recommender Systems</a></strong></dt>
<dd>
<p>用帕累托前沿近似单模型多偏好分流，在 OTTO 线上实验验证 Recall@20→CTR、OD@20→CVR、乘积→销量 的离线-在线对齐关系。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/modeling_long_term_user_behaviors_with_diffusion_driven_multi_interest_netw/">Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction</a></strong></dt>
<dd>
<p>用目标正交分解构造兴趣通道做多兴趣抽取，再用条件扩散生成增强兴趣，工业 A/B 上 CTR +1.52%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/r4ec_a_reasoning_reflection_and_refinement_framework_for_recommendation_sys/">R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems</a></strong></dt>
<dd>
<p>用 actor/reflection 双小模型迭代反思精炼 LLM 知识，注入 CTR 主干，AUC 相对提升 2.5%+，线上收入 +2.2%。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (5)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/d_rdw_diversity_driven_random_walks_for_news_recommender_systems/">D-RDW: Diversity-Driven Random Walks for News Recommender Systems</a></strong></dt>
<dd>
<p>将规范目标分布(NTD)作为线性约束嵌入随机游走采样，实现可解释、低能耗的新闻多样化重排，能耗降低一个数量级。</p>
</dd>
<dt><strong><a href="multi_task_reranking/non_parametric_graph_convolution_for_re_ranking_in_recommendation_systems/">Non-parametric Graph Convolution for Re-ranking in Recommendation Systems</a></strong></dt>
<dd>
<p>将图卷积从训练期搬到测试期，用相似用户/物品构造增强对并加权聚合打分，重排平均提升 8.1% 而额外开销仅 0.5%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/paragon_parameter_generation_for_controllable_multi_task_recommendation/">Paragon: Parameter Generation for Controllable Multi-Task Recommendation</a></strong></dt>
<dd>
<p>用扩散模型按偏好权重条件生成适配器参数，免重训实现测试时可控多任务推荐，响应时间降低 94.6%+。</p>
</dd>
<dt><strong><a href="multi_task_reranking/practical_multi_task_learning_for_rare_conversions_in_ad_tech/">Practical Multi-Task Learning for Rare Conversions in Ad Tech</a></strong></dt>
<dd>
<p>以历史CVR统计自动划分软/硬转化任务，共享底层+双塔MTL提升稀有转化预估，离线AUC+0.69%、线上CPA降2%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/radar_recall_augmentation_through_deferred_asynchronous_retrieval/">RADAR: Recall Augmentation through Deferred Asynchronous Retrieval</a></strong></dt>
<dd>
<p>用离线异步全复杂度精排模型对 50 倍候选池预打分并缓存 Top-200，线上绕过召回与粗排直接注入精排，Recall@200 翻倍、线上 +0.8%。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/exploitation_over_exploration_unmasking_the_bias_in_linear_bandit_recommend/">Exploitation Over Exploration: Unmasking the Bias in Linear Bandit Recommender Offline Evaluation</a></strong></dt>
<dd>
<p>大规模实证揭示线性 Bandit 离线回放评估系统性偏袒纯利用策略，贪心模型在 89% 数据集上不劣于探索变体。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/off_policy_evaluation_and_learning_for_matching_markets/">Off-Policy Evaluation and Learning for Matching Markets</a></strong></dt>
<dd>
<p>面向双向匹配市场提出 DiPS/DPR 两阶段奖励 OPE 估计器，利用一阶段 scout 信号降方差，合成+Wantedly 真实 A/B 日志验证 MSE 与策略选择显著优于 DM/IPS/DR。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/a_reproducibility_study_of_product_side_fairness_in_bundle_recommendation/">A Reproducibility Study of Product-side Fairness in Bundle Recommendation</a></strong></dt>
<dd>
<p>首个捆绑推荐商品侧公平性复现研究，揭示捆绑级与物品级曝光分布不一致、指标间结论分歧，并发现用户捆绑偏好越强曝光越公平。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/leveraging_geometric_insights_in_hyperbolic_triplet_loss_for_improved_recom/">Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations</a></strong></dt>
<dd>
<p>洛伦兹双曲三元组损失 TriplH，以物品-物品项自适应间隔替代固定 margin，兼顾精度、去流行度偏差与 3 倍推理加速。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/mitigating_popularity_bias_in_counterfactual_explanations_using_large_langu/">Mitigating Popularity Bias in Counterfactual Explanations using Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 生成用户画像并贪心剔除“出格”历史项，作为 ACCENT 反事实解释的即插即用预处理，使解释流行度分布更贴合用户偏好。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/opening_the_black_box_interpretable_remedies_for_popularity_bias_in_recomme/">Opening the Black Box: Interpretable Remedies for Popularity Bias in Recommender Systems</a></strong></dt>
<dd>
<p>用稀疏自编码器定位流行度神经元并做激活转向，在 ML-1M/Last.fm 上以极小精度损失换取显著曝光公平提升。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/privacy_risks_of_llm_empowered_recommender_systems_an_inversion_attack_pers/">Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective</a></strong></dt>
<dd>
<p>首次系统研究 LLM 推荐系统的 logits 反演攻击，提出相似度引导精炼，可恢复 65% 交互物品与 87% 年龄性别。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/recps_privacy_risk_scoring_for_recommender_systems/">RecPS: Privacy Risk Scoring for Recommender Systems</a></strong></dt>
<dd>
<p>基于差分隐私推导交互级隐私分数，用 RecLiRA 成员推断攻击估计 TPR/FPR，指导细粒度交互移除以兼顾隐私与效用。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/rethinking_the_privacy_of_text_embeddings_a_reproducibility_study_of_text_e/">Rethinking the Privacy of Text Embeddings: A Reproducibility Study of \&quot;Text Embeddings Reveal (Almost) As Much As Text\&quot;</a></strong></dt>
<dd>
<p>复现 Vec2Text 嵌入反演攻击，揭示其长度敏感性，并提出免调参 8-bit 量化作为轻量隐私防御。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/ricciflowrec_a_geometric_root_cause_recommender_using_ricci_curvature_on_fi/">RicciFlowRec: A Geometric Root Cause Recommender Using Ricci Curvature on Financial Graphs</a></strong></dt>
<dd>
<p>用 Ollivier/Forman-Ricci 曲率与 Ricci 流在动态金融图上做根因归因，以曲率风险惩罚重排资产，NDCG@10 提升 4.9%、Top-10 波动降 28%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/stairway_to_fairness_connecting_group_and_individual_fairness/">Stairway to Fairness: Connecting Group and Individual Fairness</a></strong></dt>
<dd>
<p>首次用同一族度量（SD/Gini/Atk）在 3 数据集 8 组 LLMRec 运行上实证组公平与个体公平的错位，揭示组间公平掩盖组内与个体不公。</p>
</dd>
</dl>
</div>

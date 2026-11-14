# 📍 时空轨迹与 POI 本地生活推荐 (POI & Spatio-Temporal Recommendation)

> **研究范畴**：Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐

当前方向共收录 **19** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/poi_spatiotemporal_rec/spatial_temporal_knowledge_distillation_for_takeaway_recommendation/">Spatial-Temporal Knowledge Distillation for Takeaway Recommendation</a></strong></dt>
<dd>
<p>用 STKG 教师模型蒸馏时空知识给 ST-Transformer 学生模型，融合地理距离与功能区序列建模外卖动态偏好，三城 HR@10 提升约 1.5–2.8%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/poi_spatiotemporal_rec/triptailor_a_real_world_benchmark_for_personalized_travel_planning/">TripTailor: A Real-World Benchmark for Personalized Travel Planning</a></strong></dt>
<dd>
<p>构建50万真实POI与近4000条真人行程的个性化旅行规划基准，提出可行性/合理性/个性化三维评估与奖励模型对比框架，揭示SOTA LLM行程超越真人比例不足10%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/poi_spatiotemporal_rec/harnessing_large_language_models_for_group_poi_recommendations/">Harnessing Large Language Models for Group POI Recommendations</a></strong></dt>
<dd>
<p>用语义增强 POI Token + QLoRA 双适配器 + 行程目的自监督，破解群体 POI 推荐偏好多样与数据极稀疏难题，HR@10 平均提升 4.4%–9.0%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/spot_trip_dual_preference_driven_out_of_town_trip_recommendation/">SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation</a></strong></dt>
<dd>
<p>首次形式化异地行程推荐，用知识图谱静态偏好+Neural ODE动态偏好解耦建模，F1最高提升17.01%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/poi_spatiotemporal_rec/generative_next_poi_recommendation_with_semantic_id/">Generative Next POI Recommendation with Semantic ID</a></strong></dt>
<dd>
<p>用RQ-VAE将POI多源语义量化为层级语义ID并微调LLaMA3-8B生成下一POI，Acc@1最高提升16%，token开销降55%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/high_precision_audience_expansion_via_extreme_classification_in_a_two_sided/">High Precision Audience Expansion via Extreme Classification in a Two-Sided Marketplace</a></strong></dt>
<dd>
<p>将 Airbnb 位置召回从矩形边界重构为 25M 级 S2 单元极端多分类，离线精度 +11.01%、线上未取消预订者 +0.17%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/poi_spatiotemporal_rec/causalpoi_spatio_temporal_graph_based_causal_modeling_for_cold_start_poi_ch/">CausalPOI: Spatio-Temporal Graph-Based Causal Modeling for Cold-Start POI Check-in Forecasting</a></strong></dt>
<dd>
<p>用功能交互图+结构对齐的干预/对照图做冷启动POI签到量预测与因果提升估计，四区域RMSE最高降57.8%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/poi_spatiotemporal_rec/think2go_generative_next_poi_recommendation_with_llm_reasoning/">Think2Go: Generative Next POI Recommendation with LLM Reasoning</a></strong></dt>
<dd>
<p>统一SFT与RL推理的生成式Next-POI框架，用KDE时空认知不确定性与难度感知奖励差校准优势函数，Acc@1较GNPR提升5.8–7.5%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/poi_spatiotemporal_rec/exploring_the_effect_of_context_awareness_and_popularity_calibration_on_pop/">Exploring the Effect of Context-Awareness and Popularity Calibration on Popularity Bias in POI Recommendations</a></strong></dt>
<dd>
<p>系统评测上下文感知模型与校准流行度(CP)重排在POI推荐中的去偏效果，发现二者组合可在保精度同时最贴近用户流行度画像。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/poi_spatiotemporal_rec/generative_spatiotemporal_intent_sequence_recommendation_via_implicit_reaso/">Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap</a></strong></dt>
<dd>
<p>将LLM显式CoT压缩为隐式潜token并配合时空反事实DPO，在Amap工业数据与线上A/B中实现低延迟意图序列生成。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/poi_spatiotemporal_rec/give_the_long_tail_more_space_promoting_provider_fairness_in_next_poi_recom/">Give the Long-tail More SPACE: Promoting Provider Fairness in Next POI Recommendation</a></strong></dt>
<dd>
<p>以社区推断+非平衡最优传输配额+约束引导潜扩散生成虚拟用户，在物理可达与供给约束下提升长尾POI曝光，公平性最高提升94.11%且精度不降。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/towards_full_candidate_interaction_a_comprehensive_comparison_network_for_b/">Towards Full Candidate Interaction: A Comprehensive Comparison Network for Better Route Recommendation</a></strong></dt>
<dd>
<p>提出比较优先范式，用非重叠路段构造显式比较张量并在成对空间直接推理，线上CR +1.2%、偏航率 -0.8%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/poi_spatiotemporal_rec/comapoi_a_collaborative_multi_agent_framework_for_next_poi_prediction_bridg/">CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction Bridging the Gap Between Trajectory and Language</a></strong></dt>
<dd>
<p>用 Profiler/Forecaster/Predictor 三智能体把数值轨迹转语言画像、动态收缩候选集并反向推理微调，三数据集全指标提升 4%–12%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/sequential_rec/fim_frequency_aware_multi_view_interest_modeling_for_local_life_service_rec/">FIM: Frequency-Aware Multi-View Interest Modeling for Local-Life Service Recommendation</a></strong></dt>
<dd>
<p>多视图搜索分离本地生活多周期需求，FFT 频域感知演化建模周期兴趣，Kuaishou 线上 GMV +8.46%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/poi_spatiotemporal_rec/sep_gcn_leveraging_similar_edge_pairs_with_temporal_and_spatial_contexts_fo/">SEP-GCN: Leveraging Similar Edge Pairs with Temporal and Spatial Contexts for Location-Based Recommender Systems</a></strong></dt>
<dd>
<p>以时空相似交互边对构建边级相似图并注入LightGCN传播，在NYC/Gowalla/Brightkite全库排序上Recall@20最高提升14.89%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/poi_spatiotemporal_rec/rest_a_plug_and_play_spatially_constrained_representation_enhancement_frame/">ReST: A Plug-and-Play Spatially-Constrained Representation Enhancement Framework for Local-Life Recommendation</a></strong></dt>
<dd>
<p>以物品为中心，用属性语义预热+空间约束难负采样对比学习增强长尾物品ID表征，离线AUC最高+5.90%，线上GMV+2.804%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/poi_spatiotemporal_rec/a_survey_on_point_of_interest_recommendation_models_architectures_and_secur/">A Survey on Point-of-Interest Recommendation: Models, Architectures, and Security</a></strong></dt>
<dd>
<p>首个贯通模型演进、架构部署与安全隐私三维度的POI推荐综述，覆盖LDA到LLM/DM、集中式到联邦式及投毒/同态加密防护。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/poi_spatiotemporal_rec/point_of_interest_recommendation_pitfalls_and_viable_solutions/">Point of Interest Recommendation: Pitfalls and Viable Solutions</a></strong></dt>
<dd>
<p>系统梳理POI推荐在数据集、算法与评估三维度的20项陷阱，并提出六方向研究议程与陷阱-方案映射表</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/poi_spatiotemporal_rec/adaptive_location_hierarchy_learning_for_long_tailed_mobility_prediction/">Adaptive Location Hierarchy Learning for Long-Tailed Mobility Prediction</a></strong></dt>
<dd>
<p>提出架构无关插件 ALOHA：LLM+CoT 自动构建城市定制四级语义层级，配合 Gumbel 扰动与节点自适应权重的层级损失，在 6 城数据集上对 RNN/GNN/Transformer/Diffusion 四类骨干一致提升，M@1 最高 +16.59%。</p>
</dd>
</dl>

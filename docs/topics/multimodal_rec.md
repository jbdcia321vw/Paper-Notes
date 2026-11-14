# 🧩 多模态推荐与内容理解 (Multimodal Recommendation)

> **研究范畴**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

当前方向共收录 **129** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/befa_a_general_behavior_driven_feature_adapter_for_multimedia_recommendatio/">BeFA: A General Behavior-driven Feature Adapter for Multimedia Recommendation</a></strong></dt>
<dd>
<p>用行为特征引导的轻量适配器解耦-过滤-重构预训练内容特征，修复信息漂移与遗漏，全库排序下平均提升约 9-11%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/beyond_graph_convolution_multimodal_recommendation_with_topology_aware_mlps/">Beyond Graph Convolution: Multimodal Recommendation with Topology-aware MLPs</a></strong></dt>
<dd>
<p>用拓扑剪枝+邻域对齐损失替代GCN消息传递，以MLP建模多模态物品关系，全库排序下Recall@10最高提升7.53%</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/exploiting_fine_grained_skip_behaviors_for_micro_video_recommendation/">Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation</a></strong></dt>
<dd>
<p>将短视频跳过行为细分为高正/低正/负三级，构建双层正图与分层 BPR 损失，两数据集八项指标全面超越 FRAME 等基线。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/llm_rec/harnessing_multimodal_large_language_models_for_multimodal_sequential_recom/">Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 将商品图转为文本摘要，再以循环式提示推断动态用户偏好，SFT 微调 LLaVA 做多模态序列推荐，AUC 最高提升约 4.3%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/mentor_multi_level_self_supervised_learning_for_multimodal_recommendation/">MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation</a></strong></dt>
<dd>
<p>以 ID 嵌入为锚做四级跨模态分布对齐，叠加特征掩码与图扰动自监督，Amazon 三集 Recall@20 最高提升 74.73%</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/debias_fairness_industrial/multi_modal_recommendation_unlearning_for_legal_licensing_and_modality_cons/">Multi-Modal Recommendation Unlearning for Legal, Licensing, and Modality Constraints</a></strong></dt>
<dd>
<p>首个多模态推荐遗忘框架 MMRecUn，用反向 BPR 衰减遗忘集、正向 BPR 保真保留集，Recall 最高提升 49.85% 且比从头重训快 1.3 倍。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/stair_manipulating_collaborative_and_multimodal_information_for_e_commerce_/">STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation</a></strong></dt>
<dd>
<p>揭示电商场景模态信息弱于协同信号，提出逐步图卷积沿维度分配层权重实现协同-模态共存，并以反向逐步卷积约束更新缓解模态遗忘，三数据集全排序提升2-6%且开销近LightGCN。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/cold_start_rec/motorec_sparse_regularized_multimodal_tokenization_for_cold_start_recommend/">MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation</a></strong></dt>
<dd>
<p>用稀疏正则化 RQ-VAE 将多模态特征离散化为组合语义码，配合稀有度放大与多源图编码，冷启动物品 N@20 最高提升 12.58%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/generative_rec/multi_aspect_cross_modal_quantization_for_generative_recommendation/">Multi-Aspect Cross-modal Quantization for Generative Recommendation</a></strong></dt>
<dd>
<p>在 RQ-VAE 量化各层注入跨模态对比与重建对齐，并在 GR 训练中加隐式/显式对齐，Full-Ranking 下 HR@10 最高提升约 9.4%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/agent_conversational_rec/styletailor_towards_personalized_fashion_styling_via_hierarchical_negative_/">StyleTailor: Towards Personalized Fashion Styling via Hierarchical Negative Feedback</a></strong></dt>
<dd>
<p>首个统一个性化服装设计、购物推荐与虚拟试穿的协作智能体框架，以三级分层负反馈闭环迭代优化，在自建 64 样本基准上 Style Consistency 相对基线提升 39.4%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/multimodal_rec/icg_improving_cover_image_generation_via_mllm_based_prompting_and_personali/">ICG: Improving Cover Image Generation via MLLM-based Prompting and Personalized Preference Alignment</a></strong></dt>
<dd>
<p>用 MLLM 元令牌+双路适配器把个性化上下文注入扩散模型，并以多奖励对齐实现无标签封面生成，下游 MMGCN 推荐 NDCG@10 提升 19.1%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/mira_empowering_one_touch_ai_services_on_smartphones_with_mllm_based_instru/">MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation</a></strong></dt>
<dd>
<p>用结构化推理+模板检索增强+前缀树约束解码，把长按图文触发对象映射到预定义手机AI指令，F1较SFT提升约60%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/muse_a_multimodal_conversational_recommendation_dataset_with_scenario_groun/">Muse: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles</a></strong></dt>
<dd>
<p>首个多模态对话推荐数据集，用场景锚定用户画像的多智能体框架合成 7K 段图文对话，并验证三种 MLLM 的可学习性。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/multimodal_rec/premise_matching_based_prediction_for_accurate_review_recommendation/">PREMISE: Matching-based Prediction for Accurate Review Recommendation</a></strong></dt>
<dd>
<p>以多尺度多字段语义匹配分数替代跨模态融合表征，用于多模态评论有用性排序，MAP 较 GBDT 提升 5–17 点且训练提速 42%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/re2a_situated_conversational_recommendation_via_rubric_based_preference_rea/">Re2A: Situated Conversational Recommendation via Rubric-based Preference Reasoning and Alignment</a></strong></dt>
<dd>
<p>将情境对话推荐重构为“先推理后对齐”两阶段，用自动评分标准引导偏好状态推理并以异构负例 DPO 实现用户-场景双重对齐，SIMMC 2.1 上 Hit@1 提升 8.17pt。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/multimodal_rec/reasonrec_a_reasoning_augmented_multimodal_agent_for_unified_recommendation/">ReasonRec: A Reasoning-Augmented Multimodal Agent for Unified Recommendation</a></strong></dt>
<dd>
<p>将推荐重构为 Observe-Deliberate-Act 三阶段 CoT 推理智能体，以证据视界课程学习与不确定性引导工具委派，在五数据集四任务上 HR@5 相对提升超 30% 并降低推理延迟。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/sequential_rec/two_sided_state_space_models_for_sequential_recommendation_with_non_random_/">Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback</a></strong></dt>
<dd>
<p>将评论视为双侧状态空间中的内生事件，用 MNAR 观测模式、非对称评论结转与局部图传播联合演化用户/物品状态，Full-Ranking 下 R@20 较 BSARec 提升 14.8%–18.8%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conve/">Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 SiPeR 框架，用场景转移估计解决“去哪”与贝叶斯逆向推理解决“要什么”，在 SIMMC 2.1 与 SCREEN 上 R@1 平均提升约 10.9%/10.6%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/dual_phase_playtime_guided_recommendation_interest_intensity_exploration_an/">Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks</a></strong></dt>
<dd>
<p>用双 Beta 分布解耦游玩时长强弱兴趣，并以多模态随机游走做跨类目探索，在 Steam 数据上同时提升准确率与覆盖度。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/generating_negative_samples_for_multi_modal_recommendation/">Generating Negative Samples for Multi-Modal Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 三步提示生成兼具凝聚力与难度的属性级负样本，并以因果学习解耦关键特征，全库排序下 Recall@10 最高提升 9.02%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/i_3_mrec_invariant_learning_with_information_bottleneck_for_incomplete_moda/">I$^3$-MRec: Invariant Learning with Information Bottleneck for Incomplete Modality Recommendation</a></strong></dt>
<dd>
<p>将各模态视为独立语义环境做 IRM 不变学习，并用 IB 压缩模态特有信息，在模态缺失场景下 Full-Ranking Recall@20 稳定提升 2%–4%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/refining_contrastive_learning_and_homography_relations_for_multi_modal_reco/">Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation</a></strong></dt>
<dd>
<p>用元网络+正交约束精炼多模态对比学习，并补建用户兴趣图与物品共现图，全库排序下 R@20 最高提升约 9.8%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_modality_harmony_orthogonal_purification_and_topology_guided_moe_for/">Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation</a></strong></dt>
<dd>
<p>以纯协同锚点为几何参考轴正交分解并保能截断模态噪声，配合解耦 Sigmoid 路由 MoE 与冲突感知 safe-SSL，在 Amazon 三数据集 8:1:1 划分下 R@20 最高提升 6.55%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_noisy_signals_dual_level_denoising_for_multi_modal_sequential_recomm/">Beyond Noisy Signals: Dual-Level Denoising for Multi-modal Sequential Recommendation</a></strong></dt>
<dd>
<p>图拉普拉斯低通滤波去特征噪声 + FFT 可学习频域滤波去序列噪声，四数据集 Full-Ranking 全面 SOTA，R@20 最高 +19.33%。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/one_graph_multiple_gains_single_high_quality_item_item_graph_for_multimodal/">One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation</a></strong></dt>
<dd>
<p>构建一次高质量物品-物品图，经三元闭包重加权后复用于表征、交互图与优化三阶段，四数据集全指标SOTA且训练更快更省显存。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2026/debias_fairness_industrial/tru_targeted_reverse_update_for_efficient_multimodal_recommendation_unlearn/">TRU: Targeted Reverse Update for Efficient Multimodal Recommendation Unlearning</a></strong></dt>
<dd>
<p>揭示多模态推荐遗忘中排序/模态/模块三层非均匀性，提出门控+分支缩放+参数组隔离的靶向反向更新，逼近重训练前沿且收敛更快。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/a_content_driven_micro_video_recommendation_dataset_at_scale/">A Content-Driven Micro-Video Recommendation Dataset at Scale</a></strong></dt>
<dd>
<p>发布十亿级含原始视频模态的微视频推荐数据集 MicroLens，并系统基准 IDRec/VIDRec/VideoRec 三类范式，揭示端到端视频编码优于冻结特征的关键结论。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/bilisting_modality_alignment_for_listings/">BiListing: Modality Alignment for Listings</a></strong></dt>
<dd>
<p>用 LLM 生成视觉画像 + PhotoSet Transformer 聚合多图，将 CLIP 图文对齐扩展到多图多文本的房源实体，线上 NDCG +0.425%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/distribution_guided_auto_encoder_for_user_multimodal_interest_cross_fusion/">Distribution-Guided Auto-Encoder for User Multimodal Interest Cross Fusion</a></strong></dt>
<dd>
<p>以行为粒度做模态内/模态间交叉融合，并用兴趣分布解码器作自监督约束，工业 A/B 上 CTR +12.9%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/do_recommender_systems_really_leverage_multimodal_content_a_comprehensive_a/">Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation</a></strong></dt>
<dd>
<p>用受控噪声实验证明多模态增益源于语义而非模型容量，并以 LVLM 结构化提示生成「原生多模态」[EOS] 嵌入替代晚期融合，Borda 聚合排名第一。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/efficient_multimodal_streaming_recommendation_via_expandable_side_mixture_o/">Efficient Multimodal Streaming Recommendation via Expandable Side Mixture-of-Experts</a></strong></dt>
<dd>
<p>冻结 BERT/ViT 主干，用可扩展侧调 MoE 增量吸收新时段知识并按利用率剪枝，在三个流式数据集上 HR@10 全面超越 SOTA 且显存/训练时间更省。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/generative_rec/empowering_large_language_model_for_sequential_recommendation_via_multimoda/">Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs</a></strong></dt>
<dd>
<p>揭示 LLM4SR 的嵌入坍缩与语义ID灾难性遗忘，用多模态嵌入+MMD重建+码本初始化破局，全库排序下 nDCG@5 提升 4.4%–10.5%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/hypercomplex_prompt_aware_multimodal_recommendation/">Hypercomplex Prompt-aware Multimodal Recommendation</a></strong></dt>
<dd>
<p>用 Cayley-Dickson 超复数多分量嵌入 + 超复数乘法非线性融合 + Prompt 动态补偿，在四个 Amazon 数据集 Full-Ranking 下 Recall@20 最高提升约 13%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/miss_multi_modal_tree_indexing_and_searching_with_lifelong_sequential_behav/">MISS: Multi-Modal Tree Indexing and Searching with Lifelong Sequential Behavior for Retrieval Recommendation</a></strong></dt>
<dd>
<p>用多模态嵌入构建树索引，并以 Co-GSU/MM-GSU 双视角检索终身行为，快手线上召回@800 相对提升 37.9%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/modality_alignment_with_multi_scale_bilateral_attention_for_multimodal_reco/">Modality Alignment with Multi-scale Bilateral Attention for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 MambaRec，用多尺度空洞卷积+双注意力做局部图文对齐、MMD+InfoNCE 做全局分布对齐，Amazon 三数据集 Recall@20 最高提升约 5.4%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cold_start_rec/multi_modal_adaptive_mixture_of_experts_for_cold_start_recommendation/">Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation</a></strong></dt>
<dd>
<p>用双层 MoE（模态专属专家+动态门控融合）替代拼接/平均融合，在 Amazon 三域冷启动全库排序上 Recall@10 最高提升 16.17%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/generative_rec/progressive_semantic_residual_quantization_for_multimodal_joint_interest_mo/">Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation</a></strong></dt>
<dd>
<p>提出前缀语义约束的渐进残差量化 PSRQ 与多码本交叉注意力 MCCA，缓解语义漂移与跨模态建模缺口，冷启动 AUC 显著提升并完成工业上线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multimodal_rec/qarm_quantitative_alignment_multi_modal_recommendation_at_kuaishou/">QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou</a></strong></dt>
<dd>
<p>用业务 item-item 对微调 MLLM 做表示对齐，再以 VQ/RQ 量化成语义码实现端到端可学习，快手广告与电商线上 GMV+2.296%、冷启曝光+5.324%</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/bridge_behavior_guided_residual_integration_with_dual_frequency_graph_evide/">BRIDGE: Behavior-Guided Residual Integration with Dual-Frequency Graph Evidence</a></strong></dt>
<dd>
<p>揭示跨视图对齐存在有效区间与频谱共享/私有分裂，提出双频图证据+行为符号残差仅在 top-K 候选内校准，Baby/Sports/Electronics 全库排序 R@20 最高提升 7.3%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/information_guided_selective_modality_interest_alignment_for_multimodal_rec/">Information-Guided Selective Modality-Interest Alignment for Multimodal Recommendation</a></strong></dt>
<dd>
<p>以互信息/条件熵分解为理论视角，用行为校准图精炼+兴趣对比+共享子空间选择性对齐，三数据集 Full-Ranking 全面 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/native_multimodal_representation_learning_for_click_through_rate_prediction/">Native Multimodal Representation Learning for Click-Through Rate Prediction in E-Commerce Scenarios</a></strong></dt>
<dd>
<p>揭示端到端联合训练在强预训练编码器上失效的根因（CTR 监督多模态不可解释+梯度不一致），提出 Mine-Then-Train 挖掘三元组微调编码器，线上 CTR +1.5%、RPM +0.5%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multimodal_rec/sequential_modality_dropout_for_robust_multi_modal_sequential_recommendatio/">Sequential Modality Dropout for Robust Multi-Modal Sequential Recommendation</a></strong></dt>
<dd>
<p>在融合点注入逐样本伯努利模态掩码，四行代码即插即用，使四种多模态序列骨干在模态缺失下 HR@10 留存率提升 1.0–3.2×。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/multi_task_reranking/sequential_multimodal_evidence_optimization_for_product_media_ranking_in_e_/">Sequential Multimodal Evidence Optimization for Product Media Ranking in E-Commerce</a></strong></dt>
<dd>
<p>将商品媒体排序建模为证据累积序列决策，两阶段解耦效用学习与生存加权自回归策略，离线 DR-CVR +5.4%、MSC −15.3%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/sequential_rec/empowering_sequential_recommendation_from_collaborative_signals_and_semanti/">Empowering Sequential Recommendation from Collaborative Signals and Semantic Relatedness</a></strong></dt>
<dd>
<p>双流 Transformer 分别编码 ID 与内容序列，用用户级/物品级分层对比对齐跨模态语义鸿沟，全库排序下 NDCG@10 最高提升 15.44%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/multimodal_rec/truthsr_trustworthy_sequential_recommender_systems_via_user_generated_multi/">TruthSR: Trustworthy Sequential Recommender Systems via User-generated Multimodal Content</a></strong></dt>
<dd>
<p>用评论-图片跨模态一致性降噪 + 主观逻辑证据融合做可信序列推荐，四数据集 Full-Ranking 全面超越 MMSRec。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/multimodal_rec/are_multimodal_embeddings_truly_beneficial_for_recommendation_a_deep_dive_i/">Are Multimodal Embeddings Truly Beneficial for Recommendation? A Deep Dive into Whole vs. Individual Modalities</a></strong></dt>
<dd>
<p>用模态剔除（常量/噪声替换）在14个模型上做受控实证，发现文本主导、图像近乎无效，简单融合基线增益源于架构伪影。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2026/multimodal_rec/cammsr_category_guided_attentive_mixture_of_experts_for_multimodal_sequenti/">CAMMSR: Category-Guided Attentive Mixture of Experts for Multimodal Sequential Recommendation</a></strong></dt>
<dd>
<p>用类别预测任务引导注意力 MoE 动态分配模态权重，并以模态交换对比学习对齐跨模态序列表征，Full-Ranking 下 NDCG@5 最高提升 15.3%。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2026/multimodal_rec/gala_generative_aligned_learning_for_adaptive_multimodal_representation_in_/">GALA: Generative Aligned Learning for Adaptive Multimodal Representation in the Taobao Shangou Recommender System</a></strong></dt>
<dd>
<p>三阶段流水线：行为三元组预训练 + GRPO 生成式行为对齐 + 自适应门控融合，离线 AUC +0.12/+0.20，线上订单量 +0.55%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/multimodal_rec/fuse_failure_aware_usage_of_subagent_evidence_for_multimodal_search_and_rec/">FUSE: Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation</a></strong></dt>
<dd>
<p>用紧凑 GDR 替代原始图像提示，系统对比 7 种上下文预算策略并以五阶段归因层定位失败，Context Compression 在意图/路由/召回/NDCG@5 全面最优且 p95 延迟降 45%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/llm_rec/scaling_multimodal_search_and_recommendation_with_small_language_models_via/">Scaling Multimodal Search and Recommendation with Small Language Models via Upside-Down Reinforcement Learning</a></strong></dt>
<dd>
<p>用 Llama-3 合成 5200 万意图-提示对蒸馏进 104M nanoGPT，并以倒置强化学习实现长度/模态可控的多任务提示生成，性能逼近 8B 模型而显存仅 500MB。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/generative_rec/multimodal_quantitative_language_for_generative_recommendation/">Multimodal Quantitative Language for Generative Recommendation</a></strong></dt>
<dd>
<p>用 RQ-VAE 将多域多模态物品翻译为统一量化语言，配合多任务预训练微调实现知识迁移，全库排序下 NDCG@10 最高提升 14.82%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/multimodal_rec/structured_spectral_reasoning_for_frequency_adaptive_multimodal_recommendat/">Structured Spectral Reasoning for Frequency-Adaptive Multimodal Recommendation</a></strong></dt>
<dd>
<p>将多模态图信号分解为等能量频带，用频带掩码一致性训练、低秩超谱算子与谱对比正则实现频率自适应融合，全库排序下冷启动 Recall@20 提升约 10%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/multimodal_rec/visuallens_personalization_through_task_agnostic_visual_history/">VisualLens: Personalization through Task-Agnostic Visual History</a></strong></dt>
<dd>
<p>用任务无关的日常照片历史构建“图像-描述-方面词”谱系画像，经检索+网格化+联合微调实现个性化推荐，Hit@3 超 UniMP 5-10%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multimodal_rec/can_complementary_signals_bridge_similarity_islands_manifold_augmented_grap/">Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 MAGE，用离线 Soft Conceptor 残差注入 + 在线非对称双流 GNN 桥接互补品“相似性孤岛”，四个 Amazon 数据集 Full-Ranking 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multimodal_rec/controllable_and_content_based_recommendations/">Controllable and Content-Based Recommendations</a></strong></dt>
<dd>
<p>用多模态 LLM 将图像/音频/视频条目转述为文本画像，经标签监督与 InfoNCE 对齐冻结 CF 骨干，实现可读可编辑的文本瓶颈控制。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multimodal_rec/intentlens_grounding_underspecified_multimodal_queries_for_recommendation_v/">IntentLens: Grounding Underspecified Multimodal Queries for Recommendation via Tool-Augmented Reasoning</a></strong></dt>
<dd>
<p>用 MLLM 工具增强推理将欠指定图文查询接地为结构化意图，再以轻量双塔排序，构建两个新基准并显著超越 MLLM 与检索基线。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/agent_conversational_rec/siturecbench_evaluating_frontier_models_as_situated_recommendation_assistan/">SituRecBench: Evaluating Frontier Models as Situated Recommendation Assistants in 3D Environments</a></strong></dt>
<dd>
<p>构建 UE5 三维购物环境与共享第一人称视角的交互式基准，统一评测 VLM 助手的对话质量、推荐效用与导航有效性。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/generative_rec/syngr_unleashing_the_potential_of_cross_modal_synergy_for_generative_recomm/">SynGR: Unleashing the Potential of Cross-Modal Synergy for Generative Recommendation</a></strong></dt>
<dd>
<p>用 PID 信息分解揭示多模态生成式推荐的协同信息缺失，提出显著性感知掩码+协同对比学习，Full-Ranking 下平均提升约 10%</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/token_efficient_item_representation_via_images_for_llm_recommender_systems/">Token-Efficient Item Representation via Images for LLM Recommender Systems</a></strong></dt>
<dd>
<p>用单张物品图像替代冗长文本描述作为 LLM 推荐输入，配合 RISA 对齐与 RERI 检索式推荐，在 Amazon 四域上兼顾效率与效果。</p>
</dd>
<dt><strong><a href="../../venues/IJCAI/2026/sequential_rec/rethinking_convolutional_networks_for_attribute_aware_sequential_recommenda/">Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation</a></strong></dt>
<dd>
<p>用金字塔式降尺度卷积替代自注意力聚合属性感知序列，线性复杂度下在四个 Amazon 数据集上 HR@10 最高提升 4.08%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/graph_rec/collaborative_filtering_meets_spectrum_shift_connecting_user_item_interacti/">Collaborative Filtering Meets Spectrum Shift: Connecting User-Item Interaction with Graph-Structured Side Information</a></strong></dt>
<dd>
<p>揭示增广图频谱右移现象，提出仅含平移/缩放两因子的 SSC 校正，即插即用提升 LightGCN/JGCF 最高 23%，零额外开销。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multimodal_rec/findrec_stein_guided_entropic_flow_for_multi_modal_sequential_recommendatio/">FindRec: Stein-Guided Entropic Flow for Multi-Modal Sequential Recommendation</a></strong></dt>
<dd>
<p>用 Stein 核对齐图文与 ID 流分布、MoE 路由动态筛选跨模态信号，Mamba 建模长序列，三数据集 NDCG@5 相对提升 1.4–3.1%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/agent_conversational_rec/lavic_adapting_large_vision_language_models_to_visually_aware_conversationa/">LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation</a></strong></dt>
<dd>
<p>用视觉知识自蒸馏把商品图压成5个[CLS] token，再做推荐提示微调，在新建Reddit-Amazon三域上HR@1最高提升54.2%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multimodal_rec/mdvt_enhancing_multimodal_recommendation_with_model_agnostic_multimodal_dri/">MDVT: Enhancing Multimodal Recommendation with Model-Agnostic Multimodal-Driven Virtual Triplets</a></strong></dt>
<dd>
<p>用多模态融合表征动态构造虚拟三元组作为额外监督信号，配合三种预热阈值策略与增强 pairwise 损失，即插即用缓解数据稀疏，稀疏用户组 NDCG@10 最高提升 12.85%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multimodal_rec/notellm_2_multimodal_large_representation_models_for_recommendation/">NoteLLM-2: Multimodal Large Representation Models for Recommendation</a></strong></dt>
<dd>
<p>端到端微调 LLM+视觉编码器构建高效 MLRM，用 mICL 与门控后期融合缓解视觉忽视，I2I 全库召回 R@100 显著提升并落地线上 A/B。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/pcr_ca_parallel_codebook_representations_with_contrastive_alignment_for_mul/">PCR-CA: Parallel Codebook Representations with Contrastive Alignment for Multiple-Category App Recommendation</a></strong></dt>
<dd>
<p>用并行码本VQ-AE替代RQ-VAE层级量化建模多类别App语义，配合用户/物品双层对比对齐与双流注意力，微软商店线上CTR+10.52%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multimodal_rec/text_guided_visual_representation_learning_for_robust_multimodal_e_commerce/">Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation</a></strong></dt>
<dd>
<p>以结构化元数据初始化语义查询、双流解耦视觉证据并用可靠性双门控校准，在 134K 全库 I2I 检索上 H@100 相对提升 6.04%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multimodal_rec/uninote_a_unified_embedding_model_for_multimodal_representation_and_ranking/">UniNote: A Unified Embedding Model for Multimodal Representation and Ranking</a></strong></dt>
<dd>
<p>小红书工业级 I2I 统一嵌入模型：对比 SFT + GRPO 相对重排两阶段训练，10 类检索任务全 SOTA，MRL 支持弹性维度部署。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multimodal_rec/vi_mmrec_similarity_aware_training_cost_free_virtual_user_item_interactions/">VI-MMRec: Similarity-Aware Training Cost-free Virtual User-Item Interactions for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用模态相似度构造虚拟交互矩阵并以统计权重融合，零训练开销即插即用地缓解多模态推荐数据稀疏，Full-Ranking 下 Recall@10 最高提升 22.25%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/affect_aware_cross_domain_recommendation_for_art_therapy_via_music_preferen/">Affect-aware Cross-Domain Recommendation for Art Therapy via Music Preference Elicitation</a></strong></dt>
<dd>
<p>以音乐偏好诱导替代纯视觉诱导，提出 Mozart/Haydn/Salieri 三种情感感知跨域艺术治疗推荐，200 人用户研究验证情绪改善。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/aps_explorer_navigating_algorithm_performance_spaces_for_informed_dataset_s/">APS Explorer: Navigating Algorithm Performance Spaces for Informed Dataset Selection</a></strong></dt>
<dd>
<p>将 29 算法 × 96 数据集的性能向量投影为可交互 PCA 性能空间，用可视化工具支撑可辩护的数据集选择。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/beyond_the_past_leveraging_audio_and_human_memory_for_sequential_music_reco/">Beyond the past: Leveraging Audio and Human Memory for Sequential Music Recommendation</a></strong></dt>
<dd>
<p>用音频编码器预测新歌的 ACT-R 类激活值，将记忆式重复建模扩展到未听歌曲，NDCG_Exp 相对 PISA 提升约 40%，RepBias 降至 0.09%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/image_seeking_intent_prediction_for_cross_device_product_search/">Image-Seeking Intent Prediction for Cross-Device Product Search</a></strong></dt>
<dd>
<p>用语音查询+商品摘要预测图像意图，以可微精度损失抑制误触发，离线 Precision 达 78.5。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/improving_visual_recommendation_on_e_commerce_platforms_using_vision_langua/">Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models</a></strong></dt>
<dd>
<p>在 Mercari C2C 平台微调 SigLIP 图像编码器做视觉相似召回，离线 nDCG@5 +9.1%，线上 CTR +50%、CVR +14%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/just_ask_for_music_jam_multimodal_and_personalized_natural_language_music_r/">Just Ask for Music (JAM): Multimodal and Personalized Natural Language Music Recommendation</a></strong></dt>
<dd>
<p>将查询建模为用户到物品的向量平移，用交叉注意力聚合音频/歌词/CF 多模态，在 JAMSessions 全库排序上 NDCG@10 相对 TalkRec 提升约 105%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/llm_powered_nuanced_video_attribute_annotation_for_enhanced_recommendations/">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></strong></dt>
<dd>
<p>用 Gemini 多模态 LLM 做短视频“氛围”属性标注，经金/银集评估与蒸馏扩量，接入个性化受限召回，线上参与度 +0.49%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/personalized_image_generation_for_recommendations_beyond_catalogs/">Personalized Image Generation for Recommendations Beyond Catalogs</a></strong></dt>
<dd>
<p>用轻量条件扩散先验从隐式反馈采样用户个性化 CLIP 嵌入，冻结 SD+IP-Adapter 解码，免微调实现可扩展个性化图像生成。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/vl_clip_enhancing_multimodal_recommendations_via_visual_grounding_and_llm_a/">VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings</a></strong></dt>
<dd>
<p>用 Grounding DINO 裁剪商品主体 + LLM 自反思改写商品描述，再微调 CLIP 双塔，Walmart 线上 CTR +18.6%、GMV +4.0%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/yambda_5b_a_large_scale_multi_modal_dataset_for_ranking_and_retrieval/">Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval</a></strong></dt>
<dd>
<p>发布 Yandex Music 4.79B 交互工业级数据集，含音频嵌入与 is_organic 标志，提出 Global Temporal Split 评测协议并给出 6 类基线基准。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multimodal_rec/binge_watch_reproducible_multimodal_benchmarks_datasets_for_large_scale_mov/">Binge Watch: Reproducible Multimodal Benchmark Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M</a></strong></dt>
<dd>
<p>为 ML-10M/20M 构建可复现多模态基准 M3L（剧情/海报/预告片文本-图像-视频-音频），保留&gt;90%交互并验证音频模态的关键增益。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/cold_start_rec/idproxy_ctr_prediction_with_multimodal_llms_for_cold_start_recommendation_a/">IDProxy: CTR Prediction with Multimodal LLMs for Cold-Start Recommendation at Xiaohongshu</a></strong></dt>
<dd>
<p>用 MLLM 生成代理 ID 嵌入，经粗到细两阶段对齐工业 CTR 排序器，新笔记在线 AUC 提升约 2 倍于全局流量。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/cold_start_rec/learning_sparse_representations_of_multimodal_content_for_enhanced_cold_ite/">Learning Sparse Representations of Multimodal Content for Enhanced Cold Item Recommendation</a></strong></dt>
<dd>
<p>将多模态内容冷启动表征改为 top-K 稀疏嵌入，并用 α-entmax 预稀疏激活诱导锐化与去噪，冷启动 NDCG@20 提升 16.6%–75.5% 且存储更低。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multimodal_rec/mend_the_measurement_gap_latent_user_preference_modeling_for_short_form_vid/">Mend the Measurement Gap: Latent User Preference Modeling for Short-Form Video Recommendation</a></strong></dt>
<dd>
<p>将异构行为反馈视为隐价值状态的噪声测量，用受限基线+因子化路由瓶颈解耦时长混杂，YouTube Shorts 线上主指标提升 2.67%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/cohesion_composite_graph_convolutional_network_with_dual_stage_fusion_for_m/">COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用行为模态早期精炼+晚期注意力融合的双阶段融合，配合异质/同质复合图与自适应BPR，在三个 Amazon 数据集上 R@20 相对提升 4.4%–6.1%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/sequential_rec/diff_dual_side_information_filtering_and_fusion_for_sequential_recommendati/">DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation</a></strong></dt>
<dd>
<p>频域滤波去噪 + 早期/中间双重融合的侧信息序列推荐，Full-Ranking 下 R@20 最高提升 14.1%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/diffusion_based_multi_modal_synergy_interest_network_for_click_through_rate/">Diffusion-based Multi-modal Synergy Interest Network for Click-through Rate Prediction</a></strong></dt>
<dd>
<p>用 PLE 式共享/独有专家解耦模态知识、以扩散式多步跨模态去噪捕捉协同信号，FDAF 非侵入融合，四数据集 AUC 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/disentangling_and_generating_modalities_for_recommendation_in_missing_modal/">Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios</a></strong></dt>
<dd>
<p>将模态特征解耦为通用/特有分量并用互信息损失分离，再以跨模态对齐特征与用户模态偏好生成缺失模态，缺失场景全库排序 R@20 相对提升 4.7%–8.2%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/social_group_rec/frog_effective_friend_recommendation_in_online_games_via_modality_aware_use/">FROG: Effective Friend Recommendation in Online Games via Modality-aware User Preferences</a></strong></dt>
<dd>
<p>面向在线游戏好友推荐，用模态感知双向匹配+局部/全局双路注意力建模用户偏好，离线 HR/NDCG 最高提升 15.82%/14.59%，线上 accRate 提升 33.3%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/mscrs_multi_modal_semantic_graph_prompt_learning_framework_for_conversation/">MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>构建协同/文本/图像三模态语义图并以共享嵌入融合，图提示注入 DialoGPT，ReDial 与 INSPIRED 全指标 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/multi_modal_multi_behavior_sequential_recommendation_with_conditional_diffu/">Multi-Modal Multi-Behavior Sequential Recommendation with Conditional Diffusion-Based Feature Denoising</a></strong></dt>
<dd>
<p>用条件扩散分别对模态与行为特征去噪，并以多专家网络解耦共性/特性兴趣，在 Rec-Tmal/Kuaishou 上 HR@10 提升约 4-8%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/short_video_segment_level_user_dynamic_interests_modeling_in_personalized_r/">Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation</a></strong></dt>
<dd>
<p>将短视频切分为5秒片段，用CLIP视觉+ID双模态跨注意力建模片段级动态兴趣，以段内排序损失解决标签缺失，在跳看预测与视频推荐两任务上大幅超越基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multimodal_rec/u_sticker_a_large_scale_multi_domain_user_sticker_dataset_for_retrieval_and/">U-Sticker: A Large-Scale Multi-Domain User Sticker Dataset for Retrieval and Personalization</a></strong></dt>
<dd>
<p>构建含 22K 用户、370K 贴纸、8.3M 消息的多领域多语言贴纸数据集，支撑个性化贴纸检索与用户建模。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 1.5B LLaMA2+BLIP-2 将图像转字幕并 token 化为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%，延迟中性。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recomm/">CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation</a></strong></dt>
<dd>
<p>将文本/图像/协同信号渲染为统一视觉语义单元，并用可逆非均匀变换校正偏斜分布，提升码本利用率与生成式推荐精度。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/discrete_preference_learning_for_personalized_multimodal_generation/">Discrete Preference Learning for Personalized Multimodal Generation</a></strong></dt>
<dd>
<p>用模态专属LightGCN+RQ-VAE把协同偏好量化成离散token注入LLM与扩散模型，配合跨模态一致性奖励实现个性化图文生成。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/dual_diffusional_generative_fashion_recommendation/">Dual-Diffusional Generative Fashion Recommendation</a></strong></dt>
<dd>
<p>双分支扩散 Transformer 联合生成时尚单品图像与属性级文本，以结构化偏好加权采样建模用户，PFITB/GOR 双任务个性化指标显著领先且训练提速约 7 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/from_raw_features_to_effective_embeddings_a_three_stage_approach_for_multim/">From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation</a></strong></dt>
<dd>
<p>用 VLM/LLM 摘要 + LightGCN 传播 + 跨视图对比学习三阶段精炼食谱多模态特征，Full-Ranking 下 Recall@10 提升 7–15%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/mllmrec_a_preference_reasoning_paradigm_with_graph_refinement_for_multimoda/">MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 推理用户偏好画像替代噪声聚合初始化，并以阈值去噪+共现增强精修物品图，三数据集 11/12 指标 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/multi_modal_relational_item_representation_learning_for_inferring_substitut/">Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items</a></strong></dt>
<dd>
<p>融合多模态基础模型与元路径行为编码，以自监督对比去噪与 LLM 校验弱监督推断可替代/互补关系，M@10 平均提升 26.1%/39.2%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/multimodal_large_language_models_with_adaptive_preference_optimization_for_/">Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation</a></strong></dt>
<dd>
<p>面向 MLLM 序列推荐，用难度感知重加权 DPO + 高斯噪声正则分布优化，缓解样本难度失衡与跨模态语义偏差，三数据集 NDCG@3 提升 7.4%–11.0%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/robust_multimodal_recommendation_via_graph_retrieval_enhanced_modality_comp/">Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion</a></strong></dt>
<dd>
<p>用模态感知子图检索替代 K 跳邻居聚合，配合图 Transformer 与稀疏路由码本补全缺失模态，40% 缺失率下 Recall@20 相对提升 4.2%–5.1%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/structural_and_disentangled_adaptation_of_large_vision_language_models_for_/">Structural and Disentangled Adaptation of Large Vision Language Models for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用模态内结构软教师对齐 LVLM 跨模态表征，并以专家化门控 LoRA 解耦梯度流，即插即用提升多模态与序列推荐，长尾增益最高 18.70%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/timemm_time_as_operator_spectral_filtering_for_dynamic_multimodal_recommend/">TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation</a></strong></dt>
<dd>
<p>将交互新近度映射为时间核加权图算子族，隐式实现免特征分解的多尺度谱滤波，配合自适应算子混合与模态路由，全库排序下 R@20 相对提升 3%–10%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用交互物品模态特征+聚类全局语义免训练初始化用户表征，弥合用户-物品语义鸿沟，六模型四数据集Recall@5最高提升36.4%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/multimodal_rec/efficient_and_effective_adaptation_of_multimodal_foundation_models_in_seque/">Efficient and Effective Adaptation of Multimodal Foundation Models in Sequential Recommendation</a></strong></dt>
<dd>
<p>IISAN-Versa 用解耦 PEFT + 组 LayerDrop + 维度变换层，统一适配对称/非对称多模态基础模型，缓存隐藏态后 GPU 显存降 12–15 倍、训练提速 16–20 倍，MicroLens 达 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/multimodal_rec/tackling_the_modality_inhibitance_effect_with_wavelet_based_modality_integr/">Tackling the Modality Inhibitance Effect with Wavelet-based Modality Integration in Multi-modal Recommendation</a></strong></dt>
<dd>
<p>提出模态抑制效应(MIE)概念，用Haar小波多分辨率分解提取各模态本质特征、结构感知多模态图学习与交叉加权自适应调节器，在4数据集Full-Ranking下平均提升6.48%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/multimodal_rec/training_free_graph_based_imputation_of_missing_modalities_in_multimodal_re/">Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation</a></strong></dt>
<dd>
<p>将缺失模态重铸为物品-物品共购图上的图特征插值，提出四种免训练传播式填补算子，可即插即用并保持甚至扩大多模态与传统推荐的性能差距。</p>
</dd>
<dt><strong><a href="../../venues/TMM/2025/multimodal_rec/knowledge_aware_diffusion_enhanced_multimedia_recommendation/">Knowledge-aware Diffusion-Enhanced Multimedia Recommendation</a></strong></dt>
<dd>
<p>用重启随机游走 Jaccard 注意力矩阵替代可学习注意力，并以用户嵌入引导扩散模型重建低噪知识图谱构建对比视图，三数据集 Recall@20 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/TMM/2026/multimodal_rec/cross_modal_attention_network_with_dual_graph_learning_in_multimodal_recomm/">Cross-Modal Attention Network with Dual Graph Learning in Multimodal Recommendation</a></strong></dt>
<dd>
<p>递归跨模态注意力 RCA 迭代对齐图文特征，配合用户-物品异构图与物品-物品语义图双图 + 对比学习，四数据集平均提升约 5%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/debias_fairness_industrial/causality_inspired_fair_representation_learning_for_multimodal_recommendati/">Causality-Inspired Fair Representation Learning for Multimodal Recommendation</a></strong></dt>
<dd>
<p>因果启发的模态解耦与公平/不公平关系挖掘，实现多模态推荐的反事实公平，公平性显著优于 SOTA 且精度可比。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/multimodal_rec/formalizing_multimedia_recommendation_through_multimodal_deep_learning/">Formalizing Multimedia Recommendation through Multimodal Deep Learning</a></strong></dt>
<dd>
<p>首次将多媒体推荐对齐多模态深度学习五步流水线，提出 Which/How/When 三问形式化 Schema，并在 Elliot 上统一复现 10 个模型、兼顾精度与超越精度指标。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/multimodal_rec/adopting_state_of_the_art_pretrained_audio_representations_for_music_recomm/">Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems</a></strong></dt>
<dd>
<p>系统评测 9 个 MIR 预训练音频表征在 5 类推荐器上的热/冷启动表现，揭示 MIR 性能无法直接迁移到推荐任务。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/graph_rec/an_aspect_performance_aware_hypergraph_neural_network_for_review_based_reco/">An Aspect Performance-aware Hypergraph Neural Network for Review-based Recommendation</a></strong></dt>
<dd>
<p>用四元组超图建模用户-物品-方面-情感极性，以用户偏好加权聚合冲突情感推断物品方面表现，六数据集 MSE 平均降 2.30%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/multimodal_rec/spectrum_based_modality_representation_fusion_graph_convolutional_network_f/">Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation</a></strong></dt>
<dd>
<p>将多模态特征投影至频域做滤波去噪与点积融合，配合模态专属/融合双图学习与模态感知偏好模块，在 Amazon 三数据集全库排序下 Recall@20 相对最优基线提升约 5%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/multimodal_rec/teach_me_how_to_denoise_a_universal_framework_for_denoising_multi_modal_rec/">Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration</a></strong></dt>
<dd>
<p>揭示 ID 空间去噪假设在多模态空间失效，提出 AMSC 重校准 + DBPR + OT 蒸馏的即插即用去噪框架，全库排序下 Recall@20 最高提升约 11%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/comment_staytime_prediction_with_llm_enhanced_comment_understanding/">Comment Staytime Prediction with LLM-enhanced Comment Understanding</a></strong></dt>
<dd>
<p>首个评论区停留时长数据集 KuaiComt 与两阶段 LCU 框架：LLM 领域微调 + 评论排序辅助任务，离线全指标提升且线上停留时长 +1.27%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cold_start_rec/crown_a_novel_approach_to_comprehending_users_preferences_for_accurate_pers/">CROWN: A Novel Approach to Comprehending Users&#x27; Preferences for Accurate Personalized News Recommendation</a></strong></dt>
<dd>
<p>以类别引导意图解耦、标题-内容一致性加权与 GNN 混合用户表征，统一解决新闻推荐的多意图、读后偏好差异与冷启动三大难题。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/ctr_driven_advertising_image_generation_with_multimodal_large_language_mode/">CTR-Driven Advertising Image Generation with Multimodal Large Language Models</a></strong></dt>
<dd>
<p>用 MLLM 双分支奖励模型把 CTR 相对比较转化为偏好信号，结合产品中心偏好优化 PCPO 驱动广告图生成，线上 CTR 提升 2%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/sequential_rec/flare_fusing_language_models_and_collaborative_architectures_for_recommende/">FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancement</a></strong></dt>
<dd>
<p>用冻结 mT5+Perceiver 将文本注入 Bert4Rec，并以重调 Bert4Rec 基线、大词表评测与 Critiquing 任务重估 ID+文本增益。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/generating_with_fairness_a_modality_diffused_counterfactual_framework_for_i/">Generating with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete Multimodal Recommendations</a></strong></dt>
<dd>
<p>用模态感知扩散补全缺失模态，并以反事实推断剥离可见性偏差，在三个数据集上同时提升准确率与公平性。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/hcmrm_a_high_consistency_multimodal_relevance_model_for_search_ads/">HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads</a></strong></dt>
<dd>
<p>用视频关键词切分合成伪查询做三元组预训练，并以对称层次 Softmax 对齐四档相关性标签，快手搜索广告线上无关广告率降 6.1%、收入增 1.4%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/hierarchical_time_aware_mixture_of_experts_for_multi_modal_sequential_recom/">Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation</a></strong></dt>
<dd>
<p>用交互MoE过滤模态冗余、时序MoE注入显式时间戳，配合三项辅助任务，在四个Amazon数据集全库排序上NDCG@5最高提升20.9%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/image_fusion_for_cross_domain_sequential_recommendation/">Image Fusion for Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结 CLIP 图像嵌入补足 ID 表征，并以多注意力分别建模域内/跨域序列，在 Amazon 双域场景 MRR 提升约 5%–18%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/multimodal_semantic_retrieval_for_product_search/">Multimodal semantic retrieval for product search</a></strong></dt>
<dd>
<p>在电商产品检索中系统对比 4-tower/3-tower 图文融合双塔，用 FAISS 全库 Cosine 检索验证多模态表征在购买召回与相关性精度上的增益。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/personalized_image_generation_with_large_multimodal_models/">Personalized Image Generation with Large Multimodal Models</a></strong></dt>
<dd>
<p>Pigeon 用 LMM+掩码生成器从噪声历史图中抽取视觉偏好，两阶段偏好对齐（掩码重建+DPO）实现贴纸/海报个性化生成，个性化指标提升 7%~31%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/persrv_personalized_sticker_retrieval_with_vision_language_model/">PerSRV: Personalized Sticker Retrieval with Vision-Language Model</a></strong></dt>
<dd>
<p>用 SFT-LLaVA 生成贴纸语义、三路众包效用分与 CLIP 风格聚类，实现召回+个性化排序的贴纸检索，WeChat 数据 M-MRR@20 提升 8.95%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multimodal_rec/multimodal_enhanced_federated_recommendation_a_group_wise_fusion_approach/">Multimodal-enhanced Federated Recommendation: A Group-wise Fusion Approach</a></strong></dt>
<dd>
<p>将多模态表征学习卸载至服务器，以组感知聚合+偏好引导蒸馏把多模态语义注入联邦推荐，五数据集全库排序 HR@50 最高提升约 30%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multimodal_rec/personalized_parameter_efficient_fine_tuning_of_foundation_models_for_multi/">Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation</a></strong></dt>
<dd>
<p>按兴趣分群并为每群挂载独立 PEFT 模块，使 CLIP 编码出用户条件化的物品表征，配合群内难负采样，NDCG@20 最高提升 15.3%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/multimodal_rec/prism_personalized_recommendation_via_information_synergy_module/">PRISM: Personalized Recommendation via Information Synergy Module</a></strong></dt>
<dd>
<p>用 MoE 显式分解模态唯一/冗余/协同信息并以用户偏好动态重加权，即插即用提升三类序列骨干，Yelp N@10 超最强基线 10.91%。</p>
</dd>
</dl>

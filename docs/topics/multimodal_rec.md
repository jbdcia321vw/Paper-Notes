# 🧩 多模态推荐与内容理解 (Multimodal Recommendation)

> **研究范畴**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

当前方向共收录 **42** 篇顶会/顶刊论文解读：

<dl>
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
<dt><strong><a href="../../venues/IJCAI/2026/sequential_rec/rethinking_convolutional_networks_for_attribute_aware_sequential_recommenda/">Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation</a></strong></dt>
<dd>
<p>用金字塔式降尺度卷积替代自注意力聚合属性感知序列，线性复杂度下在四个 Amazon 数据集上 HR@10 最高提升 4.08%。</p>
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
<dt><strong><a href="../../venues/TKDE/2026/multimodal_rec/tackling_the_modality_inhibitance_effect_with_wavelet_based_modality_integr/">Tackling the Modality Inhibitance Effect with Wavelet-based Modality Integration in Multi-modal Recommendation</a></strong></dt>
<dd>
<p>提出模态抑制效应(MIE)概念，用Haar小波多分辨率分解提取各模态本质特征、结构感知多模态图学习与交叉加权自适应调节器，在4数据集Full-Ranking下平均提升6.48%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/multimodal_rec/training_free_graph_based_imputation_of_missing_modalities_in_multimodal_re/">Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation</a></strong></dt>
<dd>
<p>将缺失模态重铸为物品-物品共购图上的图特征插值，提出四种免训练传播式填补算子，可即插即用并保持甚至扩大多模态与传统推荐的性能差距。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/multimodal_rec/adopting_state_of_the_art_pretrained_audio_representations_for_music_recomm/">Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems</a></strong></dt>
<dd>
<p>系统评测 9 个 MIR 预训练音频表征在 5 类推荐器上的热/冷启动表现，揭示 MIR 性能无法直接迁移到推荐任务。</p>
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

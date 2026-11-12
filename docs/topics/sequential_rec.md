# ⏱️ 序列与会话推荐 (Sequential & Session-Based Recommendation)

> **研究范畴**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

当前方向共收录 **15** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在 Full-Ranking 留一法协议下 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 自适应学习用户级增强，在噪声序列推荐上 Full-Ranking 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/beyond_item_ids_scaling_short_form_video_recommendation_via_semantic_native/">Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling</a></strong></dt>
<dd>
<p>用深度截断 Bi-gram 语义ID替代正交Video ID，并以无参时间折叠压缩超长序列，在十亿用户短视频平台实现L=2000序列建模与线上显著增益。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong></dt>
<dd>
<p>用日历时间二值信号+共享多尺度时序CNN+ISAB集合编码做下一篮复购推荐，全库候选排序下离线与生产均显著提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/every_preference_has_its_strength_injecting_ordinal_semantics_into_llm_base/">Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders</a></strong></dt>
<dd>
<p>将显式评分建模为序数语义锚点，在LLM隐空间做强度感知对齐，保留细粒度偏好强度，成对偏好准确率最高提升44.9%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/fave_flow_based_average_velocity_establishment_for_sequential_recommendatio/">FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义锚先验替代高斯噪声并学习全局平均速度场，实现单步流匹配生成式序列推荐，全库排序下NDCG@20相对提升9.90%且推理加速20倍以上。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>提出目标条件化频域滤波双分支网络，用注意力谱熵观察引导频域去噪，三数据集GAUC/AUC全面超越SOTA</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结+可学习双网络的模块化集成模拟指数级多样性，推理仅用单网络实现零开销的集成级序列推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong></dt>
<dd>
<p>提出线性时间双向注意力 GBLA，以 Conv1D+Key Gating+Gated RMSNorm 三件套替换编码器自注意力，混合 1:2 堆叠在 Yandex Music 上持平双向 SA 质量，L=32768 单层提速 8.2×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/kuailive_a_real_time_interactive_dataset_for_live_streaming_recommendation/">KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation</a></strong></dt>
<dd>
<p>首个含直播间生命周期、多行为与负反馈的实时直播推荐数据集，构建Top-K与CTR基准。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/l2rec_towards_dual_view_understanding_of_llms_for_personalized_recommendati/">L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation</a></strong></dt>
<dd>
<p>在 LLM 参数空间用双视图个性化 MoE (LoRA) 分别适配行为/语义视图，ACF 融合，四数据集 N@10 提升 3.87%–8.02%，线上 CTR +9.24%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/learning_decomposed_contextual_token_representations_from_pretrained_and_co/">Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation</a></strong></dt>
<dd>
<p>DECOR 通过分解嵌入融合保留冻结预训练码本语义、上下文化 token 组合动态消歧，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 最高提升 14%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致SR模型评估失真，系统分析SSS生效条件与机理并给出公平评估建议</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/recnexteval_a_reference_implementation_for_temporal_next_batch_recommendati/">RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation</a></strong></dt>
<dd>
<p>提出滑动时间窗 next-batch 评估框架，强制全局时间线数据释放以消除数据穿越，支持增量学习与宏/微平均指标聚合。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/wpgrec_wavelet_packet_guided_graph_enhanced_sequential_recommendation/">WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用平稳小波包变换生成等长平移不变子带，逐子带做 Chebyshev 图传播并以能量-谱平坦度门控融合，全排序协议下稀疏数据集 HR@20 提升 15.4%</p>
</dd>
</dl>

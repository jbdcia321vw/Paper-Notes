# ⏱️ 序列与会话推荐

**SIGIR 2026** · **14** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

🔥 **高频主题：** 召回 ×11 · 精排 ×3

<dl>
<dt><strong><a href="beyond_item_ids_scaling_short_form_video_recommendation_via_semantic_native/">Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling</a></strong></dt>
<dd>
<p>用深度截断 Bi-gram 语义ID替代正交Video ID，并以无参时间折叠压缩超长序列，在十亿用户短视频平台实现L=2000序列建模与线上显著增益。</p>
</dd>
<dt><strong><a href="case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong></dt>
<dd>
<p>用日历时间二值信号+共享多尺度时序CNN+ISAB集合编码做下一篮复购推荐，全库候选排序下离线与生产均显著提升</p>
</dd>
<dt><strong><a href="cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong></dt>
<dd>
<p>提出可学习多序列构造+线性注意力，将用户历史解耦为K条主题潜序列以缓解上下文污染，Meta四场景离线NE与线上A/B均显著提升</p>
</dd>
<dt><strong><a href="divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong></dt>
<dd>
<p>借鉴最近发展区理论提出 Teacher-Peer-Self 多源负采样框架，以分歧重排提升多样性、共识蒸馏复用打分，六数据集五骨干稳定超越 SOTA。</p>
</dd>
<dt><strong><a href="flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结+可学习双网络的模块化集成模拟指数级多样性，推理仅用单网络实现零开销的集成级序列推荐。</p>
</dd>
<dt><strong><a href="fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习缓解长尾物品表征不足，在三个数据集多骨干上稳定提升。</p>
</dd>
<dt><strong><a href="learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱足门控+TTNP自监督学习兴趣生命周期，PPRR降超60%。</p>
</dd>
<dt><strong><a href="pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致SR模型评估失真，系统分析SSS生效条件与机理并给出公平评估建议</p>
</dd>
<dt><strong><a href="recpfn_prior_fitted_networks_for_in_context_based_recommendations/">RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations</a></strong></dt>
<dd>
<p>纯合成因果先验预训练的 PFN 式上下文学习序列推荐器，单次前向零样本适配新域，8 个基准上超越零样本基线并逼近低算力监督方法</p>
</dd>
<dt><strong><a href="rote_coarse_to_fine_multi_level_rotary_time_embedding_for_sequential_recomm/">RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation</a></strong></dt>
<dd>
<p>将时间戳分解为年/月/日多粒度并借 RoPE 旋转注入 Q/K，即插即用提升 SASRec/RPG，NDCG@5 最高 +20.11%</p>
</dd>
<dt><strong><a href="spectran_spectral_aware_transformer_based_adapter_for_llm_enhanced_sequenti/">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用谱域 Transformer 适配器替代静态 SVD 截断，全谱注意力聚合 LLM 语义嵌入，四数据集三骨干平均提升 9.17%</p>
</dd>
<dt><strong><a href="sprint_scalable_and_predictive_intent_refinement_for_llm_enhanced_session_b/">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></strong></dt>
<dd>
<p>用全局意图池约束LLM生成、以推荐效用验证意图，并蒸馏为轻量意图预测器，实现免LLM推理的会话推荐。</p>
</dd>
<dt><strong><a href="time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong></dt>
<dd>
<p>提出 TIDE，用 Hawkes 增强傅里叶时间编码 + 双专家解耦习惯/探索意图，在四个 Amazon NBR 数据集上显著超越 SOTA。</p>
</dd>
<dt><strong><a href="wpgrec_wavelet_packet_guided_graph_enhanced_sequential_recommendation/">WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用平稳小波包变换生成等长平移不变子带，逐子带做 Chebyshev 图传播并以能量-谱平坦度门控融合，全排序协议下稀疏数据集 HR@20 提升 15.4%</p>
</dd>
</dl>

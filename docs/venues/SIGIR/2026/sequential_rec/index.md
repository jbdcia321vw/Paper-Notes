# ⏱️ 序列与会话推荐

**SIGIR 2026** · **9** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [KDD 2026 (4)](../../../KDD/2026/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2026 (2)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** 召回 ×6 · 精排 ×3

<dl>
<dt><strong><a href="cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong></dt>
<dd>
<p>提出可学习多序列构造+线性注意力，将用户历史解耦为K条主题潜序列以缓解上下文污染，Meta四场景离线NE与线上A/B均显著提升</p>
</dd>
<dt><strong><a href="divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong></dt>
<dd>
<p>借鉴最近发展区理论提出 Teacher-Peer-Self 多源负采样框架，以分歧重排提升多样性、共识蒸馏复用打分，六数据集五骨干稳定超越 SOTA。</p>
</dd>
<dt><strong><a href="from_time_and_place_to_preference_llm_driven_geo_temporal_context_in_recomm/">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></strong></dt>
<dd>
<p>用 LLM 从时间戳+粗粒度位置生成地理-时间上下文嵌入，经轻量信息量测试与两种融合策略注入序列推荐，MovieLens 与生产数据均获增益。</p>
</dd>
<dt><strong><a href="fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习缓解长尾物品表征不足，在三个数据集多骨干上稳定提升。</p>
</dd>
<dt><strong><a href="learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱足门控+TTNP自监督学习兴趣生命周期，PPRR降超60%。</p>
</dd>
<dt><strong><a href="recpfn_prior_fitted_networks_for_in_context_based_recommendations/">RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations</a></strong></dt>
<dd>
<p>纯合成因果先验预训练的 PFN 式上下文学习序列推荐器，单次前向零样本适配新域，8 个基准上超越零样本基线并逼近低算力监督方法</p>
</dd>
<dt><strong><a href="rote_coarse_to_fine_multi_level_rotary_time_embedding_for_sequential_recomm/">RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation</a></strong></dt>
<dd>
<p>将时间戳分解为年/月/日多粒度并借 RoPE 旋转注入 Q/K，即插即用提升 SASRec/RPG，NDCG@5 最高 +20.11%</p>
</dd>
<dt><strong><a href="sprint_scalable_and_predictive_intent_refinement_for_llm_enhanced_session_b/">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></strong></dt>
<dd>
<p>用全局意图池约束LLM生成、以推荐效用验证意图，并蒸馏为轻量意图预测器，实现免LLM推理的会话推荐。</p>
</dd>
<dt><strong><a href="time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong></dt>
<dd>
<p>提出 TIDE，用 Hawkes 增强傅里叶时间编码 + 双专家解耦习惯/探索意图，在四个 Amazon NBR 数据集上显著超越 SOTA。</p>
</dd>
</dl>

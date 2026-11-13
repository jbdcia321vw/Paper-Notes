---
hide:
  - toc
---

# 📖 IEEE TKDE 2026 推荐系统论文解读

5 篇 IEEE TKDE 2026 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/memory_is_no_longer_a_bottleneck_memory_efficient_graph_filtering_for_scala/">Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering</a></strong></dt>
<dd>
<p>用逐用户 Krylov 子空间替代全量物品相似度图，实现免训练多项式图滤波，内存降 5.74×、推理加速 26× 且精度反超 SOTA。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl>
<dt><strong><a href="sequential_rec/an_adaptive_context_aware_and_stacked_attention_network_based_recommendatio/">An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users&#x27; Temporal Preference</a></strong></dt>
<dd>
<p>用上下文物品注意力+堆叠多头用户注意力捕捉用户时变偏好，Leave-One-Out+100负采样排序，三数据集HR/NDCG/MRR@10全面超越SOTA。</p>
</dd>
<dt><strong><a href="sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl>
<dt><strong><a href="multimodal_rec/tackling_the_modality_inhibitance_effect_with_wavelet_based_modality_integr/">Tackling the Modality Inhibitance Effect with Wavelet-based Modality Integration in Multi-modal Recommendation</a></strong></dt>
<dd>
<p>提出模态抑制效应(MIE)概念，用Haar小波多分辨率分解提取各模态本质特征、结构感知多模态图学习与交叉加权自适应调节器，在4数据集Full-Ranking下平均提升6.48%。</p>
</dd>
<dt><strong><a href="multimodal_rec/training_free_graph_based_imputation_of_missing_modalities_in_multimodal_re/">Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation</a></strong></dt>
<dd>
<p>将缺失模态重铸为物品-物品共购图上的图特征插值，提出四种免训练传播式填补算子，可即插即用并保持甚至扩大多模态与传统推荐的性能差距。</p>
</dd>
</dl>
</div>

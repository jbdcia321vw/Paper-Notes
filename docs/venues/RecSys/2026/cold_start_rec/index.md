# ❄️ 冷启动与长尾推荐

**RecSys 2026** · **3** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/) · [WWW (TheWebConf) 2026 (2)](../../../WWW/2026/cold_start_rec/)

🔥 **高频主题：** 召回 ×2

<dl>
<dt><strong><a href="idproxy_ctr_prediction_with_multimodal_llms_for_cold_start_recommendation_a/">IDProxy: CTR Prediction with Multimodal LLMs for Cold-Start Recommendation at Xiaohongshu</a></strong></dt>
<dd>
<p>用 MLLM 生成代理 ID 嵌入，经粗到细两阶段对齐工业 CTR 排序器，新笔记在线 AUC 提升约 2 倍于全局流量。</p>
</dd>
<dt><strong><a href="learning_sparse_representations_of_multimodal_content_for_enhanced_cold_ite/">Learning Sparse Representations of Multimodal Content for Enhanced Cold Item Recommendation</a></strong></dt>
<dd>
<p>将多模态内容冷启动表征改为 top-K 稀疏嵌入，并用 α-entmax 预稀疏激活诱导锐化与去噪，冷启动 NDCG@20 提升 16.6%–75.5% 且存储更低。</p>
</dd>
<dt><strong><a href="predicting_custom_feed_returns_for_new_bluesky_posts_a_prospective_study/">Predicting Custom-Feed Returns for New Bluesky Posts: A Prospective Study</a></strong></dt>
<dd>
<p>将新帖作为查询、信息流作为候选的冷启动路由任务，构建 collect-first/label-later 基准，LambdaRank 达 cR@10 0.7361。</p>
</dd>
</dl>

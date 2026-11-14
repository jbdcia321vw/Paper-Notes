# 🧩 多模态推荐与内容理解

**WWW (TheWebConf) 2025** · **7** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (3)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/)

🔥 **高频主题：** WWW 2025 ×7 · 多模态推荐 ×6 · 精排 ×3 · 召回 ×2

<dl>
<dt><strong><a href="comment_staytime_prediction_with_llm_enhanced_comment_understanding/">Comment Staytime Prediction with LLM-enhanced Comment Understanding</a></strong></dt>
<dd>
<p>首个评论区停留时长数据集 KuaiComt 与两阶段 LCU 框架：LLM 领域微调 + 评论排序辅助任务，离线全指标提升且线上停留时长 +1.27%。</p>
</dd>
<dt><strong><a href="generating_with_fairness_a_modality_diffused_counterfactual_framework_for_i/">Generating with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete Multimodal Recommendations</a></strong></dt>
<dd>
<p>用模态感知扩散补全缺失模态，并以反事实推断剥离可见性偏差，在三个数据集上同时提升准确率与公平性。</p>
</dd>
<dt><strong><a href="hcmrm_a_high_consistency_multimodal_relevance_model_for_search_ads/">HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads</a></strong></dt>
<dd>
<p>用视频关键词切分合成伪查询做三元组预训练，并以对称层次 Softmax 对齐四档相关性标签，快手搜索广告线上无关广告率降 6.1%、收入增 1.4%。</p>
</dd>
<dt><strong><a href="hierarchical_time_aware_mixture_of_experts_for_multi_modal_sequential_recom/">Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation</a></strong></dt>
<dd>
<p>用交互MoE过滤模态冗余、时序MoE注入显式时间戳，配合三项辅助任务，在四个Amazon数据集全库排序上NDCG@5最高提升20.9%。</p>
</dd>
<dt><strong><a href="multimodal_semantic_retrieval_for_product_search/">Multimodal semantic retrieval for product search</a></strong></dt>
<dd>
<p>在电商产品检索中系统对比 4-tower/3-tower 图文融合双塔，用 FAISS 全库 Cosine 检索验证多模态表征在购买召回与相关性精度上的增益。</p>
</dd>
<dt><strong><a href="personalized_image_generation_with_large_multimodal_models/">Personalized Image Generation with Large Multimodal Models</a></strong></dt>
<dd>
<p>Pigeon 用 LMM+掩码生成器从噪声历史图中抽取视觉偏好，两阶段偏好对齐（掩码重建+DPO）实现贴纸/海报个性化生成，个性化指标提升 7%~31%。</p>
</dd>
<dt><strong><a href="persrv_personalized_sticker_retrieval_with_vision_language_model/">PerSRV: Personalized Sticker Retrieval with Vision-Language Model</a></strong></dt>
<dd>
<p>用 SFT-LLaVA 生成贴纸语义、三路众包效用分与 CLIP 风格聚类，实现召回+个性化排序的贴纸检索，WeChat 数据 M-MRR@20 提升 8.95%。</p>
</dd>
</dl>

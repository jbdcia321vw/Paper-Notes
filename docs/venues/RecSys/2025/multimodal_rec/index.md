# 🧩 多模态推荐与内容理解

**RecSys 2025** · **9** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [RecSys 2026 (2)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×9 · 召回 ×6

<dl>
<dt><strong><a href="aps_explorer_navigating_algorithm_performance_spaces_for_informed_dataset_s/">APS Explorer: Navigating Algorithm Performance Spaces for Informed Dataset Selection</a></strong></dt>
<dd>
<p>将 29 算法 × 96 数据集的性能向量投影为可交互 PCA 性能空间，用可视化工具支撑可辩护的数据集选择。</p>
</dd>
<dt><strong><a href="beyond_the_past_leveraging_audio_and_human_memory_for_sequential_music_reco/">Beyond the past: Leveraging Audio and Human Memory for Sequential Music Recommendation</a></strong></dt>
<dd>
<p>用音频编码器预测新歌的 ACT-R 类激活值，将记忆式重复建模扩展到未听歌曲，NDCG_Exp 相对 PISA 提升约 40%，RepBias 降至 0.09%。</p>
</dd>
<dt><strong><a href="image_seeking_intent_prediction_for_cross_device_product_search/">Image-Seeking Intent Prediction for Cross-Device Product Search</a></strong></dt>
<dd>
<p>用语音查询+商品摘要预测图像意图，以可微精度损失抑制误触发，离线 Precision 达 78.5。</p>
</dd>
<dt><strong><a href="improving_visual_recommendation_on_e_commerce_platforms_using_vision_langua/">Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models</a></strong></dt>
<dd>
<p>在 Mercari C2C 平台微调 SigLIP 图像编码器做视觉相似召回，离线 nDCG@5 +9.1%，线上 CTR +50%、CVR +14%。</p>
</dd>
<dt><strong><a href="just_ask_for_music_jam_multimodal_and_personalized_natural_language_music_r/">Just Ask for Music (JAM): Multimodal and Personalized Natural Language Music Recommendation</a></strong></dt>
<dd>
<p>将查询建模为用户到物品的向量平移，用交叉注意力聚合音频/歌词/CF 多模态，在 JAMSessions 全库排序上 NDCG@10 相对 TalkRec 提升约 105%</p>
</dd>
<dt><strong><a href="llm_powered_nuanced_video_attribute_annotation_for_enhanced_recommendations/">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></strong></dt>
<dd>
<p>用 Gemini 多模态 LLM 做短视频“氛围”属性标注，经金/银集评估与蒸馏扩量，接入个性化受限召回，线上参与度 +0.49%。</p>
</dd>
<dt><strong><a href="personalized_image_generation_for_recommendations_beyond_catalogs/">Personalized Image Generation for Recommendations Beyond Catalogs</a></strong></dt>
<dd>
<p>用轻量条件扩散先验从隐式反馈采样用户个性化 CLIP 嵌入，冻结 SD+IP-Adapter 解码，免微调实现可扩展个性化图像生成。</p>
</dd>
<dt><strong><a href="vl_clip_enhancing_multimodal_recommendations_via_visual_grounding_and_llm_a/">VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings</a></strong></dt>
<dd>
<p>用 Grounding DINO 裁剪商品主体 + LLM 自反思改写商品描述，再微调 CLIP 双塔，Walmart 线上 CTR +18.6%、GMV +4.0%。</p>
</dd>
<dt><strong><a href="yambda_5b_a_large_scale_multi_modal_dataset_for_ranking_and_retrieval/">Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval</a></strong></dt>
<dd>
<p>发布 Yandex Music 4.79B 交互工业级数据集，含音频嵌入与 is_organic 标志，提出 Global Temporal Split 评测协议并给出 6 类基线基准。</p>
</dd>
</dl>

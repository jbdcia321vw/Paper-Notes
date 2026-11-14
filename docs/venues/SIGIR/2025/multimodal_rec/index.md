# 🧩 多模态推荐与内容理解

**SIGIR 2025** · **6** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [SIGIR 2026 (8)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×6 · 召回 ×4 · 精排 ×2

<dl>
<dt><strong><a href="cohesion_composite_graph_convolutional_network_with_dual_stage_fusion_for_m/">COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用行为模态早期精炼+晚期注意力融合的双阶段融合，配合异质/同质复合图与自适应BPR，在三个 Amazon 数据集上 R@20 相对提升 4.4%–6.1%。</p>
</dd>
<dt><strong><a href="diffusion_based_multi_modal_synergy_interest_network_for_click_through_rate/">Diffusion-based Multi-modal Synergy Interest Network for Click-through Rate Prediction</a></strong></dt>
<dd>
<p>用 PLE 式共享/独有专家解耦模态知识、以扩散式多步跨模态去噪捕捉协同信号，FDAF 非侵入融合，四数据集 AUC 全面领先。</p>
</dd>
<dt><strong><a href="disentangling_and_generating_modalities_for_recommendation_in_missing_modal/">Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios</a></strong></dt>
<dd>
<p>将模态特征解耦为通用/特有分量并用互信息损失分离，再以跨模态对齐特征与用户模态偏好生成缺失模态，缺失场景全库排序 R@20 相对提升 4.7%–8.2%。</p>
</dd>
<dt><strong><a href="multi_modal_multi_behavior_sequential_recommendation_with_conditional_diffu/">Multi-Modal Multi-Behavior Sequential Recommendation with Conditional Diffusion-Based Feature Denoising</a></strong></dt>
<dd>
<p>用条件扩散分别对模态与行为特征去噪，并以多专家网络解耦共性/特性兴趣，在 Rec-Tmal/Kuaishou 上 HR@10 提升约 4-8%。</p>
</dd>
<dt><strong><a href="short_video_segment_level_user_dynamic_interests_modeling_in_personalized_r/">Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation</a></strong></dt>
<dd>
<p>将短视频切分为5秒片段，用CLIP视觉+ID双模态跨注意力建模片段级动态兴趣，以段内排序损失解决标签缺失，在跳看预测与视频推荐两任务上大幅超越基线。</p>
</dd>
<dt><strong><a href="u_sticker_a_large_scale_multi_domain_user_sticker_dataset_for_retrieval_and/">U-Sticker: A Large-Scale Multi-Domain User Sticker Dataset for Retrieval and Personalization</a></strong></dt>
<dd>
<p>构建含 22K 用户、370K 贴纸、8.3M 消息的多领域多语言贴纸数据集，支撑个性化贴纸检索与用户建模。</p>
</dd>
</dl>

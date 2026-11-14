# 🧩 多模态推荐与内容理解

**CIKM 2025** · **9** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [CIKM 2026 (4)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×9 · 召回 ×6 · 精排 ×2

<dl>
<dt><strong><a href="a_content_driven_micro_video_recommendation_dataset_at_scale/">A Content-Driven Micro-Video Recommendation Dataset at Scale</a></strong></dt>
<dd>
<p>发布十亿级含原始视频模态的微视频推荐数据集 MicroLens，并系统基准 IDRec/VIDRec/VideoRec 三类范式，揭示端到端视频编码优于冻结特征的关键结论。</p>
</dd>
<dt><strong><a href="bilisting_modality_alignment_for_listings/">BiListing: Modality Alignment for Listings</a></strong></dt>
<dd>
<p>用 LLM 生成视觉画像 + PhotoSet Transformer 聚合多图，将 CLIP 图文对齐扩展到多图多文本的房源实体，线上 NDCG +0.425%。</p>
</dd>
<dt><strong><a href="distribution_guided_auto_encoder_for_user_multimodal_interest_cross_fusion/">Distribution-Guided Auto-Encoder for User Multimodal Interest Cross Fusion</a></strong></dt>
<dd>
<p>以行为粒度做模态内/模态间交叉融合，并用兴趣分布解码器作自监督约束，工业 A/B 上 CTR +12.9%。</p>
</dd>
<dt><strong><a href="do_recommender_systems_really_leverage_multimodal_content_a_comprehensive_a/">Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation</a></strong></dt>
<dd>
<p>用受控噪声实验证明多模态增益源于语义而非模型容量，并以 LVLM 结构化提示生成「原生多模态」[EOS] 嵌入替代晚期融合，Borda 聚合排名第一。</p>
</dd>
<dt><strong><a href="efficient_multimodal_streaming_recommendation_via_expandable_side_mixture_o/">Efficient Multimodal Streaming Recommendation via Expandable Side Mixture-of-Experts</a></strong></dt>
<dd>
<p>冻结 BERT/ViT 主干，用可扩展侧调 MoE 增量吸收新时段知识并按利用率剪枝，在三个流式数据集上 HR@10 全面超越 SOTA 且显存/训练时间更省。</p>
</dd>
<dt><strong><a href="hypercomplex_prompt_aware_multimodal_recommendation/">Hypercomplex Prompt-aware Multimodal Recommendation</a></strong></dt>
<dd>
<p>用 Cayley-Dickson 超复数多分量嵌入 + 超复数乘法非线性融合 + Prompt 动态补偿，在四个 Amazon 数据集 Full-Ranking 下 Recall@20 最高提升约 13%。</p>
</dd>
<dt><strong><a href="miss_multi_modal_tree_indexing_and_searching_with_lifelong_sequential_behav/">MISS: Multi-Modal Tree Indexing and Searching with Lifelong Sequential Behavior for Retrieval Recommendation</a></strong></dt>
<dd>
<p>用多模态嵌入构建树索引，并以 Co-GSU/MM-GSU 双视角检索终身行为，快手线上召回@800 相对提升 37.9%。</p>
</dd>
<dt><strong><a href="modality_alignment_with_multi_scale_bilateral_attention_for_multimodal_reco/">Modality Alignment with Multi-scale Bilateral Attention for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 MambaRec，用多尺度空洞卷积+双注意力做局部图文对齐、MMD+InfoNCE 做全局分布对齐，Amazon 三数据集 Recall@20 最高提升约 5.4%。</p>
</dd>
<dt><strong><a href="qarm_quantitative_alignment_multi_modal_recommendation_at_kuaishou/">QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou</a></strong></dt>
<dd>
<p>用业务 item-item 对微调 MLLM 做表示对齐，再以 VQ/RQ 量化成语义码实现端到端可学习，快手广告与电商线上 GMV+2.296%、冷启曝光+5.324%</p>
</dd>
</dl>

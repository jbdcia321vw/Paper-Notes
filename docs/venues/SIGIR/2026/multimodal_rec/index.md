# 🧩 多模态推荐与内容理解

**SIGIR 2026** · **8** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [KDD 2026 (1)](../../../KDD/2026/multimodal_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×8 · 召回 ×5

<dl>
<dt><strong><a href="a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 1.5B LLaMA2+BLIP-2 将图像转字幕并 token 化为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%，延迟中性。</p>
</dd>
<dt><strong><a href="from_raw_features_to_effective_embeddings_a_three_stage_approach_for_multim/">From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation</a></strong></dt>
<dd>
<p>用 VLM/LLM 摘要 + LightGCN 传播 + 跨视图对比学习三阶段增强多模态特征，全库排序下 Recall@10 提升 7-15%。</p>
</dd>
<dt><strong><a href="mllmrec_a_preference_reasoning_paradigm_with_graph_refinement_for_multimoda/">MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 将图像转语义描述并推理用户偏好画像，配合阈值去噪与拓扑增强精修物品图，三数据集全库排序 SOTA。</p>
</dd>
<dt><strong><a href="multi_modal_relational_item_representation_learning_for_inferring_substitut/">Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items</a></strong></dt>
<dd>
<p>MMSC 融合冻结多模态基座与元路径行为编码器，用对比去噪与 LLM 辅助监督推断替代/互补关系，五数据集提升 26.1%/39.2%。</p>
</dd>
<dt><strong><a href="robust_multimodal_recommendation_via_graph_retrieval_enhanced_modality_comp/">Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion</a></strong></dt>
<dd>
<p>提出 GRE-MC，用模态感知子图检索+图 Transformer 联合编码+稀疏路由码本补全缺失模态，40% 缺失率下全库排序显著超越 SOTA。</p>
</dd>
<dt><strong><a href="structural_and_disentangled_adaptation_of_large_vision_language_models_for_/">Structural and Disentangled Adaptation of Large Vision Language Models for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 SDA 轻量框架，用结构对齐软教师与模态解耦 LoRA 适配 LVLM，在 Amazon 三数据集上 H@10 平均提升 6.15%。</p>
</dd>
<dt><strong><a href="timemm_time_as_operator_spectral_filtering_for_dynamic_multimodal_recommend/">TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation</a></strong></dt>
<dd>
<p>将交互时间衰减映射为参数化图算子族，构建轻量谱滤波器组，结合自适应谱滤波与模态路由实现动态多模态推荐，全库排序下全面超越 SOTA。</p>
</dd>
<dt><strong><a href="well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用交互物品模态特征+聚类全局语义免训练初始化用户表征，弥合用户-物品语义鸿沟，六模型四数据集Recall@5最高提升36.4%。</p>
</dd>
</dl>

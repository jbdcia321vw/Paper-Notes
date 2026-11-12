# 🧩 多模态推荐与内容理解 (Multimodal Recommendation)

> **研究范畴**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

当前方向共收录 **13** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 LLaMA2-1.5B+BLIP-2 将图像转为 caption 并 tokenize 为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recomm/">CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation</a></strong></dt>
<dd>
<p>将文本/视觉/协同信号渲染为卡片式视觉语义单元，并用可学习可逆非均匀变换的 NU-RQ-VAE 量化，缓解码本失衡与生成偏置。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/discrete_preference_learning_for_personalized_multimodal_generation/">Discrete Preference Learning for Personalized Multimodal Generation</a></strong></dt>
<dd>
<p>用模态专属LightGCN+RQ-VAE将连续偏好量化为离散token注入LLM/扩散生成器，配合跨模态一致性奖励实现个性化图文生成。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/dual_diffusional_generative_fashion_recommendation/">Dual-Diffusional Generative Fashion Recommendation</a></strong></dt>
<dd>
<p>双扩散 Transformer 联合生成时尚单品图像与属性级文本，实现可解释个性化穿搭推荐，PFITB/GOR 双任务全面领先。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/from_raw_features_to_effective_embeddings_a_three_stage_approach_for_multim/">From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation</a></strong></dt>
<dd>
<p>用 VLM/LLM 摘要 + LightGCN 传播 + 跨视图对比学习三阶段增强多模态特征，全库排序下 Recall@10 提升 7-15%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/mllmrec_a_preference_reasoning_paradigm_with_graph_refinement_for_multimoda/">MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 将图像转语义描述并推理用户偏好画像，配合阈值去噪与拓扑增强精修物品图，三数据集全库排序 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/multi_modal_relational_item_representation_learning_for_inferring_substitut/">Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items</a></strong></dt>
<dd>
<p>MMSC 融合冻结多模态基座与元路径行为编码器，用对比去噪与 LLM 辅助监督推断替代/互补关系，五数据集提升 26.1%/39.2%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/multimodal_large_language_models_with_adaptive_preference_optimization_for_/">Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation</a></strong></dt>
<dd>
<p>面向 MLLM 序列推荐，提出难度感知重加权 (HaRS) 与噪声正则分布优化 (NoDO) 改进 DPO，在三个数据集上 NDCG@3 相对提升 7.39%–11.02%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/robust_multimodal_recommendation_via_graph_retrieval_enhanced_modality_comp/">Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion</a></strong></dt>
<dd>
<p>提出 GRE-MC，用模态感知子图检索+图 Transformer 联合编码+稀疏路由码本补全缺失模态，40% 缺失率下全库排序显著超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/rq_gmm_residual_quantized_gaussian_mixture_model_for_multimodal_semantic_di/">RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction</a></strong></dt>
<dd>
<p>用高斯混合模型替代硬聚类做残差量化，将多模态嵌入离散为语义ID，提升码本利用率与重建精度，线上广告主价值+1.502%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/structural_and_disentangled_adaptation_of_large_vision_language_models_for_/">Structural and Disentangled Adaptation of Large Vision Language Models for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 SDA 轻量框架，用结构对齐软教师与模态解耦 LoRA 适配 LVLM，在 Amazon 三数据集上 H@10 平均提升 6.15%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/timemm_time_as_operator_spectral_filtering_for_dynamic_multimodal_recommend/">TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation</a></strong></dt>
<dd>
<p>将交互时间衰减映射为参数化图算子族，构建轻量谱滤波器组，结合自适应谱滤波与模态路由实现动态多模态推荐，全库排序下全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出免训练、模型无关的 SG-URInit，用交互物品模态特征+聚类全局语义初始化用户表征，弥合用户-物品语义鸿沟，全库排序下 Recall@10 最高提升 18.25%。</p>
</dd>
</dl>

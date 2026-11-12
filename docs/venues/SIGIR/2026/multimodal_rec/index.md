# 🧩 多模态推荐与内容理解

**SIGIR 2026** · **3** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

🔥 **高频主题：** 多模态推荐 ×3

<dl>
<dt><strong><a href="a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 LLaMA2-1.5B+BLIP-2 将图像转为 caption 并 tokenize 为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%。</p>
</dd>
<dt><strong><a href="discrete_preference_learning_for_personalized_multimodal_generation/">Discrete Preference Learning for Personalized Multimodal Generation</a></strong></dt>
<dd>
<p>用模态专属LightGCN+RQ-VAE将连续偏好量化为离散token注入LLM/扩散生成器，配合跨模态一致性奖励实现个性化图文生成。</p>
</dd>
<dt><strong><a href="well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出免训练、模型无关的 SG-URInit，用交互物品模态特征+聚类全局语义初始化用户表征，弥合用户-物品语义鸿沟，全库排序下 Recall@10 最高提升 18.25%。</p>
</dd>
</dl>

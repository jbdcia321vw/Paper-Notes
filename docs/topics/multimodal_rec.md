# 🧩 多模态推荐与内容理解 (Multimodal Recommendation)

> **研究范畴**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

当前方向共收录 **4** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 LLaMA2-1.5B+BLIP-2 将图像转为 caption 并 tokenize 为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/discrete_preference_learning_for_personalized_multimodal_generation/">Discrete Preference Learning for Personalized Multimodal Generation</a></strong></dt>
<dd>
<p>用模态专属LightGCN+RQ-VAE将连续偏好量化为离散token注入LLM/扩散生成器，配合跨模态一致性奖励实现个性化图文生成。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/multimodal_large_language_models_with_adaptive_preference_optimization_for_/">Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation</a></strong></dt>
<dd>
<p>面向 MLLM 序列推荐，提出难度感知重加权 (HaRS) 与噪声正则分布优化 (NoDO) 改进 DPO，在三个数据集上 NDCG@3 相对提升 7.39%–11.02%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出免训练、模型无关的 SG-URInit，用交互物品模态特征+聚类全局语义初始化用户表征，弥合用户-物品语义鸿沟，全库排序下 Recall@10 最高提升 18.25%。</p>
</dd>
</dl>

# ❄️ 冷启动与长尾推荐 (Cold-Start & Long-Tail Recommendation)

> **研究范畴**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

当前方向共收录 **5** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong></dt>
<dd>
<p>针对本地社区平台地理曝光约束下的“不可能负例”问题，提出区域约束批采样 RCBS，仅改批构造不改模型与损失，离线与线上 A/B 均显著提升召回与排序。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/filling_the_gaps_selective_knowledge_augmentation_for_llm_recommenders/">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></strong></dt>
<dd>
<p>提出 KnowSA_CKP，用比较式知识探测量化 LLM 对物品的协同知识缺口，仅对知识贫乏物品做选择性增强，免微调提升 Recall@1 与上下文效率。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/itemrag_item_based_retrieval_augmented_generation_for_llm_based_recommendat/">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 RAG 从粗粒度用户历史检索转向细粒度物品级检索，融合共购与语义信息为每个物品生成摘要，标准与冷启场景全面超越用户级 RAG 基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cold_start_rec/sparse_contrastive_learning_for_content_based_cold_item_recommendation/">Sparse Contrastive Learning for Content-Based Cold Item Recommendation</a></strong></dt>
<dd>
<p>用 α-entmax 稀疏化 sampled softmax 训练纯内容编码器，以物品-物品相似度建模冷启动，并支持内容空间知识蒸馏，冷物品排序与曝光公平性双赢。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出免训练、模型无关的 SG-URInit，用交互物品模态特征+聚类全局语义初始化用户表征，弥合用户-物品语义鸿沟，全库排序下 Recall@10 最高提升 18.25%。</p>
</dd>
</dl>

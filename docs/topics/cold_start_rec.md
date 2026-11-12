# ❄️ 冷启动与长尾推荐 (Cold-Start & Long-Tail Recommendation)

> **研究范畴**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

当前方向共收录 **9** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/aspect_aware_content_based_recommendations_for_mathematical_research_papers/">Aspect-Aware Content-Based Recommendations for Mathematical Research Papers</a></strong></dt>
<dd>
<p>面向数学论文的 aspect 条件异构图 GNN 推荐，联合文本语义、引用结构与作者谱系，在 GoldRiM/SilverRiM 上全面超越 LLM 嵌入基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong></dt>
<dd>
<p>针对地理约束曝光下的“不可能负样本”问题，提出区域约束批采样 RCBS，仅改批构造即显著提升用户表征与下游排序/召回/广告指标。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/filling_the_gaps_selective_knowledge_augmentation_for_llm_recommenders/">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></strong></dt>
<dd>
<p>提出 KnowSA_CKP，用比较式知识探测量化 LLM 对物品的协同知识缺口，仅对知识贫乏物品做选择性增强，免微调提升 Recall@1 与上下文效率。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cold_start_rec/genrecedit_adapting_model_editing_for_generative_recommendation_with_cold_s/">GenRecEdit: Adapting Model Editing for Generative Recommendation with Cold-Start Items</a></strong></dt>
<dd>
<p>将NLP模型编辑范式迁移至生成式推荐，通过位置级迭代编辑与One-One触发机制注入冷启动物品SID模式，仅需9.5%重训时间即显著提升冷启推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/itemrag_item_based_retrieval_augmented_generation_for_llm_based_recommendat/">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 RAG 从粗粒度用户历史检索转向细粒度物品级检索，融合共购与语义信息为每个物品生成摘要，标准与冷启场景全面超越用户级 RAG 基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/poi_spatiotemporal_rec/rest_a_plug_and_play_spatially_constrained_representation_enhancement_frame/">ReST: A Plug-and-Play Spatially-Constrained Representation Enhancement Framework for Local-Life Recommendation</a></strong></dt>
<dd>
<p>以物品为中心，用元ID预热+空间约束难负采样对比学习增强长尾物品表征，离线AUC提升且线上GMV+2.804%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cold_start_rec/sparse_contrastive_learning_for_content_based_cold_item_recommendation/">Sparse Contrastive Learning for Content-Based Cold Item Recommendation</a></strong></dt>
<dd>
<p>用 α-entmax 稀疏化 sampled softmax 训练纯内容编码器，以物品-物品相似度建模冷启动，并支持内容空间知识蒸馏，冷物品排序与曝光公平性双赢。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出免训练、模型无关的 SG-URInit，用交互物品模态特征+聚类全局语义初始化用户表征，弥合用户-物品语义鸿沟，全库排序下 Recall@10 最高提升 18.25%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong></dt>
<dd>
<p>零权重免训练新闻推荐框架，用时间衰减×表征相似度打分，离线AUC超越NRMS且推理快600倍，线上CTR接近NRMS</p>
</dd>
</dl>

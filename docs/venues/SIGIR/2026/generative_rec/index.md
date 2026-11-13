# ✨ 生成式推荐与离散语义 ID

**SIGIR 2026** · **12** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [CIKM 2026 (7)](../../../CIKM/2026/generative_rec/) · [KDD 2026 (8)](../../../KDD/2026/generative_rec/) · [WSDM 2026 (2)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐与离散语义ID ×9 · 端到端生成式 ×6 · 召回 ×2 · 生成式端到端 ×2

<dl>
<dt><strong><a href="a_cascaded_generative_approach_for_e_commerce_recommendations/">A Cascaded Generative Approach for e-Commerce Recommendations</a></strong></dt>
<dd>
<p>级联 LLM 生成货架主题与 RAG 约束关键词驱动检索，教师-学生蒸馏落地，线上加购 +2.7%</p>
</dd>
<dt><strong><a href="bi_level_optimization_for_generative_recommendation_bridging_tokenization_a/">Bi-Level Optimization for Generative Recommendation: Bridging Tokenization and Generation</a></strong></dt>
<dd>
<p>将 tokenizer 与 recommender 重构为双层优化问题，用元学习+梯度手术对齐语义ID与推荐目标，全库排序下稳定超越 SOTA。</p>
</dd>
<dt><strong><a href="card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recomm/">CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation</a></strong></dt>
<dd>
<p>将文本/视觉/协同信号渲染为卡片式视觉语义单元，并用可学习可逆非均匀变换的 NU-RQ-VAE 量化，缓解码本失衡与生成偏置。</p>
</dd>
<dt><strong><a href="differentiable_semantic_id_for_generative_recommendation/">Differentiable Semantic ID for Generative Recommendation</a></strong></dt>
<dd>
<p>用 Gumbel 噪声+不确定性衰减实现可微语义ID联合训练，缓解码本坍缩，Full-Ranking 下 NDCG@10 稳定提升</p>
</dd>
<dt><strong><a href="dual_diffusional_generative_fashion_recommendation/">Dual-Diffusional Generative Fashion Recommendation</a></strong></dt>
<dd>
<p>双扩散 Transformer 联合生成时尚单品图像与属性级文本，实现可解释个性化穿搭推荐，PFITB/GOR 双任务全面领先。</p>
</dd>
<dt><strong><a href="gflowgr_fine_tuning_generative_recommendation_frameworks_with_generative_fl/">GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks</a></strong></dt>
<dd>
<p>将GR重构为多步集合生成任务，用GFlowNets做token级微调对齐物品效用，三数据集双骨干一致提升并落地淘宝搜索广告。</p>
</dd>
<dt><strong><a href="mitigating_collaborative_semantic_id_staleness_in_generative_retrieval/">Mitigating Collaborative Semantic ID Staleness in Generative Retrieval</a></strong></dt>
<dd>
<p>提出模型无关的 SID 对齐更新，将新日志重建的协同语义 ID 映射回旧词表以支持热启动微调，严格时序评估下 Recall@500 提升且训练算力降低约 8-9 倍。</p>
</dd>
<dt><strong><a href="mviger_multi_view_variational_integration_of_complementary_knowledge_for_ge/">MVIGER: Multi-View Variational Integration of Complementary Knowledge for Generative Recommender</a></strong></dt>
<dd>
<p>将异构模板-索引组合视为多视图，用类别隐变量变分整合互补知识，三数据集全库排序显著超越生成式基线。</p>
</dd>
<dt><strong><a href="next_interest_flow_a_generative_pre_training_paradigm_for_recommender_syste/">Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines</a></strong></dt>
<dd>
<p>将用户意图建模为隐流形上的连续演化轨迹，用切空间正交与测地平滑约束生成式预训练，并以双向对齐+TSP桥接判别式精排，淘宝线上CTCVR+11.6%。</p>
</dd>
<dt><strong><a href="rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_gener/">RCLRec: Reverse Curriculum Learning for Modeling Sparse Conversions in Generative Recommendation</a></strong></dt>
<dd>
<p>逆向课程前缀+课程质量感知损失为稀疏转化目标注入实例级中间监督，离线全库排序显著提升并线上A/B +2.09%广告收入。</p>
</dd>
<dt><strong><a href="request_only_optimization_for_recommendation_systems/">Request-Only Optimization for Recommendation Systems</a></strong></dt>
<dd>
<p>以用户请求为训练样本单元，通过数据/系统/模型协同设计消除请求内特征冗余，训练吞吐提升最高570%，同等算力下模型FLOPs扩展7倍。</p>
</dd>
<dt><strong><a href="universal_item_tokenization_for_transferable_generative_recommendation/">Universal Item Tokenization for Transferable Generative Recommendation</a></strong></dt>
<dd>
<p>基于MLLM与树状码本的通用物品分词器，融合多模态重建与协同对齐，实现跨域可迁移生成式推荐。</p>
</dd>
</dl>

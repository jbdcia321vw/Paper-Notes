# ✨ 生成式推荐与离散语义 ID

**SIGIR 2026** · **17** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

🔥 **高频主题：** 生成式推荐与离散语义ID ×12 · 端到端生成式 ×8 · 召回 ×3 · 生成式端到端 ×3 · RQ-VAE 语义ID ×2 · 码本坍缩缓解 ×2

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
<dt><strong><a href="fave_flow_based_average_velocity_establishment_for_sequential_recommendatio/">FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义锚先验替代高斯噪声并学习全局平均速度场，实现单步流匹配生成式序列推荐，全库排序下NDCG@20相对提升9.90%且推理加速20倍以上。</p>
</dd>
<dt><strong><a href="gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong></dt>
<dd>
<p>提出线性时间双向注意力 GBLA，以 Conv1D+Key Gating+Gated RMSNorm 三件套替换编码器自注意力，混合 1:2 堆叠在 Yandex Music 上持平双向 SA 质量，L=32768 单层提速 8.2×。</p>
</dd>
<dt><strong><a href="genrec_a_preference_oriented_generative_framework_for_large_scale_recommend/">GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出 Page-wise NTP + 线性 Token Merger + GRPO-SR 偏好对齐的工业级生成式召回框架，京东线上点击 +9.5%、成交 +8.7%。</p>
</dd>
<dt><strong><a href="gflowgr_fine_tuning_generative_recommendation_frameworks_with_generative_fl/">GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks</a></strong></dt>
<dd>
<p>将GR重构为多步集合生成任务，用GFlowNets做token级微调对齐物品效用，三数据集双骨干一致提升并落地淘宝搜索广告。</p>
</dd>
<dt><strong><a href="learning_decomposed_contextual_token_representations_from_pretrained_and_co/">Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation</a></strong></dt>
<dd>
<p>DECOR 通过分解嵌入融合保留冻结预训练码本语义、上下文化 token 组合动态消歧，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 最高提升 14%。</p>
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
<dt><strong><a href="semantic_ids_for_recommender_systems_at_snapchat_use_cases_technical_challe/">Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices</a></strong></dt>
<dd>
<p>Snapchat 工业级 SID 落地实践：STE 全码本反传与多模态融合缓解码本坍缩，启发式桶内消歧解决 SID-Item 碰撞，离线与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="sigma_a_semantic_grounded_instruction_driven_generative_multi_task_recommen/">SIGMA: A Semantic-Grounded Instruction-Driven Generative Multi-Task Recommender at AliExpress</a></strong></dt>
<dd>
<p>多视图语义对齐+混合 SID/ID 分词+指令驱动多任务 SFT+三步生成与自适应概率融合，AliExpress 线上 GMV +7.84%</p>
</dd>
<dt><strong><a href="universal_item_tokenization_for_transferable_generative_recommendation/">Universal Item Tokenization for Transferable Generative Recommendation</a></strong></dt>
<dd>
<p>基于MLLM与树状码本的通用物品分词器，融合多模态重建与协同对齐，实现跨域可迁移生成式推荐。</p>
</dd>
</dl>

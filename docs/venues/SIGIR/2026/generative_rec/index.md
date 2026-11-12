# ✨ 生成式推荐与离散语义 ID

**SIGIR 2026** · **6** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

🔥 **高频主题：** 生成式推荐与离散语义ID ×3 · 端到端生成式 ×2

<dl>
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
<dt><strong><a href="learning_decomposed_contextual_token_representations_from_pretrained_and_co/">Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation</a></strong></dt>
<dd>
<p>DECOR 通过分解嵌入融合保留冻结预训练码本语义、上下文化 token 组合动态消歧，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 最高提升 14%。</p>
</dd>
<dt><strong><a href="semantic_ids_for_recommender_systems_at_snapchat_use_cases_technical_challe/">Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices</a></strong></dt>
<dd>
<p>Snapchat 工业级 SID 落地实践：STE 全码本反传与多模态融合缓解码本坍缩，启发式桶内消歧解决 SID-Item 碰撞，离线与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="sigma_a_semantic_grounded_instruction_driven_generative_multi_task_recommen/">SIGMA: A Semantic-Grounded Instruction-Driven Generative Multi-Task Recommender at AliExpress</a></strong></dt>
<dd>
<p>多视图语义对齐+混合 SID/ID 分词+指令驱动多任务 SFT+三步生成与自适应概率融合，AliExpress 线上 GMV +7.84%</p>
</dd>
</dl>

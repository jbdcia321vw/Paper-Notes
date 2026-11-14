# ✨ 生成式推荐与离散语义 ID

**RecSys 2025** · **6** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [RecSys 2026 (12)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐与离散语义ID ×6 · 生成式端到端 ×3 · RQ-VAE ×2 · Semantic ID ×2 · 召回 ×2

<dl>
<dt><strong><a href="enhancing_embedding_representation_stability_in_recommendation_systems_with/">Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID</a></strong></dt>
<dd>
<p>提出 Semantic ID prefix-ngram 分层前缀参数化，用语义聚类碰撞替代随机哈希，缓解 ID 漂移与长尾稀疏，Meta 线上 NE 提升 0.15% 且 A/A 方差降 43%</p>
</dd>
<dt><strong><a href="grace_generative_recommendation_via_journey_aware_sparse_attention_on_chain/">GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization</a></strong></dt>
<dd>
<p>用 PKG 属性链式 CoT token 增强语义 ID，并以旅程感知四路稀疏注意力替代全注意力，Home 域 HR@10 相对提升 106.9%，长序列注意力计算降 48%。</p>
</dd>
<dt><strong><a href="how_fair_is_your_diffusion_recommender_model/">How Fair is Your Diffusion Recommender Model?</a></strong></dt>
<dd>
<p>首个针对 DiffRec/L-DiffRec 的公平性实证研究，在 ML1M/FTKY 上以 Full-Ranking 评估效用与消费者/提供者公平，揭示扩散模型放大偏见而 L-DiffRec 可缓解。</p>
</dd>
<dt><strong><a href="prompt_to_slate_diffusion_models_for_prompt_conditioned_slate_generation/">Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation</a></strong></dt>
<dd>
<p>用潜空间扩散 Transformer 从纯文本 Prompt 直接生成整组 Slate，离线 NDCG 提升 17%、线上重复曝光降 13.4%。</p>
</dd>
<dt><strong><a href="semantic_ids_for_joint_generative_search_and_recommendation/">Semantic IDs for Joint Generative Search and Recommendation</a></strong></dt>
<dd>
<p>系统对比任务专属与跨任务语义ID构建策略，发现多任务双塔嵌入+统一RQ-KMeans语义ID在联合搜索推荐中取得最佳权衡。</p>
</dd>
<dt><strong><a href="semantic_ids_for_music_recommendation/">Semantic IDs for Music Recommendation</a></strong></dt>
<dd>
<p>用 RQ-VAE 残差量化生成音乐语义 ID 替代逐曲嵌入，参数量降 75%–99% 而精度不降，线上多样性显著提升。</p>
</dd>
</dl>

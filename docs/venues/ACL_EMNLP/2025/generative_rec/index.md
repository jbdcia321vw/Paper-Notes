# ✨ 生成式推荐与离散语义 ID

**ACL / EMNLP (RecSys & CRS) 2025** · **5** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐与离散语义ID ×4 · 生成式端到端 ×4 · EMNLP 2025 Findings ×2 · 零样本跨域推荐 ×2

<dl>
<dt><strong><a href="a_generative_framework_for_personalized_sticker_retrieval/">A Generative Framework for Personalized Sticker Retrieval</a></strong></dt>
<dd>
<p>PEARL 用用户组嵌入+LLM意图排序加权解码，实现微信表情包个性化生成式检索，离线 MRR@10 提升 15%、线上 CTR +7.12%。</p>
</dd>
<dt><strong><a href="enhancing_time_awareness_in_generative_recommendation/">Enhancing Time Awareness in Generative Recommendation</a></strong></dt>
<dd>
<p>将绝对时间戳与目标相对间隔verbalize进提示，并叠加全局转移图与免训练趋势重排，Full-Ranking下R@5最高提升15.4%。</p>
</dd>
<dt><strong><a href="gram_generative_recommendation_via_semantic_aware_multi_granular_late_fusio/">GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion</a></strong></dt>
<dd>
<p>将层次/协同语义翻译为 LLM 词表内文本 ID，并用多粒度提示+后期融合保留长文本信息，Full-Ranking 下 R@5 提升 11.5–16.0%。</p>
</dd>
<dt><strong><a href="recbase_generative_foundation_model_pretraining_for_zero_shot_recommendatio/">RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation</a></strong></dt>
<dd>
<p>用课程学习增强的 RQ-VAE 将跨域物品统一离散为层级概念 ID，从零预训练 1.5B 自回归推荐基座，在 8 个未见数据集零样本 AUC 全面超越 7B LLM。</p>
</dd>
<dt><strong><a href="recgpt_a_foundation_model_for_sequential_recommendation/">RecGPT: A Foundation Model for Sequential Recommendation</a></strong></dt>
<dd>
<p>纯文本驱动+FSQ统一分词+混合双向因果注意力的序列推荐基础模型，零样本跨域显著超越10%数据微调的ID基线。</p>
</dd>
</dl>

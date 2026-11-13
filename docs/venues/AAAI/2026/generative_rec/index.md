# ✨ 生成式推荐与离散语义 ID

**AAAI (RecSys Track) 2026** · **5** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** AAAI 2026 ×5 · 生成式推荐与离散语义ID ×3 · 生成式端到端 ×3 · 召回 ×2

<dl>
<dt><strong><a href="behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_b/">Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary</a></strong></dt>
<dd>
<p>用VQ-VAE将用户/物品行为离散化为宏观兴趣+微观意图的行为词表，经多级文本监督与语义对齐正则注入冻结LLM，零样本解释推荐SOTA</p>
</dd>
<dt><strong><a href="inductive_generative_recommendation_via_retrieval_based_speculation/">Inductive Generative Recommendation via Retrieval-based Speculation</a></strong></dt>
<dd>
<p>将投机解码的草稿-验证范式迁移至生成式推荐，用归纳式检索器起草候选、GR 模型做目标感知验证，使语义 ID 生成模型具备新物品推荐能力。</p>
</dd>
<dt><strong><a href="modeling_item_level_dynamic_variability_with_residual_diffusion_for_bundle_/">Modeling Item-Level Dynamic Variability with Residual Diffusion for Bundle Recommendation</a></strong></dt>
<dd>
<p>用残差扩散在潜空间重建捆绑包item级表征，模型无关地适配B-I归属动态变化，Recall/NDCG最高提升约23%，训练开销仅增4%。</p>
</dd>
<dt><strong><a href="multi_aspect_cross_modal_quantization_for_generative_recommendation/">Multi-Aspect Cross-modal Quantization for Generative Recommendation</a></strong></dt>
<dd>
<p>在 RQ-VAE 量化各层注入跨模态对比与重建对齐，并在 GR 训练中加隐式/显式对齐，Full-Ranking 下 HR@10 最高提升约 9.4%。</p>
</dd>
<dt><strong><a href="tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi_domain/">Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation</a></strong></dt>
<dd>
<p>提出 TokenMoE+码本+互信息校准的统一物品分词框架 UniTok，单模型跨 10 域训练，NDCG@10 最高提升 51.89%，参数量降 9.63×。</p>
</dd>
</dl>

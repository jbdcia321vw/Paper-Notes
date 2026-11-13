# ✨ 生成式推荐与离散语义 ID

**CIKM 2026** · **7** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [KDD 2026 (8)](../../../KDD/2026/generative_rec/) · [RecSys 2026 (8)](../../../RecSys/2026/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [WSDM 2026 (2)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐与离散语义ID ×4 · 生成式推荐 ×3 · 生成式端到端 ×3 · 精排 ×2

<dl>
<dt><strong><a href="cross_country_code_mixing_for_generative_recommendation/">Cross-Country Code-Mixing for Generative Recommendation</a></strong></dt>
<dd>
<p>借鉴多语言 code-switching，用双约束 token 级跨国混码合成混合序列并做上下文自适应重加权，在无重叠 ID 的跨国场景显著提升小国召回。</p>
</dd>
<dt><strong><a href="faithful_evaluation_of_semantic_id_tokenizers_for_generative_recommendation/">Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation</a></strong></dt>
<dd>
<p>揭示SID碰撞导致SID级指标虚高至103.36%，提出碰撞校正评估CCE与零碰撞重分配ZCR，翻转多个Tokenizer排序。</p>
</dd>
<dt><strong><a href="latent_residual_reasoning_for_ranking_in_generative_recommendation/">Latent Residual Reasoning for Ranking in Generative Recommendation</a></strong></dt>
<dd>
<p>在冻结的生成式排序骨干上叠加 LoRA 与残差推理头，用多步隐空间加性修正精化候选打分，公开集 GAUC 提升 2.7%–4.3%，线上 PV-CTR +0.75%。</p>
</dd>
<dt><strong><a href="learning_from_unreachable_rewards_hint_conditioned_reinforcement_learning_f/">Learning from Unreachable Rewards: Hint-Conditioned Reinforcement Learning for Generative Recommendation</a></strong></dt>
<dd>
<p>用 SFT 检查点诊断语义 ID 前缀树可达性，对不可达样本注入最短目标前缀提示并以提示感知信用分解（前缀 SFT + 后缀 GRPO）恢复学习信号，零优势组从 70%+ 降至 20% 以下。</p>
</dd>
<dt><strong><a href="palate_pretrained_autoregressive_latent_embeddings/">PALATE: Pretrained Autoregressive LATent Embeddings</a></strong></dt>
<dd>
<p>用GPT式自回归模型对消费者门店互动序列建模，导出末词元隐状态为冻结256维用户嵌入，供DCNv2广告排序器跨场景复用，离线AUC+0.59%/+0.57%，线上收入+0.79%/+0.94%。</p>
</dd>
<dt><strong><a href="rethinking_item_tokenization_in_generative_recommenders_from_fixed_atoms_to/">Rethinking Item Tokenization in Generative Recommenders: From Fixed Atoms to Semantic Subwords</a></strong></dt>
<dd>
<p>指出固定长度SID复用于历史侧引发“物品内注意力过载”，提出历史侧语义子词重分词(IST)+行为共现增强(BCA)的非对称分词，三数据集三骨干全库排序NDCG@10最高+13.8%。</p>
</dd>
<dt><strong><a href="sidinspector_a_mapping_first_diagnostic_resource_for_semantic_id_tokenizers/">SIDInspector: A Mapping-First Diagnostic Resource for Semantic-ID Tokenizers</a></strong></dt>
<dd>
<p>提出映射优先的 SID 分词器诊断资源，用 D1–D5 探针在生成器训练前分离容量、可寻址性与行为前缀对齐，并给出四类分词器工件画像。</p>
</dd>
</dl>

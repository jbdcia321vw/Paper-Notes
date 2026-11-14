# ✨ 生成式推荐与离散语义 ID

**CIKM 2026** · **14** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [CIKM 2025 (6)](../../2025/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐 ×8 · 生成式推荐与离散语义ID ×5 · 召回 ×4 · 生成式端到端 ×4 · 端到端生成式 ×3 · 精排 ×3

<dl>
<dt><strong><a href="cross_country_code_mixing_for_generative_recommendation/">Cross-Country Code-Mixing for Generative Recommendation</a></strong></dt>
<dd>
<p>借鉴多语言 code-switching，用双约束 token 级跨国混码合成混合序列并做上下文自适应重加权，在无重叠 ID 的跨国场景显著提升小国召回。</p>
</dd>
<dt><strong><a href="decoupled_temporal_encoding_for_generative_recommendation/">Decoupled Temporal Encoding for Generative Recommendation</a></strong></dt>
<dd>
<p>将宏观时间动态与微观序列顺序解耦的双模块位置编码，工业数据集 GAUC +1.16pt，线上 CTR +1.8%、RPM +3.0%，延迟仅 +0.3%。</p>
</dd>
<dt><strong><a href="faithful_evaluation_of_semantic_id_tokenizers_for_generative_recommendation/">Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation</a></strong></dt>
<dd>
<p>揭示SID碰撞导致SID级指标虚高至103.36%，提出碰撞校正评估CCE与零碰撞重分配ZCR，翻转多个Tokenizer排序。</p>
</dd>
<dt><strong><a href="intsr_an_integrated_generative_framework_for_search_and_recommendation/">IntSR: An Integrated Generative Framework for Search and Recommendation</a></strong></dt>
<dd>
<p>以 query 模态统一搜索/推荐与召回/排序四任务，提出 query 驱动解码器与时间对齐负采样，Amap 线上 GMV+9.34%。</p>
</dd>
<dt><strong><a href="latent_residual_reasoning_for_ranking_in_generative_recommendation/">Latent Residual Reasoning for Ranking in Generative Recommendation</a></strong></dt>
<dd>
<p>在冻结的生成式排序骨干上叠加 LoRA 与残差推理头，用多步隐空间加性修正精化候选打分，公开集 GAUC 提升 2.7%–4.3%，线上 PV-CTR +0.75%。</p>
</dd>
<dt><strong><a href="learning_from_unreachable_rewards_hint_conditioned_reinforcement_learning_f/">Learning from Unreachable Rewards: Hint-Conditioned Reinforcement Learning for Generative Recommendation</a></strong></dt>
<dd>
<p>用 SFT 检查点诊断语义 ID 前缀树可达性，对不可达样本注入最短目标前缀提示并以提示感知信用分解（前缀 SFT + 后缀 GRPO）恢复学习信号，零优势组从 70%+ 降至 20% 以下。</p>
</dd>
<dt><strong><a href="merge_next_generation_item_indexing_paradigm_for_large_scale_streaming_reco/">MERGE: Next-Generation Item Indexing Paradigm for Large-Scale Streaming Recommendation</a></strong></dt>
<dd>
<p>以并查集动态生成聚类替代 VQ 盲分配，配合占用监控与由细到粗合并构建分层码本，工业 A/B 显著提升曝光多样性与互动。</p>
</dd>
<dt><strong><a href="palate_pretrained_autoregressive_latent_embeddings/">PALATE: Pretrained Autoregressive LATent Embeddings</a></strong></dt>
<dd>
<p>用GPT式自回归模型对消费者门店互动序列建模，导出末词元隐状态为冻结256维用户嵌入，供DCNv2广告排序器跨场景复用，离线AUC+0.59%/+0.57%，线上收入+0.79%/+0.94%。</p>
</dd>
<dt><strong><a href="rethinking_item_tokenization_in_generative_recommenders_from_fixed_atoms_to/">Rethinking Item Tokenization in Generative Recommenders: From Fixed Atoms to Semantic Subwords</a></strong></dt>
<dd>
<p>指出固定长度SID复用于历史侧引发“物品内注意力过载”，提出历史侧语义子词重分词(IST)+行为共现增强(BCA)的非对称分词，三数据集三骨干全库排序NDCG@10最高+13.8%。</p>
</dd>
<dt><strong><a href="revisiting_generative_vs_dense_retrieval_a_static_value_prior_reverses_the_/">Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap</a></strong></dt>
<dd>
<p>在受控 GR/DR 对比下揭示指标选择反转胜负，用免训练静态价值先验重排 DR 即反超 GR 最佳物品价值并保持召回领先。</p>
</dd>
<dt><strong><a href="self_evolving_memory_for_generative_recommendation/">Self-Evolving Memory for Generative Recommendation</a></strong></dt>
<dd>
<p>揭示生成式推荐持续演化中的“演化冲突”问题，提出稀疏 KV 记忆层 LION，以隔离记忆+巩固损失实现异构行为模式的持续演化。</p>
</dd>
<dt><strong><a href="sidinspector_a_mapping_first_diagnostic_resource_for_semantic_id_tokenizers/">SIDInspector: A Mapping-First Diagnostic Resource for Semantic-ID Tokenizers</a></strong></dt>
<dd>
<p>提出映射优先的 SID 分词器诊断资源，用 D1–D5 探针在生成器训练前分离容量、可寻址性与行为前缀对齐，并给出四类分词器工件画像。</p>
</dd>
<dt><strong><a href="sprig_semantic_id_enhanced_paths_for_knowledge_graph_based_generative_recom/">SPRIG: Semantic-ID-enhanced Paths for Knowledge Graph-based Generative Recommendation</a></strong></dt>
<dd>
<p>将内容派生的分层语义 ID 注入知识图谱路径语言建模，以 768 个共享码替代逐物品词表，nDCG@10 较 KGGLM 提升 44%/58% 且参数与算力更低。</p>
</dd>
<dt><strong><a href="tlow_flow_based_item_tokenizer_for_recommendation/">Tlow: Flow-based Item Tokenizer for Recommendation</a></strong></dt>
<dd>
<p>用可逆流模型将语义嵌入变换为标准正态潜空间后做PQ独立分词，配合码本引导对齐，微信线上新物品CTR提升11.64%</p>
</dd>
</dl>

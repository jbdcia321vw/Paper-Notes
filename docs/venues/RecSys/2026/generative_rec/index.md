# ✨ 生成式推荐与离散语义 ID

**RecSys 2026** · **12** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式端到端 ×8 · 生成式推荐与离散语义ID ×7 · 召回 ×2 · 端到端生成式 ×2

<dl>
<dt><strong><a href="addressing_cross_stage_decoupling_of_semantic_and_collaborative_signals_in_/">Addressing Cross-Stage Decoupling of Semantic and Collaborative Signals in Generative Recommendation</a></strong></dt>
<dd>
<p>提出 SCRec，通过协同增强分词、语义引导生成与双曲流形对齐三模块双向补全语义与协同信号，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 相对提升 18.7%–26.6%。</p>
</dd>
<dt><strong><a href="cofirec_coarse_to_fine_tokenization_for_generative_recommendation/">CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation</a></strong></dt>
<dd>
<p>将物品元数据按类目→标题→描述→CF信号分层独立量化，自回归由粗到细解码，Leave-One-Out全库排序下平均超越SOTA约6.4%。</p>
</dd>
<dt><strong><a href="distribution_level_contrastive_supervision_for_generative_recommendation/">Distribution-Level Contrastive Supervision for Generative Recommendation</a></strong></dt>
<dd>
<p>用分词器多层码本概率分布作软语义目标，以负对称KL+BPR对比监督对齐推荐器表征，即插即用提升多种生成式骨干。</p>
</dd>
<dt><strong><a href="egr_embedding_native_generative_retrieval_with_a_shared_llm/">EGR: Embedding-Native Generative Retrieval with a Shared LLM</a></strong></dt>
<dd>
<p>用单一共享 LLM 同时编码物品与用户历史，以双对比损失联合训练，免量化免 grounding，线上 CVR +2.91%</p>
</dd>
<dt><strong><a href="empowering_cross_domain_sequential_recommendation_with_hybrid_tokenization_/">Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding</a></strong></dt>
<dd>
<p>跨域共享-专属多塔 RQ-VAE 混合分词 + 两段式串并解码，Leave-One-Out 全库排序下精度 +1.5%、时延 -85.1%。</p>
</dd>
<dt><strong><a href="genpage_towards_end_to_end_generative_homepage_construction_at_netflix/">GenPage: Towards End-to-End Generative Homepage Construction at Netflix</a></strong></dt>
<dd>
<p>用单个 Transformer 将用户上下文 token 化后自回归生成整页多行首页，WBC/RL 后训练，线上 +0.24% 核心参与度且延迟降 20%。</p>
</dd>
<dt><strong><a href="hypothesis_driven_shelf_generation_for_personalised_recommendation/">Hypothesis-Driven Shelf Generation for Personalised Recommendation</a></strong></dt>
<dd>
<p>用自然语言货架假设替代固定模板，解耦规划与目录履约，生成式检索+货架对齐显著提升个性化供给</p>
</dd>
<dt><strong><a href="not_all_branches_are_equal_adaptive_semantic_id_construction_for_generative/">Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation</a></strong></dt>
<dd>
<p>将语义 ID 重构为树路径，用轮廓系数自适应分支+概率质量约束构建 AST，五域 Amazon 全库排序平均相对提升 13.08%</p>
</dd>
<dt><strong><a href="tokens_are_all_you_need_dual_purpose_semantic_ids_for_achieving_llm_level_i/">Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in Recommendation Systems</a></strong></dt>
<dd>
<p>用 RQ-VAE 语义ID 同时承担协同身份与内容重建双职责，以 SiDec 在线解码替代稠密向量 I/O，检索 HitRate@100 提升至 0.2910 且吞吐较原始稠密嵌入快 20.4%。</p>
</dd>
<dt><strong><a href="topology_aware_tokenization_for_generative_recommendation/">Topology-Aware Tokenization for Generative Recommendation</a></strong></dt>
<dd>
<p>揭示 RQ-VAE 语义 ID 量化中的拓扑失真问题，提出由粗到细的三级拓扑蒸馏 TopoTok，在 Amazon 三数据集 Full-Ranking 下 Recall@5 最高提升 9.42%。</p>
</dd>
<dt><strong><a href="towards_generalizable_and_efficient_large_scale_generative_recommenders/">Towards Generalizable and Efficient Large-Scale Generative Recommenders</a></strong></dt>
<dd>
<p>将生成式推荐骨干从2M扩至1B，用偏移幂律缩放律诊断任务天花板，并以MTP、采样softmax与语义塔解决缓存陈旧与冷启动，1M用户影子MRR全线提升。</p>
</dd>
<dt><strong><a href="unirec_bridging_the_expressive_gap_between_generative_and_discriminative_re/">UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute</a></strong></dt>
<dd>
<p>用贝叶斯等价性证明生成式推荐差距源于特征覆盖，以属性链 CoA 前缀解码 + 容量约束 SID + 条件解码上下文补齐，Shopee 线上 GMV +5.60%。</p>
</dd>
</dl>

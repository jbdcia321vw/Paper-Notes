# 🧠 大语言模型推荐 (LLM4Rec)

**ACL / EMNLP (RecSys & CRS) 2026** · **10** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2026 (21)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×9 · ACL 2026 ×3 · 端到端生成式 ×3 · ACL 2026 Findings ×2 · EMNLP 2026 Findings ×2 · 召回 ×2 · 精排 ×2 · 重排 ×2

<dl>
<dt><strong><a href="beyond_retrieval_learning_compact_user_representations_for_scalable_llm_per/">Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization</a></strong></dt>
<dd>
<p>用双轨前缀（用户态+查询感知时序记录）替代提示序列化与逐用户适配器，LaMP 六任务全面领先且每用户参数降 130×</p>
</dd>
<dt><strong><a href="bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_l/">Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders</a></strong></dt>
<dd>
<p>构建 BLaIR 基准，用 11 个 LLM 作语义编码器评测序列推荐、协同过滤与复杂查询商品搜索，揭示其排名与 MTEB 几乎不相关。</p>
</dd>
<dt><strong><a href="eager_enrich_and_align_generative_query_recommendation_from_clicked_items_i/">EAGER: Enrich-and-Align Generative Query Recommendation from Clicked Items in E-commerce Search</a></strong></dt>
<dd>
<p>两阶段生成式 I2Q 查询推荐：课程化 SFT 扩意图覆盖 + GRPO 混合奖励对齐线上约束，线上 UCTR +0.83%、Pay Count +3.49%。</p>
</dd>
<dt><strong><a href="graphlora_structure_aware_low_rank_adaptation_for_large_language_model_reco/">GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation</a></strong></dt>
<dd>
<p>将可训练 GNN 消息传递嵌入 LoRA 瓶颈（A 与 B 之间），让协同拓扑在参数空间传播；在 ML-1M/Amazon-Book 上以 r=8 的极小参数预算超越 CoRA 等 SOTA。</p>
</dd>
<dt><strong><a href="hsuga_llm_enhanced_recommendation_with_hierarchical_semantic_understanding_/">HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment</a></strong></dt>
<dd>
<p>用分阶段可编辑兴趣演化替代长序列一次性摘要，并按用户活跃度分组调节语义对齐强度，三数据集全库排序稳定提升。</p>
</dd>
<dt><strong><a href="memrec_collaborative_memory_augmented_agentic_recommender_system/">MemRec: Collaborative Memory-Augmented Agentic Recommender System</a></strong></dt>
<dd>
<p>将记忆管理从推理中架构解耦，用轻量 LM_Mem 在协同记忆图上做零样本规则剪枝与异步传播，仅向重型 LLM_Rec 注入蒸馏后的高信号协同上下文，四基准全库排序 H@1 最高 +28.98%。</p>
</dd>
<dt><strong><a href="one_polluted_page_is_enough_evaluating_web_content_pollution_in_llm_recomme/">One Polluted Page Is Enough: Evaluating Web Content Pollution in LLM Recommenders</a></strong></dt>
<dd>
<p>提出 FORGE 基准，在冻结检索包上本地改写真实品牌为虚假品牌，量化 12 个 LLM 推荐器被 GEO 网页污染欺骗的比率，并揭示推理与防御的反效果。</p>
</dd>
<dt><strong><a href="rerec_reasoning_augmented_llm_based_recommendation_assistant_via_reinforcem/">ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning</a></strong></dt>
<dd>
<p>用双图奖励塑形、推理感知优势估计与在线课程调度改造 RFT，使 3B LLM 在复杂查询推荐上大幅超越 GRPO 系基线并保留通用能力。</p>
</dd>
<dt><strong><a href="tatk_triple_aware_top_k_learning_with_knowledge_grounded_verification_for_l/">TATK: Triple-Aware Top-K Learning with Knowledge-Grounded Verification for LLM-based Sequential Recommendation</a></strong></dt>
<dd>
<p>用元数据物品图贯穿提示接地、Top-K 位置奖励与推理期 top-M 重排，全库排序下 NDCG@10 最高提升 27.03%</p>
</dd>
<dt><strong><a href="what_makes_llms_effective_sequential_recommenders_a_study_on_preference_int/">What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context</a></strong></dt>
<dd>
<p>揭示二元偏好对齐丢失偏好强度与时序上下文，提出自适应奖励边际 RecPO，五数据集 HR@1 全面领先 S-DPO。</p>
</dd>
</dl>

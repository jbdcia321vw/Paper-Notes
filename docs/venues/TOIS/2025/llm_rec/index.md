# 🧠 大语言模型推荐 (LLM4Rec)

**ACM TOIS 2025** · **9** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** TOIS ×9 · 大语言模型推荐 ×9 · 召回 ×4 · LLM4Rec ×2 · 重排 ×2

<dl>
<dt><strong><a href="enhancing_id_based_recommendation_with_large_language_models/">Enhancing ID-based Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 LoRA 微调 LLM 仅凭纯 ID 序列生成伪交互数据，经双重过滤后注入任意 ID 基推荐模型，全库排序下平均提升 3%–9%。</p>
</dd>
<dt><strong><a href="fellas_enhancing_federated_sequential_recommendation_with_llm_as_external_s/">FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services</a></strong></dt>
<dd>
<p>将 LLM 作为外部服务为联邦序列推荐提供物品级与序列级表征，配合 dX-privacy 扰动与对比学习，在 Full-Ranking 下显著超越 FedSeqRec 并逼近集中式训练。</p>
</dd>
<dt><strong><a href="how_can_recommender_systems_benefit_from_large_language_models_a_survey/">How Can Recommender Systems Benefit from Large Language Models: A Survey</a></strong></dt>
<dd>
<p>以“WHERE×HOW”双正交视角系统梳理 LLM 融入推荐全链路的五阶段角色与四象限适配范式，并给出效率/效果/伦理三大挑战。</p>
</dd>
<dt><strong><a href="mitigating_propensity_bias_of_large_language_models_for_recommender_systems/">Mitigating Propensity Bias of Large Language Models for Recommender Systems</a></strong></dt>
<dd>
<p>用谱条件 iVAE 编码 LLM 侧信息避免维度坍缩，并以反事实推断消除 LLM 倾向性偏差，在 6 个 CF 骨干上全库排序一致提升。</p>
</dd>
<dt><strong><a href="one_model_for_all_large_language_models_are_domain_agnostic_recommendation_/">One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems</a></strong></dt>
<dd>
<p>用 LLM 编码跨域混合标题序列实现域无关推荐，采样 1000 负例评测，SPIAO 上较最优基线相对提升约 19%。</p>
</dd>
<dt><strong><a href="pept_expert_finding_meets_personalized_pre_training/">PEPT: Expert Finding Meets Personalized Pre-training</a></strong></dt>
<dd>
<p>将专家历史问答、投票分与专家ID统一进BERT预训练，用问题级MLM与投票预测双任务学习个性化兴趣与专长，六域CQA全库排序MRR全面领先。</p>
</dd>
<dt><strong><a href="recranker_instruction_tuning_large_language_model_as_ranker_for_top_k_recom/">RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation</a></strong></dt>
<dd>
<p>以自适应用户采样、位置偏移与检索信号增强构建三类排序指令微调 LLM 重排器，并提出混合排序集成，Leave-One-Out 全库排序下 HR/NDCG 最高提升 213%。</p>
</dd>
<dt><strong><a href="reinforced_prompt_personalization_for_recommendation_with_large_language_mo/">Reinforced Prompt Personalization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>将提示词拆为四类句式模式，用多智能体强化学习为每个用户逐句选择并组合个性化提示，在三个数据集上 NDCG@10 相对任务级提示提升 107%–132%。</p>
</dd>
<dt><strong><a href="thoroughly_modeling_multi_domain_pre_trained_recommendation_as_language/">Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language</a></strong></dt>
<dd>
<p>将多域序列推荐彻底转为文本生成任务，用五项指标指导行为-文本双向适配，7 数据集微调与零样本均最优，NDCG@1 最高提升 23%。</p>
</dd>
</dl>

# 🧠 大语言模型推荐 (LLM4Rec)

**SIGIR 2026** · **22** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [SIGIR 2025 (14)](../../2025/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×20 · 召回 ×8 · 生成式端到端 ×5 · 重排 ×4 · LLM推荐 ×2

<dl>
<dt><strong><a href="ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在保持语义层级的同时缓解几何失衡，Full-Ranking 下 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="agentic_multi_source_grounding_for_enhanced_query_intent_understanding_a_do/">Agentic Multi-Source Grounding for Enhanced Query Intent Understanding: A DoorDash Case Study</a></strong></dt>
<dd>
<p>用目录实体检索+Agentic 网页搜索双源 grounding 替代 winner-takes-all 分类，输出有序双意图并由可插拔消歧层裁决，长尾意图准确率 +13.0pp。</p>
</dd>
<dt><strong><a href="bear_towards_beam_search_aware_optimization_for_recommendation_with_large_l/">BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>揭示 SFT 训练与 beam search 推理的剪枝不一致，提出 token 级 top-B 必要条件的可微正则 BEAR，平均提升 12.50% 且训练开销几乎不变。</p>
</dd>
<dt><strong><a href="beyond_static_best_of_n_bayesian_list_wise_alignment_for_llm_based_recommen/">Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用贝叶斯后验融合静态先验与在线 rollout 证据，构建自演化 BoN 目标，突破静态对齐的性能上界，NDCG@5 相对提升 23.1%。</p>
</dd>
<dt><strong><a href="every_preference_has_its_strength_injecting_ordinal_semantics_into_llm_base/">Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders</a></strong></dt>
<dd>
<p>将显式评分建模为LLM词表中的序数语义锚点，以强度感知对齐交互表征，在序列推荐Hit@1与成对偏好准确率上显著超越CF-LLM混合基线。</p>
</dd>
<dt><strong><a href="factorized_latent_reasoning_for_llm_based_recommendation/">Factorized Latent Reasoning for LLM-based Recommendation</a></strong></dt>
<dd>
<p>将 LLM 隐式推理拆为 K 个解耦偏好因子并配 GRPO 对齐，全库约束解码下 N@5 最高提升 10.26%</p>
</dd>
<dt><strong><a href="filling_the_gaps_selective_knowledge_augmentation_for_llm_recommenders/">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></strong></dt>
<dd>
<p>提出 CKP 比较式知识探测量化 LLM 对物品的协同知识缺口，仅对知识贫乏物品选择性注入属性与参考物品，免训练提升 Recall@1 与长尾覆盖。</p>
</dd>
<dt><strong><a href="genfacet_end_to_end_generative_faceted_search_via_multi_task_preference_ali/">GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment in E-Commerce</a></strong></dt>
<dd>
<p>将电商分面搜索重构为分面生成与查询改写两个耦合生成任务，经蒸馏+SFT+GRPO 对齐搜索效用，线上分面 CTR 相对提升 42.0%。</p>
</dd>
<dt><strong><a href="itemrag_item_based_retrieval_augmented_generation_for_llm_based_recommendat/">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 RAG 从粗粒度用户历史检索转向细粒度物品级检索，融合共购与语义信息，标准与冷启动场景全面超越用户级 RAG 基线。</p>
</dd>
<dt><strong><a href="l2rec_towards_dual_view_understanding_of_llms_for_personalized_recommendati/">L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation</a></strong></dt>
<dd>
<p>以参数级双视图 LoRA-MoE 替代输入/输出级融合，个性化路由对齐行为与语义视图，四数据集 N@10 提升 3.87%–8.02%，线上 CTR +9.24%。</p>
</dd>
<dt><strong><a href="lwgr_lagrangian_constrained_personalized_world_knowledge_for_generative_rec/">LWGR: Lagrangian-Constrained Personalized World Knowledge for Generative Recommendation</a></strong></dt>
<dd>
<p>用并行码本生成用户个性化软指令抽取 LLM 世界知识，并以拉格朗日原始-对偶约束控制知识融合，Full-Ranking 下最高相对提升 11.23%，线上广告收入 +1.35%</p>
</dd>
<dt><strong><a href="modular_representation_compression_adapting_llms_for_efficient_and_effectiv/">Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 中间层表征优势 (MRA) 并用模块化解耦压缩修复，128 维表征在 CTR/重排/召回全面超越基线，线上 eCPM +2.82%</p>
</dd>
<dt><strong><a href="multimodal_large_language_models_with_adaptive_preference_optimization_for_/">Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation</a></strong></dt>
<dd>
<p>面向 MLLM 序列推荐，用难度感知重加权 DPO + 高斯噪声正则分布优化，缓解样本难度失衡与跨模态语义偏差，三数据集 NDCG@3 提升 7.4%–11.0%。</p>
</dd>
<dt><strong><a href="promax_exploring_the_potential_of_llm_derived_profiles_with_distribution_sh/">ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems</a></strong></dt>
<dd>
<p>将 LLM 画像视为检索信号而非融合特征，用双路分布重塑（监督+自监督）引导基座模型，LightGCN 上 NDCG@10 最高提升 8.44%。</p>
</dd>
<dt><strong><a href="prompt_unknown_promotion_attacks_against_llm_based_sequential_recommender_s/">Prompt-Unknown Promotion Attacks against LLM-based Sequential Recommender Systems</a></strong></dt>
<dd>
<p>在系统提示词与受害模型均不可见的全黑盒设定下，用进化式提示蒸馏+代理模型+文本与行为双投毒实现长尾商品推广攻击，Beauty 上 HR@10 提升约 3.9 倍。</p>
</dd>
<dt><strong><a href="recgpt_mobile_on_device_large_language_models_for_user_intent_understanding/">RecGPT-Mobile: On-Device Large Language Models for User Intent Understanding in Taobao Feed Recommendation</a></strong></dt>
<dd>
<p>将 Qwen3-0.6B 量化后部署于手机端，用意图漂移触发机制按需推理生成下一搜索意图，线上 GMV +2.5%</p>
</dd>
<dt><strong><a href="rethinking_semantic_collaborative_integration_why_alignment_is_not_enough/">Rethinking Semantic–Collaborative Integration: Why Alignment Is Not Enough</a></strong></dt>
<dd>
<p>论证语义-协同全局对齐假设过强，提出共享+私有潜结构与互补性诊断，最小融合在 Amazon 三域 Full-Ranking 上超越对齐范式最高 41.7%。</p>
</dd>
<dt><strong><a href="spectran_spectral_aware_transformer_based_adapter_for_llm_enhanced_sequenti/">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用全谱注意力+可学习谱位置编码替代静态SVD截断，缓解LLM语义嵌入投影的维度坍缩，四数据集三骨干平均提升9.17%。</p>
</dd>
<dt><strong><a href="think_when_needed_model_aware_reasoning_routing_for_llm_based_ranking/">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></strong></dt>
<dd>
<p>提出轻量即插即用的模型感知推理路由器，在生成前逐实例决定 Think/Non-Think，以 Pareto 前沿策略在降 token 的同时提升排序效用。</p>
</dd>
<dt><strong><a href="unifying_search_and_recommendation_in_llms_via_gradient_multi_subspace_tuni/">Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning</a></strong></dt>
<dd>
<p>用梯度多子空间分解+零空间投影替代 LoRA，在统一 S&amp;R 生成式建模中消解任务梯度冲突并保住通用知识，Qilin/Amazon 上大幅超越 SOTA。</p>
</dd>
<dt><strong><a href="verifiable_reasoning_for_llm_based_generative_recommendation/">Verifiable Reasoning for LLM-based Generative Recommendation</a></strong></dt>
<dd>
<p>提出 reason-verify-recommend 范式，用混合验证器对隐空间推理逐步审计与修正，四数据集 Recall@10 最高提升约 47%</p>
</dd>
<dt><strong><a href="when_how_to_write_for_personalized_demand_aware_query_rewriting_in_video_se/">When &amp; How to Write for Personalized Demand-aware Query Rewriting in Video Search</a></strong></dt>
<dd>
<p>用后验行为挖掘「何时改写」样本、SFT+GRPO 对齐检索索引风格，并以并行轻量召回实现零感知延迟的个性化查询改写。</p>
</dd>
</dl>

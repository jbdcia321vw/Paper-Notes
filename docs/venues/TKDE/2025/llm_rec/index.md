# 🧠 大语言模型推荐 (LLM4Rec)

**IEEE TKDE 2025** · **4** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [IEEE TKDE 2026 (1)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** TKDE ×4 · 大语言模型推荐 ×4 · 精排 ×2

<dl>
<dt><strong><a href="collm_integrating_collaborative_embeddings_into_large_language_models_for_r/">CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>将传统协同过滤模型学到的用户/物品表征经 MLP 映射进 LLM 词嵌入空间，以两阶段调优兼顾暖启动与冷启动，AUC 相对 TALLRec 提升约 8%。</p>
</dd>
<dt><strong><a href="enhancing_collaborative_semantics_of_language_model_driven_recommendations_/">Enhancing Collaborative Semantics of Language Model-Driven Recommendations via Graph-Aware Learning</a></strong></dt>
<dd>
<p>用多跳提示+图感知对比学习把 GNN 聚合意图注入 LLM，在三个 Amazon/Yelp 数据集 Full-Ranking 下 HR@10 最高提升约 15%。</p>
</dd>
<dt><strong><a href="exact_and_efficient_unlearning_for_large_language_model_based_recommendatio/">Exact and Efficient Unlearning for Large Language Model-based Recommendation</a></strong></dt>
<dd>
<p>首个面向 LLMRec 的精确遗忘框架 APA：语义分片训练子 LoRA、仅重训受影响分片，并以样本自适应参数级聚合实现单次推理，遗忘提速约 4 倍且 AUC 反超全量重训。</p>
</dd>
<dt><strong><a href="preference_consistent_knowledge_distillation_for_recommender_system/">Preference-Consistent Knowledge Distillation for Recommender System</a></strong></dt>
<dd>
<p>揭示特征蒸馏中投影器导致的偏好不一致现象，提出成对/列表两种正则项约束投影器，全库排序下 NDCG@20 最高提升 20%。</p>
</dd>
</dl>

# 🧠 大语言模型推荐 (LLM4Rec)

**CIKM 2025** · **6** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [CIKM 2026 (9)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×6 · 召回 ×4 · 精排 ×2

<dl>
<dt><strong><a href="alibaba_international_e_commerce_product_search_competition_dcuragons_team_/">Alibaba International E-commerce Product Search Competition DcuRAGONs Team Technical Report</a></strong></dt>
<dd>
<p>以 Gemma-3-12B 为骨干，用翻译增强+任务自适应预训练+类别感知查询分组划分，在阿里国际电商多语言相关性竞赛夺冠（F1_avg≈89.1）。</p>
</dd>
<dt><strong><a href="exploring_the_upper_limits_of_text_based_collaborative_filtering_using_larg/">Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights</a></strong></dt>
<dd>
<p>将 TCF 物品编码器从 125M 一路放大到 175B，系统检验缩放极限、通用表征、对 IDCF 的竞争力与零样本迁移四项命题。</p>
</dd>
<dt><strong><a href="llm_enhanced_linear_autoencoders_for_recommendation/">LLM-Enhanced Linear Autoencoders for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义矩阵构造 item-item 相关阵并以知识蒸馏正则注入 EASE 闭式解，全库排序下 R@20 平均提升 27.6%。</p>
</dd>
<dt><strong><a href="prompt_tuning_as_user_inherent_profile_inference_machine/">Prompt Tuning as User Inherent Profile Inference Machine</a></strong></dt>
<dd>
<p>将用户潜在画像建模为可训练软提示，用因果掩码+EM 推断、码本量化成协同 ID，AUC 最高相对提升 4.77%，已上线华为应用市场。</p>
</dd>
<dt><strong><a href="self_surrogate_light_feature_selection_with_large_language_models_in_deep_r/">SELF: Surrogate-light Feature Selection with Large Language Models in Deep Recommender Systems</a></strong></dt>
<dd>
<p>用多 LLM 迭代提示生成语义特征排序，再以可学习桥接向量对齐任务空间，实现轻量特征选择，AUC 全面领先且线上 CTR +3.01%。</p>
</dd>
<dt><strong><a href="trawl_external_knowledge_enhanced_recommendation_with_llm_assistance/">TRAWL: External Knowledge-Enhanced Recommendation with LLM Assistance</a></strong></dt>
<dd>
<p>用 LLM 从原始外部文本抽取推荐知识，再以对比学习适配器对齐语义与行为空间，ML-1M 上 AUC 稳定提升且微信线上 A/B 三项指标正向。</p>
</dd>
</dl>

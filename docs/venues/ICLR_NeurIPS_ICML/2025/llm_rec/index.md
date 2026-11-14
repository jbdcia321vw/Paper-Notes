# 🧠 大语言模型推荐 (LLM4Rec)

**ICLR / NeurIPS / ICML (RecSys) 2025** · **10** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×10 · NeurIPS ×6 · 生成式端到端 ×4 · 精排 ×3 · ICLR ×2 · 召回 ×2

<dl>
<dt><strong><a href="can_llms_outshine_conventional_recommenders_a_comparative_evaluation/">Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation</a></strong></dt>
<dd>
<p>RecBench 系统对比 17 个 LLM 与 DLRM 在 4 种物品表征、CTR/SeqRec 双任务上的精度-效率权衡，LLM 精度领先但推理慢千倍。</p>
</dd>
<dt><strong><a href="context_steering_controllable_personalization_at_inference_time/">Context Steering: Controllable Personalization at Inference Time</a></strong></dt>
<dd>
<p>免训练解码期个性化：用有/无上下文两次前向的似然差线性缩放上下文影响力，实现可控个性化推荐与贝叶斯意图反演。</p>
</dd>
<dt><strong><a href="face_a_general_framework_for_mapping_collaborative_filtering_embeddings_int/">FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens</a></strong></dt>
<dd>
<p>用解耦投影+RQ-VAE 将 CF 嵌入映射为冻结 LLM 词表中的离散 token，配合对比对齐实现免微调的语义可解释与性能增益。</p>
</dd>
<dt><strong><a href="fine_grained_list_wise_alignment_for_generative_medication_recommendation/">Fine-grained List-wise Alignment for Generative Medication Recommendation</a></strong></dt>
<dd>
<p>将用药推荐重构为逐药序列决策，提出逐步GRPO与势函数奖励塑形实现细粒度信用分配，在MIMIC-III上Jaccard达0.4836并支持安全-精度可控权衡。</p>
</dd>
<dt><strong><a href="improving_llms_for_recommendation_with_out_of_vocabulary_tokens/">Improving LLMs for Recommendation with Out-Of-Vocabulary Tokens</a></strong></dt>
<dd>
<p>用元路径聚类生成 OOV 层级 token 构建用户/物品 META ID，兼顾记忆性与多样性，序列推荐 HR@10 最高提升 30%+。</p>
</dd>
<dt><strong><a href="language_representations_can_be_what_recommenders_need_findings_and_potenti/">Language Representations Can be What Recommenders Need: Findings and Potentials</a></strong></dt>
<dd>
<p>揭示语言空间与推荐行为空间同构，提出纯语言表征的 AlphaRec，在 Amazon 三数据集 Full-Ranking 上超越 ID-based CF 6.8%–9.8%。</p>
</dd>
<dt><strong><a href="r_2_ec_towards_large_recommender_models_with_reasoning/">R$^2$ec: Towards Large Recommender Models with Reasoning</a></strong></dt>
<dd>
<p>双头架构统一推理生成与物品打分，RecPO 融合奖励 RL 联合优化，Full-Ranking 下 NDCG@10 最高提升 72%。</p>
</dd>
<dt><strong><a href="slmrec_distilling_large_language_models_into_small_for_sequential_recommend/">SLMRec: Distilling Large Language Models into Small for Sequential Recommendation</a></strong></dt>
<dd>
<p>发现 LLM 中间层冗余，用逐层分块特征蒸馏将 8 层教师压缩为 4 层学生，仅 13% 参数达 6.6x/8.0x 加速且性能不降。</p>
</dd>
<dt><strong><a href="think_before_recommendation_autonomous_reasoning_enhanced_recommender/">Think before Recommendation: Autonomous Reasoning-enhanced Recommender</a></strong></dt>
<dd>
<p>用纯 GRPO 强化学习替代教师蒸馏，让单个 LLM 自主习得「用户分析→物品分析→匹配→打分」推理链，评分预测 MAE 最高降 29.9%。</p>
</dd>
<dt><strong><a href="tree_of_preferences_for_diversified_recommendation/">Tree of Preferences for Diversified Recommendation</a></strong></dt>
<dd>
<p>用 LLM 构建粗到细偏好树挖掘欠探索兴趣并合成交互去偏，在 Twitter/Weibo/Amazon 上同时提升多样性与相关性。</p>
</dd>
</dl>

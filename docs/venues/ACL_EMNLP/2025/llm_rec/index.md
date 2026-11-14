# 🧠 大语言模型推荐 (LLM4Rec)

**ACL / EMNLP (RecSys & CRS) 2025** · **19** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×19 · ACL 2025 ×8 · 召回 ×5 · ACL ×4 · 生成式端到端 ×4 · 精排 ×4 · EMNLP 2025 ×3 · ACL/EMNLP 2025 ×2

<dl>
<dt><strong><a href="a_text_based_recommender_system_that_leverages_explicit_affective_state_pre/">A Text-Based Recommender System that Leverages Explicit Affective State Preferences</a></strong></dt>
<dd>
<p>提出情感-认知状态显式偏好推荐任务，用 GPT-4o 从书评挖掘 AC 语句构建数据集，Transformer 融合长短期偏好与 AC 描述，全库排序 HR@10 达 15.6%。</p>
</dd>
<dt><strong><a href="copl_collaborative_preference_learning_for_personalizing_llms/">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></strong></dt>
<dd>
<p>用用户-回复二部图协同过滤学习用户表征，结合 MoLE 路由个性化奖励模型，并支持无优化未见用户适配，在 UltraFeedback-P 等基准上显著超越个性化奖励模型基线。</p>
</dd>
<dt><strong><a href="cove_compressed_vocabulary_expansion_makes_better_llm_based_recommender_sys/">CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>将物品映射为扩展词表唯一 token，用 logits 直接打分替代文本生成，并以哈希压缩嵌入表，实现最高 62% 精度提升与约 100× 推理加速。</p>
</dd>
<dt><strong><a href="decoding_in_latent_spaces_for_efficient_inference_in_llm_based_recommendati/">Decoding in Latent Spaces for Efficient Inference in LLM-based Recommendation</a></strong></dt>
<dd>
<p>用训练样本末层隐状态聚合出物品潜在表征，以 L2 距离匹配替代自回归语言空间解码，推理提速 10 倍以上且性能不降反升。</p>
</dd>
<dt><strong><a href="easyrec_simple_yet_effective_language_models_for_recommendation/">EasyRec: Simple yet Effective Language Models for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 生成用户/物品画像并以对比学习对齐协同信号，训练轻量文本编码器实现零样本推荐，Full-Ranking 下 Recall@20 最高提升 32%。</p>
</dd>
<dt><strong><a href="generative_product_recommendations_for_implicit_superlative_queries/">Generative Product Recommendations for Implicit Superlative Queries</a></strong></dt>
<dd>
<p>提出 SUPERB 四级标注体系与深思式属性生成，用 LLM 列表级重排处理隐式最高级商品查询，nDCG@10 相对 BM25 提升约 18%。</p>
</dd>
<dt><strong><a href="kerl_knowledge_enhanced_personalized_recipe_recommendation_using_large_lang/">KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models</a></strong></dt>
<dd>
<p>用 FoodKG 子图作上下文 + 多 LoRA 适配器，统一完成约束食谱推荐、烹饪步骤生成与微量营养素估计，KGQA 基准 F1 达 0.973。</p>
</dd>
<dt><strong><a href="knowledge_graph_retrieval_augmented_generation_for_llm_based_recommendation/">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用跳场知识子图索引+流行度选择性检索+GNN软提示，将KG结构知识注入冻结LLM，MovieLens-1M ACC 相对提升58.6%且推理仅慢0.14s</p>
</dd>
<dt><strong><a href="lamp_val_large_language_models_empower_personalized_valuation_in_auction/">LaMP-Val: Large Language Models Empower Personalized Valuation in Auction</a></strong></dt>
<dd>
<p>用 LLM 从用户评论语义中做个性化估值，构建 23k 估值数据集 + 多样化指令 SFT，并提出 PU/PV 闭环拍卖指标，PU 较 GPT-4 提升约 6.5 倍。</p>
</dd>
<dt><strong><a href="maps_motivation_aware_personalized_search_via_llm_driven_consultation_align/">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></strong></dt>
<dd>
<p>用 LLM+MoAE 将售前咨询动机与查询/商品/历史双向对齐，Commercial 与 Amazon 上排序 HR@10 提升约 20%/35%。</p>
</dd>
<dt><strong><a href="mira_empowering_one_touch_ai_services_on_smartphones_with_mllm_based_instru/">MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation</a></strong></dt>
<dd>
<p>用结构化推理+模板检索增强+前缀树约束解码，把长按图文触发对象映射到预定义手机AI指令，F1较SFT提升约60%。</p>
</dd>
<dt><strong><a href="personax_a_recommendation_agent_oriented_user_modeling_framework_for_long_b/">PersonaX: A Recommendation Agent Oriented User Modeling Framework for Long Behavior Sequence</a></strong></dt>
<dd>
<p>面向推荐智能体的离线多画像用户建模框架，用聚类+原型-多样性均衡采样压缩长行为序列，30-50%数据即超越全量基线并大幅降低在线延迟。</p>
</dd>
<dt><strong><a href="q_strum_debate_query_driven_contrastive_summarization_for_recommendation_co/">Q-STRUM Debate: Query-Driven Contrastive Summarization for Recommendation Comparison</a></strong></dt>
<dd>
<p>用双人格辩论式提示替代单次摘要，为查询驱动的推荐对比摘要注入对比性与多样性，在酒店/餐厅/旅游三域 Win Rate 最高达 89%。</p>
</dd>
<dt><strong><a href="reclm_recommendation_instruction_tuning/">RecLM: Recommendation Instruction Tuning</a></strong></dt>
<dd>
<p>以两轮协同指令微调+RL画像精炼为LLM注入协同信号，作为即插即用画像增强器在零样本冷启动下带来最高 200%+ 的 Recall 提升。</p>
</dd>
<dt><strong><a href="rethinking_llm_based_recommendations_a_personalized_query_driven_parallel_i/">Rethinking LLM-Based Recommendations: A Personalized Query-Driven Parallel Integration</a></strong></dt>
<dd>
<p>提出 QUEREC 并行框架，用 LLM 零样本生成个性化用户查询与特征增强物品描述，与 CF 模型并行全库检索后自适应重排，Full-Ranking 下最高提升 57%。</p>
</dd>
<dt><strong><a href="similarity_value_consultation_value_assessment_and_alignment_for_personaliz/">Similarity = Value? Consultation Value Assessment and Alignment for Personalized Search</a></strong></dt>
<dd>
<p>指出语义相似度无法代理咨询价值，提出场景范围/后验行为/时间衰减三维价值评估与 CAI 对齐的 VAPS，检索排序双任务全面领先。</p>
</dd>
<dt><strong><a href="the_efficiency_vs_accuracy_trade_off_optimizing_rag_enhanced_llm_recommende/">The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit</a></strong></dt>
<dd>
<p>用 GCN 检索器替代 LLM embedder 加速 RAG，并以多头早退动态终止 LLM 推理，在 5 个 CTR 数据集上兼顾 AUC 与吞吐。</p>
</dd>
<dt><strong><a href="user_behavior_prediction_as_a_generic_robust_scalable_and_low_cost_evaluati/">User Behavior Prediction as a Generic, Robust, Scalable, and Low-Cost Evaluation Strategy for Estimating Generalization in LLMs</a></strong></dt>
<dd>
<p>提出以用户行为预测（个性化）作为抗数据污染的 LLM 泛化评测范式，用熵-交叉熵拐点框架在电影/音乐推荐上量化 GPT-4o、GPT-4o-mini、Llama 的泛化边界。</p>
</dd>
<dt><strong><a href="user_feedback_alignment_for_llm_powered_exploration_in_large_scale_recommen/">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></strong></dt>
<dd>
<p>解耦新颖性与用户对齐双 LLM，用集体隐式反馈训练对齐模型做 best-of-n 选择，线上同时提升新颖曝光与满意度。</p>
</dd>
</dl>

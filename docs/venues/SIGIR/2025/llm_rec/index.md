# 🧠 大语言模型推荐 (LLM4Rec)

**SIGIR 2025** · **14** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [SIGIR 2026 (22)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×14 · 召回 ×4 · 生成式端到端 ×2 · 端到端生成式 ×2 · 精排 ×2

<dl>
<dt><strong><a href="adore_autonomous_domain_oriented_relevance_engine_for_e_commerce/">ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce</a></strong></dt>
<dd>
<p>用 CoT-LLM 自动挖掘难样本、KTO 对齐用户行为、错误类型感知合成对抗样本，再以关键属性显式蒸馏到轻量 BERT，离线 F1 85.45%、线上 CTR +1.24%。</p>
</dd>
<dt><strong><a href="corona_a_coarse_to_fine_framework_for_graph_based_recommendation_with_large/">CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 做偏好/意图推理在交互图上逐级缩小候选子图，再由 GNN 精排，全库排序下 Recall 平均相对提升 18.6%。</p>
</dd>
<dt><strong><a href="do_llms_memorize_recommendation_datasets_a_preliminary_study_on_movielens_1/">Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M</a></strong></dt>
<dd>
<p>用少样本提示探测 GPT/Llama 对 MovieLens-1M 的物品、用户与交互记忆覆盖率，揭示记忆量与推荐性能强相关且放大流行度偏置。</p>
</dd>
<dt><strong><a href="efficiency_unleashed_inference_acceleration_for_llm_based_recommender_syste/">Efficiency Unleashed: Inference Acceleration for LLM-based Recommender Systems with Speculative Decoding</a></strong></dt>
<dd>
<p>将检索式推测解码引入 LLM 离线知识生成，用定制检索池与松弛验证实现 3-5× 无损加速，线上省 67% 算力。</p>
</dd>
<dt><strong><a href="improving_llm_powered_recommendations_with_personalized_information/">Improving LLM-powered Recommendations with Personalized Information</a></strong></dt>
<dd>
<p>将用户偏好与物品感知两条 CoT 显式抽取并注入 CRM 召回与 LLM 重排，Leave-One-Out 全库排序下 Hit@10 最高提升约 11%，并显著降低位置偏差。</p>
</dd>
<dt><strong><a href="intent_representation_learning_with_large_language_model_for_recommendation/">Intent Representation Learning with Large Language Model for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 思维链抽取文本意图，与交互意图双塔对齐+动量蒸馏匹配，作为模型无关插件在三个数据集上稳定提升 Recall/NDCG。</p>
</dd>
<dt><strong><a href="large_language_models_enhanced_hyperbolic_space_recommender_systems/">Large Language Models Enhanced Hyperbolic Space Recommender Systems</a></strong></dt>
<dd>
<p>HyperLLM 用 LLM 生成层级标签与偏好摘要，经元优化两阶段训练将结构/语义层级信息注入双曲推荐，Recall@20 最高提升 41.7%。</p>
</dd>
<dt><strong><a href="msl_not_all_tokens_are_what_you_need_for_tuning_llm_as_a_recommender/">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></strong></dt>
<dd>
<p>提出掩码 Softmax 损失屏蔽虚构物品 token，并配自适应温度策略，四数据集 NDCG@10 平均提升 42.24%。</p>
</dd>
<dt><strong><a href="multi_grained_patch_training_for_efficient_llm_based_recommendation/">Multi-Grained Patch Training for Efficient LLM-based Recommendation</a></strong></dt>
<dd>
<p>用层次化 item/session patch 压缩长交互历史，两阶段训练让 LLM 理解压缩表征，同 token 预算下 HR@20 最高提升 72%。</p>
</dd>
<dt><strong><a href="pre_train_align_and_disentangle_empowering_sequential_recommendation_with_l/">Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 MK-MMD 特征核做推荐锚定对齐，再以三专家频率门控解耦，Full-Ranking 下冷物品 HR@10 提升最高 462%。</p>
</dd>
<dt><strong><a href="process_supervised_llm_recommenders_via_flow_guided_tuning/">Process-Supervised LLM Recommenders via Flow-guided Tuning</a></strong></dt>
<dd>
<p>用 GFlowNet 将物品级奖励分解为 token 级过程奖励，替代 SFT 的似然最大化，在 Amazon 三数据集上同时提升准确率、公平性与多样性。</p>
</dd>
<dt><strong><a href="review_driven_personalized_preference_reasoning_with_large_language_models_/">Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>用评论抽取偏好画像并蒸馏逐步推理，评分预测与重排双任务均超 CF 与 LLM 基线，解释性显著提升</p>
</dd>
<dt><strong><a href="synthtrips_a_knowledge_grounded_framework_for_benchmark_query_generation_fo/">SynthTRIPs: A Knowledge-Grounded Framework for Benchmark Query Generation for Personalized Tourism Recommenders</a></strong></dt>
<dd>
<p>用 PersonaHub 人格 + 可持续性过滤器 + KB 接地，让 LLM 批量合成个性化城市游查询，4,604 条查询经 J-LLM 与双专家双轨验证。</p>
</dd>
<dt><strong><a href="you_are_what_you_bought_generating_customer_personas_for_e_commerce_applica/">You Are What You Bought: Generating Customer Personas for E-commerce Applications</a></strong></dt>
<dd>
<p>用 LLM 标注少量原型用户、随机游走传播生成可读人物画像，并以 RevAff 近似加速，画像注入 GCN 推荐最高提升 12%。</p>
</dd>
</dl>

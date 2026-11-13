---
hide:
  - toc
---

# 🏛️ ACL / EMNLP (RecSys & CRS) 2026 推荐系统论文解读

30 篇 ACL / EMNLP (RecSys & CRS) 2026 论文解读，覆盖 10 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/clusterrag_cluster_based_collaborative_filtering_for_personalized_retrieval/">ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation</a></strong></dt>
<dd>
<p>用 HDBSCAN 将用户按画像嵌入聚类，簇内 ColBERTv2 排序选近邻，簇-文档两级检索融合协同画像做个性化 RAG，LaMP 全任务最优。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl>
<dt><strong><a href="graph_rec/continual_graph_memory_for_adaptive_recommendation_under_intent_drift/">Continual Graph Memory for Adaptive Recommendation under Intent Drift</a></strong></dt>
<dd>
<p>将知识图谱视为可写双记忆（语义图记忆+情景教训记忆），冻结参数下仅靠质量门控图编辑实现意图漂移自适应重排。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl>
<dt><strong><a href="sequential_rec/two_sided_state_space_models_for_sequential_recommendation_with_non_random_/">Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback</a></strong></dt>
<dd>
<p>将评论视为双侧状态空间中的内生事件，用 MNAR 观测模式、非对称评论结转与局部图传播联合演化用户/物品状态，Full-Ranking 下 R@20 较 BSARec 提升 14.8%–18.8%。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (2)</h2>
<dl>
<dt><strong><a href="generative_rec/from_past_to_path_masked_history_learning_for_next_item_prediction_in_gener/">From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation</a></strong></dt>
<dd>
<p>在自回归生成式推荐上叠加掩码历史重建辅助任务，用熵引导掩码与课程调度弥合训练-推理鸿沟，Amazon 三域 Full-Ranking 全面 SOTA。</p>
</dd>
<dt><strong><a href="generative_rec/what_makes_an_ideal_quote_recommending_unexpected_yet_rational_quotations_v/">What Makes an Ideal Quote? Recommending &#x27;Unexpected yet Rational&#x27; Quotations via Novelty</a></strong></dt>
<dd>
<p>提出 NovelQR，用生成式标签智能体构建深层语义知识库并做标签增强检索，再以 token 级新颖性估计器缓解自回归续写偏差进行重排，双语数据集上人类评审 78% 偏好本系统。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (10)</h2>
<dl>
<dt><strong><a href="llm_rec/beyond_retrieval_learning_compact_user_representations_for_scalable_llm_per/">Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization</a></strong></dt>
<dd>
<p>用双轨前缀（用户态+查询感知时序记录）替代提示序列化与逐用户适配器，LaMP 六任务全面领先且每用户参数降 130×</p>
</dd>
<dt><strong><a href="llm_rec/bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_l/">Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders</a></strong></dt>
<dd>
<p>构建 BLaIR 基准，用 11 个 LLM 作语义编码器评测序列推荐、协同过滤与复杂查询商品搜索，揭示其排名与 MTEB 几乎不相关。</p>
</dd>
<dt><strong><a href="llm_rec/eager_enrich_and_align_generative_query_recommendation_from_clicked_items_i/">EAGER: Enrich-and-Align Generative Query Recommendation from Clicked Items in E-commerce Search</a></strong></dt>
<dd>
<p>两阶段生成式 I2Q 查询推荐：课程化 SFT 扩意图覆盖 + GRPO 混合奖励对齐线上约束，线上 UCTR +0.83%、Pay Count +3.49%。</p>
</dd>
<dt><strong><a href="llm_rec/graphlora_structure_aware_low_rank_adaptation_for_large_language_model_reco/">GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation</a></strong></dt>
<dd>
<p>将可训练 GNN 消息传递嵌入 LoRA 瓶颈（A 与 B 之间），让协同拓扑在参数空间传播；在 ML-1M/Amazon-Book 上以 r=8 的极小参数预算超越 CoRA 等 SOTA。</p>
</dd>
<dt><strong><a href="llm_rec/hsuga_llm_enhanced_recommendation_with_hierarchical_semantic_understanding_/">HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment</a></strong></dt>
<dd>
<p>用分阶段可编辑兴趣演化替代长序列一次性摘要，并按用户活跃度分组调节语义对齐强度，三数据集全库排序稳定提升。</p>
</dd>
<dt><strong><a href="llm_rec/memrec_collaborative_memory_augmented_agentic_recommender_system/">MemRec: Collaborative Memory-Augmented Agentic Recommender System</a></strong></dt>
<dd>
<p>将记忆管理从推理中架构解耦，用轻量 LM_Mem 在协同记忆图上做零样本规则剪枝与异步传播，仅向重型 LLM_Rec 注入蒸馏后的高信号协同上下文，四基准全库排序 H@1 最高 +28.98%。</p>
</dd>
<dt><strong><a href="llm_rec/one_polluted_page_is_enough_evaluating_web_content_pollution_in_llm_recomme/">One Polluted Page Is Enough: Evaluating Web Content Pollution in LLM Recommenders</a></strong></dt>
<dd>
<p>提出 FORGE 基准，在冻结检索包上本地改写真实品牌为虚假品牌，量化 12 个 LLM 推荐器被 GEO 网页污染欺骗的比率，并揭示推理与防御的反效果。</p>
</dd>
<dt><strong><a href="llm_rec/rerec_reasoning_augmented_llm_based_recommendation_assistant_via_reinforcem/">ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning</a></strong></dt>
<dd>
<p>用双图奖励塑形、推理感知优势估计与在线课程调度改造 RFT，使 3B LLM 在复杂查询推荐上大幅超越 GRPO 系基线并保留通用能力。</p>
</dd>
<dt><strong><a href="llm_rec/tatk_triple_aware_top_k_learning_with_knowledge_grounded_verification_for_l/">TATK: Triple-Aware Top-K Learning with Knowledge-Grounded Verification for LLM-based Sequential Recommendation</a></strong></dt>
<dd>
<p>用元数据物品图贯穿提示接地、Top-K 位置奖励与推理期 top-M 重排，全库排序下 NDCG@10 最高提升 27.03%</p>
</dd>
<dt><strong><a href="llm_rec/what_makes_llms_effective_sequential_recommenders_a_study_on_preference_int/">What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context</a></strong></dt>
<dd>
<p>揭示二元偏好对齐丢失偏好强度与时序上下文，提出自适应奖励边际 RecPO，五数据集 HR@1 全面领先 S-DPO。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (9)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_un/">Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents</a></strong></dt>
<dd>
<p>将决策解耦为文档客观打分矩阵与贝叶斯偏好推断，用决策感知信息增益主动提问，Top-1 准确率最高提升约 20%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/from_recall_to_forgetting_benchmarking_long_term_memory_for_personalized_ag/">From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents</a></strong></dt>
<dd>
<p>提出 Memora 长周期记忆基准与 FAMA 遗忘感知指标，揭示 LLM 与记忆智能体在记忆突变下频繁复用失效记忆。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/harpo_hierarchical_agentic_reasoning_for_user_aligned_conversational_recomm/">HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation</a></strong></dt>
<dd>
<p>将对话推荐重构为多维质量驱动的结构化决策，用分层偏好奖励引导树搜索推理，ReDial R@10 达 29.8%（+18.7%）。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/intent_driven_semantic_id_generation_for_grounded_conversational_news_recom/">Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation</a></strong></dt>
<dd>
<p>用 LLM 直接生成三层语义 ID 前缀再模糊匹配新闻池，结构上消除幻觉，冷启动 L1 达 18.0%，延迟 85ms。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/mirroring_users_towards_building_preference_aligned_user_simulator_with_use/">Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation</a></strong></dt>
<dd>
<p>用 LLM 生成 EKB 决策过程作澄清、按认知不确定性蒸馏并拒绝采样去噪，微调 3B 模拟器在 8 域上超越 GPT-5 与教师模型。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/re2a_situated_conversational_recommendation_via_rubric_based_preference_rea/">Re2A: Situated Conversational Recommendation via Rubric-based Preference Reasoning and Alignment</a></strong></dt>
<dd>
<p>将情境对话推荐重构为“先推理后对齐”两阶段，用自动评分标准引导偏好状态推理并以异构负例 DPO 实现用户-场景双重对齐，SIMMC 2.1 上 Hit@1 提升 8.17pt。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/rectoolbench_benchmarking_recommendation_specific_tool_orchestration_under_/">RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent</a></strong></dt>
<dd>
<p>首个基于 MCP 的模糊意图推荐工具编排基准，1288 条四级难度可执行任务，揭示语法合法≠推荐成功的执行-结果鸿沟。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/towards_effective_structured_context_modeling_for_conversational_recommende/">Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search</a></strong></dt>
<dd>
<p>将对话上下文建模为双节点 MCTS 树，ELNode 搜索追问动作、EXNode 精炼检索查询，Redial/OpenDialKG 上 SR 平均提升 9.07%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conve/">Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 SiPeR 框架，用场景转移估计解决“去哪”与贝叶斯逆向推理解决“要什么”，在 SIMMC 2.1 与 SCREEN 上 R@1 平均提升约 10.9%/10.6%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/reasonrec_a_reasoning_augmented_multimodal_agent_for_unified_recommendation/">ReasonRec: A Reasoning-Augmented Multimodal Agent for Unified Recommendation</a></strong></dt>
<dd>
<p>将推荐重构为 Observe-Deliberate-Act 三阶段 CoT 推理智能体，以证据视界课程学习与不确定性引导工具委派，在五数据集四任务上 HR@5 相对提升超 30% 并降低推理延迟。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/horizon_a_benchmark_for_in_the_wild_user_behaviour_modeling/">HORIZON: A Benchmark for In-the-wild User Behaviour Modeling</a></strong></dt>
<dd>
<p>重构 Amazon Reviews 为 54M 用户跨域长程基准，提出时间-用户双轴四象限评估与 LLM 查询改写/长程建模任务，揭示 ID 模型在时间外推下崩塌。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/world_model_guided_reinforcement_learning_via_counterfactual_user_engagemen/">World Model-Guided Reinforcement Learning via Counterfactual User Engagement Simulation</a></strong></dt>
<dd>
<p>将推荐视为动作、异构反馈视为观测，用语言世界模型 UEWM 做反事实用户模拟器，为 1.7B 策略提供同状态稠密奖励，跨域零样本超越大模型。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/authority_bias_in_conversational_search_engines_for_academic_paper_recommen/">Authority Bias in Conversational Search Engines for Academic Paper Recommendation</a></strong></dt>
<dd>
<p>内容受控反事实审计揭示 LLM 学术推荐存在显著权威偏见，并提出 say-do gap 与前沿模型反向失效两大诊断现象。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/content_fuzzing_for_escaping_information_cocoons_on_digital_social_media/">Content Fuzzing for Escaping Information Cocoons on Digital Social Media</a></strong></dt>
<dd>
<p>以立场检测置信度为灰盒反馈，用 LLM 迭代改写帖子翻转机器立场标签，助创作者突破信息茧房，跨 3 数据集 4 模型平均逃逸率 40%–91%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/rethinking_fairness_in_llm_based_recommender_systems_a_survey/">Rethinking Fairness in LLM-Based Recommender Systems: A Survey</a></strong></dt>
<dd>
<p>首个 LLM4Rec 公平性综述，提出「偏置机制×公平目标」二维分类法，系统梳理数据集、评估协议、缓解策略与可信交叉议题。</p>
</dd>
</dl>
</div>

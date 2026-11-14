# 🗣️ 对话推荐与智能体仿真 (Conversational & Agentic Recommendation)

> **研究范畴**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

当前方向共收录 **116** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/agent_conversational_rec/llm_powered_user_simulator_for_recommender_system/">LLM-Powered User Simulator for Recommender System</a></strong></dt>
<dd>
<p>用 LLM 离线蒸馏物品优缺点关键词，构建关键词匹配+语义相似+序列统计三模型集成用户模拟器，为 RL 推荐提供可解释高保真环境。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/llm_rec/contextualizing_recommendation_explanations_with_llms_a_user_study/">Contextualizing Recommendation Explanations with LLMs: A User Study</a></strong></dt>
<dd>
<p>预注册在线实验(N=759)+访谈(N=30)对比无解释/通用解释/情境化解释，情境化解释显著提升认知需求与观看意愿，但对功利/情感需求增益有限。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/agent_conversational_rec/rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm_based_conve/">RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems</a></strong></dt>
<dd>
<p>首个面向对话推荐场景的人类标注 ToM 基准，从认知推断与行为预测两维评估 LLM，揭示细粒度意图与谄媚偏差瓶颈</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/seeking_help_facing_harm_auditing_tiktok_s_mental_health_recommendations/">Seeking Help, Facing Harm: Auditing TikTok&#x27;s Mental Health Recommendations</a></strong></dt>
<dd>
<p>30个LLM智能体7天控制审计TikTok推荐流，揭示交互行为主导心理健康内容曝光、求助意图无法形成安全边界。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/agent_conversational_rec/styletailor_towards_personalized_fashion_styling_via_hierarchical_negative_/">StyleTailor: Towards Personalized Fashion Styling via Hierarchical Negative Feedback</a></strong></dt>
<dd>
<p>首个统一个性化服装设计、购物推荐与虚拟试穿的协作智能体框架，以三级分层负反馈闭环迭代优化，在自建 64 样本基准上 Style Consistency 相对基线提升 39.4%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/agent_conversational_rec/when_top_ranked_recommendations_fail_modeling_multi_granular_negative_feedb/">When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation</a></strong></dt>
<dd>
<p>三智能体协同建模多粒度负反馈，S-GRPO 渐进奖励实现可解释视频推荐，线上快滑率降 9.4%</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/a_survey_on_llm_powered_agents_for_recommender_systems/">A Survey on LLM-powered Agents for Recommender Systems</a></strong></dt>
<dd>
<p>系统综述 LLM Agent 推荐的三范式（推荐/交互/仿真）与四模块架构，梳理数据集与评测协议并指出统一评测缺失。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/empowering_retrieval_based_conversational_recommendation_with_contrasting_u/">Empowering Retrieval-based Conversational Recommendation with Contrasting User Preferences</a></strong></dt>
<dd>
<p>用 LLM 抽取并扩增 like/dislike 对比偏好，以三路相似度线性打分做稠密检索式对话推荐，INSPIRED 上 Recall@10 相对提升 99.72%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/enhancing_recommendation_explanations_through_user_centric_refinement/">Enhancing Recommendation Explanations through User-Centric Refinement</a></strong></dt>
<dd>
<p>提出 RefineX 多智能体推理期精炼框架，以 plan-then-refine 与分层反思将 PETER 解释在事实性/个性化/连贯性三维度大幅提升。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/exploring_the_potential_of_llms_as_personalized_assistants_dataset_evaluati/">Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis</a></strong></dt>
<dd>
<p>提出 HiCUPID 个性化助手基准（1.5k 合成用户、17k token 长对话、单/多跳 QA），并蒸馏 Llama-3.2 评估器对齐人类偏好，揭示 SFT+DPO 最优而纯 DPO 崩溃。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/iagent_llm_agent_as_a_shield_between_user_and_recommender_systems/">iAgent: LLM Agent as a Shield between User and Recommender Systems</a></strong></dt>
<dd>
<p>提出 user-agent-platform 新范式，用个体指令感知 LLM Agent 在用户与平台间做重排护盾，四数据集平均提升 16.6%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/muse_a_multimodal_conversational_recommendation_dataset_with_scenario_groun/">Muse: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles</a></strong></dt>
<dd>
<p>首个多模态对话推荐数据集，用场景锚定用户画像的多智能体框架合成 7K 段图文对话，并验证三种 MLLM 的可学习性。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/personax_a_recommendation_agent_oriented_user_modeling_framework_for_long_b/">PersonaX: A Recommendation Agent Oriented User Modeling Framework for Long Behavior Sequence</a></strong></dt>
<dd>
<p>面向推荐智能体的离线多画像用户建模框架，用聚类+原型-多样性均衡采样压缩长行为序列，30-50%数据即超越全量基线并大幅降低在线延迟。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/q_strum_debate_query_driven_contrastive_summarization_for_recommendation_co/">Q-STRUM Debate: Query-Driven Contrastive Summarization for Recommendation Comparison</a></strong></dt>
<dd>
<p>用双人格辩论式提示替代单次摘要，为查询驱动的推荐对比摘要注入对比性与多样性，在酒店/餐厅/旅游三域 Win Rate 最高达 89%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/refining_text_generation_for_realistic_conversational_recommendation_via_di/">Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization</a></strong></dt>
<dd>
<p>用 DPO 对齐 LLM 生成的对话摘要与物品推荐信息，使评分器更准，Tabidachi 上 HR@1 提升 21.3%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/debias_fairness_industrial/stereotype_or_personalization_user_identity_biases_chatbot_recommendations/">Stereotype or Personalization? User Identity Biases Chatbot Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 聊天机器人无论用户显式还是隐式透露种族身份都会产生刻板推荐，且在被追问时系统性掩盖种族影响，跨 7 个主流模型、4 个种族、12 万条合成请求验证。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/agent_conversational_rec/towards_personalized_conversational_sales_agents_contextual_user_profiling_/">Towards Personalized Conversational Sales Agents: Contextual User Profiling for Strategic Action</a></strong></dt>
<dd>
<p>提出 CSALES 对话销售任务与 CSUSER 真实画像用户模拟器，CSI 智能体通过上下文画像驱动统一动作空间，SR 提升约 15-20pt、SWR 提升约 5-10pt。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/poi_spatiotemporal_rec/triptailor_a_real_world_benchmark_for_personalized_travel_planning/">TripTailor: A Real-World Benchmark for Personalized Travel Planning</a></strong></dt>
<dd>
<p>构建50万真实POI与近4000条真人行程的个性化旅行规划基准，提出可行性/合理性/个性化三维评估与奖励模型对比框架，揭示SOTA LLM行程超越真人比例不足10%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_un/">Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents</a></strong></dt>
<dd>
<p>将决策解耦为文档客观打分矩阵与贝叶斯偏好推断，用决策感知信息增益主动提问，Top-1 准确率最高提升约 20%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/from_recall_to_forgetting_benchmarking_long_term_memory_for_personalized_ag/">From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents</a></strong></dt>
<dd>
<p>提出 Memora 长周期记忆基准与 FAMA 遗忘感知指标，揭示 LLM 与记忆智能体在记忆突变下频繁复用失效记忆。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/harpo_hierarchical_agentic_reasoning_for_user_aligned_conversational_recomm/">HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation</a></strong></dt>
<dd>
<p>将对话推荐重构为多维质量驱动的结构化决策，用分层偏好奖励引导树搜索推理，ReDial R@10 达 29.8%（+18.7%）。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/intent_driven_semantic_id_generation_for_grounded_conversational_news_recom/">Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation</a></strong></dt>
<dd>
<p>用 LLM 直接生成三层语义 ID 前缀再模糊匹配新闻池，结构上消除幻觉，冷启动 L1 达 18.0%，延迟 85ms。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/mirroring_users_towards_building_preference_aligned_user_simulator_with_use/">Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation</a></strong></dt>
<dd>
<p>用 LLM 生成 EKB 决策过程作澄清、按认知不确定性蒸馏并拒绝采样去噪，微调 3B 模拟器在 8 域上超越 GPT-5 与教师模型。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/re2a_situated_conversational_recommendation_via_rubric_based_preference_rea/">Re2A: Situated Conversational Recommendation via Rubric-based Preference Reasoning and Alignment</a></strong></dt>
<dd>
<p>将情境对话推荐重构为“先推理后对齐”两阶段，用自动评分标准引导偏好状态推理并以异构负例 DPO 实现用户-场景双重对齐，SIMMC 2.1 上 Hit@1 提升 8.17pt。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/rectoolbench_benchmarking_recommendation_specific_tool_orchestration_under_/">RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent</a></strong></dt>
<dd>
<p>首个基于 MCP 的模糊意图推荐工具编排基准，1288 条四级难度可执行任务，揭示语法合法≠推荐成功的执行-结果鸿沟。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/rerec_reasoning_augmented_llm_based_recommendation_assistant_via_reinforcem/">ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning</a></strong></dt>
<dd>
<p>用双图奖励塑形、推理感知优势估计与在线课程调度改造 RFT，使 3B LLM 在复杂查询推荐上大幅超越 GRPO 系基线并保留通用能力。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/towards_effective_structured_context_modeling_for_conversational_recommende/">Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search</a></strong></dt>
<dd>
<p>将对话上下文建模为双节点 MCTS 树，ELNode 搜索追问动作、EXNode 精炼检索查询，Redial/OpenDialKG 上 SR 平均提升 9.07%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/agent_conversational_rec/where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conve/">Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 SiPeR 框架，用场景转移估计解决“去哪”与贝叶斯逆向推理解决“要什么”，在 SIMMC 2.1 与 SCREEN 上 R@1 平均提升约 10.9%/10.6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/agent_conversational_rec/starec_an_efficient_agent_framework_for_recommender_systems_via_autonomous_/">STARec: An Efficient Agent Framework for Recommender Systems via Autonomous Deliberate Reasoning</a></strong></dt>
<dd>
<p>将每位用户建模为具备快/慢双认知的 LLM Agent，用 SFT 蒸馏 + GRPO 排序奖励两阶段训练，仅用 0.4% 数据即在 ML-1M/CDs 上超越全量训练基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/agent_conversational_rec/step_stepwise_curriculum_learning_for_context_knowledge_fusion_in_conversat/">STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation</a></strong></dt>
<dd>
<p>用三阶段课程学习驱动 F-Former 对齐对话与知识图谱，再以双前缀提示注入冻结 PLM，ReDial R@1 提升 15.7%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/agent_conversational_rec/temporal_aware_user_behaviour_simulation_with_large_language_models_for_rec/">Temporal-Aware User Behaviour Simulation with Large Language Models for Recommender Systems</a></strong></dt>
<dd>
<p>用动态画像更新、时序模式抽取与自适应聚合构建 LLM 用户模拟器，在 ML-1M 上 nDCG@5 相对 RecAgent 提升约 27.8%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/a_behavioral_trait_leaks_into_preferences_diagnosing_trait_interference_in_/">A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators</a></strong></dt>
<dd>
<p>揭示 LLM 用户模拟器中活跃度特质干扰偏好边界并导致评估失效，提出页面级质量锚点 PQA 恢复特质独立性。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/ask_to_be_sure_informative_interactions_for_confident_multi_turn_llm_recomm/">Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation</a></strong></dt>
<dd>
<p>以推荐列表采样熵的下降量作为信息增益奖励，免真值微调 LLM 多轮对话策略，INSPIRED/ReDial 上 Hit 与对话效率双升。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/from_counterfactual_edits_to_human_narratives_llm_driven_interpretations_fo/">From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations</a></strong></dt>
<dd>
<p>用 LLM 将冗长反事实编辑列表摘要为高层偏好叙事，并以人类校准指标评估正确性/信息量/语言质量。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/on_mitigating_data_sparsity_in_conversational_recommender_systems/">On Mitigating Data Sparsity in Conversational Recommender Systems</a></strong></dt>
<dd>
<p>DACRS 用 LLM 两阶段对话增强 + KG 实体替换与相似度约束 + 对话引导注意力融合，缓解对话与实体双重稀疏，ReDial R@50 提升 11.46%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/retrieval_scoring_and_decoding_shape_performance_and_stability_in_llm_based/">Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation</a></strong></dt>
<dd>
<p>在 ReDial 上系统解耦候选池规模、一阶段检索器与解码温度对 LLM 重排器的影响，揭示协议选择可移动 NDCG@10 超过模型差距。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2026/agent_conversational_rec/curr_rlcer_curriculum_reinforcement_learning_for_coherence_explainable_reco/">Curr-RLCER: Curriculum Reinforcement Learning For Coherence Explainable Recommendation</a></strong></dt>
<dd>
<p>用三阶段课程强化学习（CTR→评分→解释生成）训练 LLM，并设计一致性奖励与评估方案，使解释与评分双向对齐。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/agent_conversational_rec/improving_conversational_recommendation_with_contextual_adaptation_of_exter/">Improving Conversational Recommendation with Contextual Adaptation of External Recommenders and LLM-based Reranking</a></strong></dt>
<dd>
<p>用轻量实体级序列推荐器产出候选、LLM 做上下文适配重排，ReDial/INSPIRED 上 H@5 平均提升 54%/25%。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/agent_conversational_rec/interplay_training_independent_simulators_for_reference_free_conversational/">Interplay: Training Independent Simulators for Reference-Free Conversational Recommendation</a></strong></dt>
<dd>
<p>训练独立用户/推荐双模拟器，以目标属性替代目标物品消除 oracle 泄漏，8B 模型 Recall@1 超 32B 基线 15 倍。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/agent_conversational_rec/usersimcrs_v2_simulation_based_evaluation_for_conversational_recommender_sy/">UserSimCRS v2: Simulation-Based Evaluation for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>升级对话推荐用户模拟评测工具包，引入 LLM 模拟器、统一数据格式与 LLM-as-a-Judge 指标，案例揭示模拟器间系统排序显著分歧。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/agent_conversational_rec/bi_nas_towards_effective_and_personalized_explanation_for_recommender_syste/">Bi-NAS: Towards Effective and Personalized Explanation for Recommender Systems via Bi-Level Neural Architecture Search</a></strong></dt>
<dd>
<p>用双层 NAS 自动搜索交叉注意力与特征交互算子，再以零样本 LLM 提示生成个性化解释，四数据集上推荐与解释指标双优。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/multimodal_rec/fuse_failure_aware_usage_of_subagent_evidence_for_multimodal_search_and_rec/">FUSE: Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation</a></strong></dt>
<dd>
<p>用紧凑 GDR 替代原始图像提示，系统对比 7 种上下文预算策略并以五阶段归因层定位失败，Context Compression 在意图/路由/召回/NDCG@5 全面最优且 p95 延迟降 45%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/agent_conversational_rec/persopilot_an_adaptive_ai_copilot_for_transparent_contextualized_persona_cl/">PersoPilot: An Adaptive AI-Copilot for Transparent Contextualized Persona Classification and Personalized Response Generation</a></strong></dt>
<dd>
<p>双模式智能体 Copilot：PersoAgent 做上下文人格抽取与社区推荐，Analytic Tools 用 Phi-4-mini+TF-IDF 主动学习闭环做可解释人格分类，UniEval 四维均分 0.84–1.0。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/agent_conversational_rec/reasoning_over_user_preferences_knowledge_graph_augmented_llms_for_explaina/">Reasoning over User Preferences: Knowledge Graph-Augmented LLMs for Explainable Conversational Recommendations</a></strong></dt>
<dd>
<p>COMPASS 用图实体描述预训练+知识感知指令微调，让 LLM 生成可解释文本偏好摘要并以门控注入现有 CRS，ReDial/INSPIRED 上 HR@10 最高提升 44.76%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/debias_fairness_industrial/an_efficient_and_effective_agentic_group_shilling_attack_on_recommender_sys/">An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems</a></strong></dt>
<dd>
<p>提出协调者驱动的角色切换多智能体群体托攻击框架 AGAS，黑盒免训练、动态策略，HR@10 相对提升 155%–198% 且显著削弱检测器。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/agent_conversational_rec/agentrecbench_benchmarking_llm_agent_based_personalized_recommender_systems/">AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems</a></strong></dt>
<dd>
<p>首个 LLM 智能体推荐系统基准，构建 U-R-I 文本交互环境与三类场景，揭示结构化工作流智能体显著优于传统与朴素 Agent。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/context_steering_controllable_personalization_at_inference_time/">Context Steering: Controllable Personalization at Inference Time</a></strong></dt>
<dd>
<p>免训练解码期个性化：用有/无上下文两次前向的似然差线性缩放上下文影响力，实现可控个性化推荐与贝叶斯意图反演。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/agent_conversational_rec/enhancing_personalized_multi_turn_dialogue_with_curiosity_reward/">Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward</a></strong></dt>
<dd>
<p>将用户模型信念增益作为 PBRS 内在奖励注入多轮 RLHF，使 LLM 主动探询用户属性，习题推荐成功率 54%→87.5%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/agent_conversational_rec/agentselect_benchmark_for_narrative_query_to_agent_recommendation/">AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation</a></strong></dt>
<dd>
<p>将异构评测产物统一为叙事查询到可部署智能体配置的推荐基准，揭示长尾一次性监督下内容感知匹配的必要性。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/agent_conversational_rec/propersim_developing_proactive_and_personalized_ai_assistants_through_user_/">ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation</a></strong></dt>
<dd>
<p>提出 ProPerSim 仿真任务与 ProPerAssistant 基线，用 LLM 用户智能体+四维评分准则+DPO 偏好学习，让助手在时机与内容上同时个性化，14 天内平均满意度从 2.2 升至 3.3/4。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/agent_conversational_rec/siturecbench_evaluating_frontier_models_as_situated_recommendation_assistan/">SituRecBench: Evaluating Frontier Models as Situated Recommendation Assistants in 3D Environments</a></strong></dt>
<dd>
<p>构建 UE5 三维购物环境与共享第一人称视角的交互式基准，统一评测 VLM 助手的对话质量、推荐效用与导航有效性。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/agent_conversational_rec/lavic_adapting_large_vision_language_models_to_visually_aware_conversationa/">LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation</a></strong></dt>
<dd>
<p>用视觉知识自蒸馏把商品图压成5个[CLS] token，再做推荐提示微调，在新建Reddit-Amazon三域上HR@1最高提升54.2%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/agent_conversational_rec/leveraging_the_power_of_conversations_optimal_key_term_selection_in_convers/">Leveraging the Power of Conversations: Optimal Key Term Selection in Conversational Contextual Bandits</a></strong></dt>
<dd>
<p>用平滑关键词语境与最小特征值自适应触发对话，将对话式上下文 Bandit 遗憾降至近极小极大最优 O(√(dT logT))，累积遗憾较基线降低至少 14.6%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/sequential_rec/why_not_together_a_multiple_round_recommender_system_for_queries_and_items/">Why Not Together? A Multiple-Round Recommender System for Queries and Items</a></strong></dt>
<dd>
<p>提出 MAGUS 多轮查询-物品联合推荐框架，以词组合关系图桥接查询与物品，用标签传播在线吸收反馈，可插拔接入 12 种基座模型。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/agent_conversational_rec/hierarchical_long_term_semantic_memory_for_linkedin_s_hiring_agent/">Hierarchical Long-Term Semantic Memory for LinkedIn&#x27;s Hiring Agent</a></strong></dt>
<dd>
<p>HLTM 用 schema 对齐的层级记忆树 + 多视图表示 + 无损增量索引，为 LinkedIn 招聘智能体实现低延迟、隐私隔离的长期个性化记忆，正确率提升 5%+、检索 F1 提升 10%+。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/hoba_hierarchical_on_policy_bidding_agents_for_adaptive_online_advertising/">HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising</a></strong></dt>
<dd>
<p>三层时间尺度解耦出价：LLM 小时级推理超参、SARSA 分钟级因果去偏选专家、专家池逐次拍卖执行，AuctionNet 与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/mgfrec_towards_reinforced_reasoning_recommendation_with_multiple_groundings/">MGFRec: Towards Reinforced Reasoning Recommendation with Multiple Groundings and Feedback</a></strong></dt>
<dd>
<p>将单步 grounding 扩展为多轮物品空间接地并引入用户 Agent 反馈，用 GRPO 训练 LLM 推理推荐，Full-Ranking 下 H@5 最高提升约 8.8%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/agent_conversational_rec/safecrs_personalized_safety_alignment_for_llm_based_conversational_recommen/">SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出 SafeRec 个性化安全基准与 Safe-SFT+Safe-GDPO 两阶段框架，在 SafeMovie 上将安全违规率相对降低 96.5% 且保持竞争力推荐质量。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/social_group_rec/consistent_explainers_or_unreliable_narrators_understanding_llm_generated_g/">Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations</a></strong></dt>
<dd>
<p>用社会选择聚合策略作多基线对4个LLM的群组推荐与解释做一致性审计，发现推荐近似ADD、解释常称平均但夹带未定义流行度/相似度/多样性。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/agent_conversational_rec/failure_prediction_in_conversational_recommendation_systems/">Failure Prediction in Conversational Recommendation Systems</a></strong></dt>
<dd>
<p>提出监督式对话性能预测任务，用AE与L1收缩多轮预测器区分系统失败与目录失败，揭示目录缺失场景下预测精度显著下降。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/image_seeking_intent_prediction_for_cross_device_product_search/">Image-Seeking Intent Prediction for Cross-Device Product Search</a></strong></dt>
<dd>
<p>用语音查询+商品摘要预测图像意图，以可微精度损失抑制误触发，离线 Precision 达 78.5。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/agent_conversational_rec/usb_rec_an_effective_framework_for_improving_conversational_recommendation_/">USB-Rec: An Effective Framework for Improving Conversational Recommendation Capability of Large Language Model</a></strong></dt>
<dd>
<p>用带标签用户模拟器自动构造偏好对做 RL 训练，再在推理期用内部用户模拟器+树搜索自增强，iEval 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/rec_a_verifiable_benchmark_for_agentic_recommender_systems/">$τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems</a></strong></dt>
<dd>
<p>用可验证目录谓词奖励、reveal-tagged 信息揭示机制与 pass^k 可靠性指标替代 LLM-as-judge，揭示智能体推荐在隐藏约束下的可靠性悬崖。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/a_position_paper_on_recommender_systems_in_the_era_of_autonomous_agents/">A Position Paper on Recommender Systems in the Era of Autonomous Agents</a></strong></dt>
<dd>
<p>提出人-智能体-平台三方混合推荐生态，将智能体视为用户侧独立代理，重构任务定义、评估范式与四大研究方向。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/attacking_and_defending_multi_agent_collaborative_filtering_systems_through/">Attacking and Defending Multi-Agent Collaborative Filtering Systems Through Connectivity</a></strong></dt>
<dd>
<p>在 AgentCF 上系统复现 6 类攻击与 4 类防御，揭示候选数 k 与目录集中度 ρ 对用户/物品分区的非对称、非单调调控规律。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/bootstrapping_conversational_recommendation_agents_at_spotify_synthetic_dat/">Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops</a></strong></dt>
<dd>
<p>用多轮合成对话+方差归因自改进循环冷启动对话推荐智能体，线上A/B提升14%收听与5%周活。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/conversational_recommendation_over_live_e_commerce_catalogues_with_self_ref/">Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval</a></strong></dt>
<dd>
<p>面向实时电商商品库的自刷新检索式对话购物助手，以双哈希增量同步替代全量重建，同步开销降至全量 1.8%–12.3%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/inverse_theory_of_mind_modeling_for_content_recommendation_from_web_browsin/">Inverse Theory of Mind Modeling for Content Recommendation: From Web Browsing to Dynamic Intelligent Interfaces</a></strong></dt>
<dd>
<p>用 LLM 从浏览轨迹反向推断信念并合成多假设人格画像，在 OPeRA 上超越真实人格基线并驱动 VisionOS 空间界面。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/llm_as_a_judge_for_evaluating_system_responses_in_conversational_music_reco/">LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation</a></strong></dt>
<dd>
<p>首个以 400 条专家标注实证 LLM-as-a-Judge 在对话音乐推荐响应评估中与人类判断的对齐度，并给出条件化与规模选择指南。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/melo_a_production_llm_powered_music_recommendation_agent/">Melo: A Production LLM-Powered Music Recommendation Agent</a></strong></dt>
<dd>
<p>网易云音乐生产级 LLM 音乐推荐 Agent，以五节点确定性状态图 + 推理时实体接地 + 反思式重试解决实体幻觉与长尾塌缩，线上 A/B 留存提升超 2pp。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/shape_your_feed_an_llm_based_agentic_system_for_conversational_recommendati/">Shape Your Feed: An LLM-based Agentic System for Conversational Recommendation</a></strong></dt>
<dd>
<p>三层 Agentic 架构（感知/服务/自进化）实现多模态实时用户可控重排，离线对齐打分 98.85% 准确率，线上 dislike 降 2.74%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/the_utility_of_llms_in_recommender_systems_explanation_evaluation/">The Utility of LLMs in Recommender Systems Explanation Evaluation</a></strong></dt>
<dd>
<p>系统评测 14 个 LLM 作为推荐解释评审员与 216 名人类评分的一致性，发现排序相关中等但绝对评分一致性低，并给出四条实用准则。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/agentcf_memory_enhanced_llm_based_agents_for_popularity_aware_cross_domain_/">AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations</a></strong></dt>
<dd>
<p>以双层记忆+两步融合+兴趣组共享记忆，令 LLM 用户智能体在跨域场景显式建模流行度影响，采样排序 MRR 平均超 AgentCF 14.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/agentic_feedback_loop_modeling_improves_recommendation_and_user_simulation/">Agentic Feedback Loop Modeling Improves Recommendation and User Simulation</a></strong></dt>
<dd>
<p>用记忆驱动的推荐智能体-用户智能体多轮反馈闭环，同时提升下一物品推荐与用户行为模拟，并验证不放大流行度/位置偏差。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/mscrs_multi_modal_semantic_graph_prompt_learning_framework_for_conversation/">MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>构建协同/文本/图像三模态语义图并以共享嵌入融合，图提示注入 DialoGPT，ReDial 与 INSPIRED 全指标 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/pscon_product_search_through_conversations/">PSCon: Product Search Through Conversations</a></strong></dt>
<dd>
<p>提出人-人教练式采集协议构建双市场双语真实 CPS 数据集 PSCon，定义六子任务流水线并给出 Transformer 统一基准模型。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/pub_an_llm_enhanced_personality_driven_user_behaviour_simulator_for_recomme/">PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation</a></strong></dt>
<dd>
<p>用大五人格驱动 LLM Agent 生成合成交互日志，Jaccard 相似度 0.31 逼近真实序列，并复现 7 种推荐算法的 nDCG 排序趋势。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/search_based_interaction_for_conversation_recommendation_via_generative_rew/">Search-Based Interaction For Conversation Recommendation via Generative Reward Model Based Simulated User</a></strong></dt>
<dd>
<p>用生成式奖励模型构建无标签模拟用户，通过属性评论与生成式打分驱动束搜索交互，INSPIRED 上 Recall@10 相对最优基线提升 83%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/simulating_before_planning_constructing_intrinsic_user_world_model_for_user/">Simulating Before Planning: Constructing Intrinsic User World Model for User-Tailored Dialogue Policy Planning</a></strong></dt>
<dd>
<p>用扩散模型刻画用户画像、布朗桥预判用户反馈，构建内在用户世界模型以定制对话策略，P4G/ESConv 上 SR 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/llm_rec/synthtrips_a_knowledge_grounded_framework_for_benchmark_query_generation_fo/">SynthTRIPs: A Knowledge-Grounded Framework for Benchmark Query Generation for Personalized Tourism Recommenders</a></strong></dt>
<dd>
<p>用 PersonaHub 人格 + 可持续性过滤器 + KB 接地，让 LLM 批量合成个性化城市游查询，4,604 条查询经 J-LLM 与双专家双轨验证。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/a_standardized_re_evaluation_of_conversational_recommender_systems_on_the_r/">A Standardized Re-evaluation of Conversational Recommender Systems on the ReDial Dataset</a></strong></dt>
<dd>
<p>统一复现 7 个 CRS 模型，揭示 Recall@1 复现粒度鸿沟、约 50% 精度来自重复捷径、增益多源于 LLM 骨干而非架构创新。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/agentic_multi_source_grounding_for_enhanced_query_intent_understanding_a_do/">Agentic Multi-Source Grounding for Enhanced Query Intent Understanding: A DoorDash Case Study</a></strong></dt>
<dd>
<p>用目录实体检索+Agentic 网页搜索双源 grounding 替代 winner-takes-all 分类，输出有序双意图并由可插拔消歧层裁决，长尾意图准确率 +13.0pp。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/bridging_passive_and_active_enhancing_conversation_starter_recommendation_v/">Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling</a></strong></dt>
<dd>
<p>用对抗分布对齐+RQ-VAE语义离散化把用户主动查询引入对话开场白推荐，打破曝光-点击回音室，线上活跃天数+0.04%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/conv_finre_a_conversational_and_longitudinal_benchmark_for_utility_grounded/">Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation</a></strong></dt>
<dd>
<p>用逆优化从纵向决策轨迹反推用户风险偏好，构建四视图多视角金融推荐基准，揭示 LLM 效用对齐与行为模仿的根本张力。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/genfacet_end_to_end_generative_faceted_search_via_multi_task_preference_ali/">GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment in E-Commerce</a></strong></dt>
<dd>
<p>将电商分面搜索重构为分面生成与查询改写两个耦合生成任务，经蒸馏+SFT+GRPO 对齐搜索效用，线上分面 CTR 相对提升 42.0%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/meituan_merchant_business_diagnosis_via_policy_guided_dual_process_user_sim/">Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation</a></strong></dt>
<dd>
<p>从行为轨迹挖掘决策策略作为共享对齐层，锚定 LLM 推理与 GBDT 拟合双过程分支，群体仿真误差降至 8.80%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/recquest_towards_estimating_user_domain_knowledge_in_conversational_recomme/">RecQuest: Towards Estimating User Domain Knowledge in Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出游戏化数据采集协议 RecQuest 与知识估计新任务，用客观问卷替代自评建立 ground truth，并给出 LLM 零/少样本与微调基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/task_aware_automated_user_profile_generation_for_recommendation_simulation_/">Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models</a></strong></dt>
<dd>
<p>提出免训练三阶段画像生成框架 APG4RecSim，用语义合并+反事实因果映射将交互历史转为任务对齐画像，在判别/排序/评分三任务上取得最佳综合表现。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/trace_a_conversational_framework_for_sustainable_tourism_recommendation_wit/">TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations</a></strong></dt>
<dd>
<p>多智能体 LLM 框架以澄清提问+反事实解释实现可持续旅游推荐的非强制绿色引导，24 人用户研究验证。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 分层 MoE 架构，显式建模“何时问、怎么问”的阶段感知偏好引导策略，Recall@10 相对最优基线提升 14.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR-AP/2025/agent_conversational_rec/limitations_of_current_evaluation_practices_for_conversational_recommender_/">Limitations of Current Evaluation Practices for Conversational Recommender Systems and the Potential of User Simulation</a></strong></dt>
<dd>
<p>批判静态数据集与 Recall@N 评估失真，提出奖励/成本效用框架与 RDL 指标，并用 LLM 用户模拟器替代真人评估。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/llm_rec/reason4rec_deliberative_user_preference_alignment_of_large_language_models_/">Reason4Rec: Deliberative User Preference Alignment of Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>提出审慎推荐任务与三专家多步推理框架，用评论监督偏好蒸馏-匹配-预测，评分预测与排序全面领先。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/agent_conversational_rec/recommender_ai_agent_integrating_large_language_models_for_interactive_reco/">Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations</a></strong></dt>
<dd>
<p>以 LLM 为大脑、传统推荐模型为工具，配合候选总线、长短期画像、计划优先执行与反思机制，在 Steam/MovieLens/Beauty 上显著超越 GPT-4 与 Chat-Rec。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/agent_conversational_rec/user_behavior_simulation_with_large_language_model_based_agents/">User Behavior Simulation with Large Language Model based Agents</a></strong></dt>
<dd>
<p>提出 LLM Agent 用户行为模拟器 RecAgent，以画像-记忆-动作三模块与沙盒环境复现信息茧房与从众现象，行为可信度逼近真人。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/agent_conversational_rec/vague_preference_policy_learning_for_conversational_recommendation/">Vague Preference Policy Learning for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 VPMCR 模糊偏好场景与 VPPL 框架，用软估计替代二值过滤，SR@15 最高提升 22%，平均轮次显著下降。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/sequential_rec/a_survey_on_intent_aware_recommender_systems/">A Survey on Intent-aware Recommender Systems</a></strong></dt>
<dd>
<p>系统梳理意图感知推荐(IARS)的三大技术路线与评估实践，指出离线精度导向、可复现性不足与显式意图建模缺失三大空白。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/agent_conversational_rec/impact_of_tone_aware_explanations_in_recommender_systems/">Impact of Tone-Aware Explanations in Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 生成六种语气解释做在线成对用户研究，序数逻辑回归揭示语气对 10 项感知指标、领域与人格年龄的差异化影响。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/agent_conversational_rec/topic_centric_explanations_for_news_recommendation/">Topic-Centric Explanations for News Recommendation</a></strong></dt>
<dd>
<p>用双层注意力主题模型 BATM 同时做新闻推荐与主题级解释，MIND 全库排序 AUC 提升约 0.4-0.5 个点，并用 NPMI/W2V 量化主题可解释性。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/agent_conversational_rec/collab_rec_an_llm_based_agentic_framework_for_balancing_recommendations_in_/">Collab-Rec: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></strong></dt>
<dd>
<p>三角色 LLM 智能体 + 确定性 Moderator 多轮约束精炼，在 900 条旅游查询上兼顾约束满足与长尾多样性。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/agent_conversational_rec/crs_arena_crowdsourced_benchmarking_of_conversational_recommender_systems/">CRS Arena: Crowdsourced Benchmarking of Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出基于成对对战与真实用户反馈的众包 CRS 评测平台，发布 474 段对话数据集与 Elo 榜单，揭示 R@10 与用户满意度负相关。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/agent_conversational_rec/reindex_then_adapt_improving_large_language_models_for_conversational_recom/">Reindex-Then-Adapt: Improving Large Language Models for Conversational Recommendation</a></strong></dt>
<dd>
<p>将多 token 电影标题重索引为单 token，再用偏置/门控适配目标平台分布，Llama2-7b 上 H@10 相对提升 59.37%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/llm_rec/towards_next_generation_recommender_systems_a_benchmark_for_personalized_re/">Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs</a></strong></dt>
<dd>
<p>构建 RecBench+ 约3万条复杂用户查询基准，覆盖显式/隐式/误导条件与画像偏好，系统评测7个LLM并揭示SFT+RFT两阶段微调最优。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/agent_conversational_rec/a_llm_based_controllable_scalable_human_involved_user_simulator_framework_f/">A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>以插件管理器分阶段管控 LLM 用户模拟器，通过记忆隔离与偏好二分抑制数据泄漏，在 ReDial/OpenDialKG/MovieLens 上显著提升 CRS 评估可信度。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/agent_conversational_rec/agentsociety_challenge_designing_llm_agents_for_user_modeling_and_recommend/">AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms</a></strong></dt>
<dd>
<p>首个 WWW LLM Agent 双赛道竞赛：Yelp/Amazon/Goodreads 三源数据 + 交互式模拟器，295 队参赛，Track1/2 最终阶段提升 9.1%/15.9%，并验证模拟 groundtruth 与真实分布强相关。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/beyond_retrieval_generating_narratives_in_conversational_recommender_system/">Beyond Retrieval: Generating Narratives in Conversational Recommender Systems</a></strong></dt>
<dd>
<p>构建 REGEN 叙事数据集并提出 CF+内容嵌入经 Adapter 注入 LLM 的融合架构，在 Office/Clothing 两域上 BLEU/ROUGE/语义相似度较单模态提升 4–12%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/agent_conversational_rec/bridging_conversational_and_collaborative_signals_for_conversational_recomm/">Bridging Conversational and Collaborative Signals for Conversational Recommendation</a></strong></dt>
<dd>
<p>构建 Reddit-ML32M 跨源数据集，用 SASRec 协同嵌入对 GPT-3.5 零样本推荐做相似度重排，H@5 提升 12.32%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/agent_conversational_rec/collaborative_retrieval_for_large_language_model_based_conversational_recom/">Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems</a></strong></dt>
<dd>
<p>用 EASE 协同检索 + 两步 LLM 反思重排为黑盒 LLM 注入 CF 知识，Reddit-v2/Redial 全库排序 Recall@20 显著超越 Zero-shot LLM。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/agent_conversational_rec/disentangling_likes_and_dislikes_in_personalized_generative_explainable_rec/">Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation</a></strong></dt>
<dd>
<p>用 LLM 重构正负观点分离的可解释推荐数据集，提出情感匹配与正负内容相似度指标，并证明将预测评分作为输入可显著提升情感感知解释生成。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/filtering_discomforting_recommendations_with_large_language_models/">Filtering Discomforting Recommendations with Large Language Models</a></strong></dt>
<dd>
<p>用可编辑偏好画像+对话式规则生成，以插件方式过滤主观不适推荐；离线代理任务与24人一周用户研究验证有效性。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/leveraging_llms_for_influence_path_planning_in_proactive_recommendation/">Leveraging LLMs for Influence Path Planning in Proactive Recommendation</a></strong></dt>
<dd>
<p>用提示工程驱动 LLM 零样本规划主动推荐影响路径，配合 LLM 用户模拟器与 SASRec 模拟器双轨评测，显著提升路径连贯性与用户接受度。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/agent_conversational_rec/recusersim_a_realistic_and_diverse_user_simulator_for_evaluating_conversati/">RecUserSim: A Realistic and Diverse User Simulator for Evaluating Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出四模块 LLM Agent 用户模拟器，以有限理性三层 Rating-Action-Response 机制实现真实角色扮演、多样人群覆盖与显式打分，并落地华为 Celia 餐饮助手。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/agentdr_dynamic_recommendation_with_implicit_item_item_relations_via_llm_ba/">AgentDR: Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents</a></strong></dt>
<dd>
<p>将全库排序委托给传统推荐工具，用 LLM Agent 做个性化工具加权与替代/互补关系重排，全排序平均提升约两倍。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/agenticshop_benchmarking_agentic_product_curation_for_personalized_web_shop/">AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping</a></strong></dt>
<dd>
<p>首个开放网络个性化商品策展基准，用真实购买史构建叙事画像与六维清单，LLM-as-a-judge 可验证评估，揭示现有智能体策展得分仅 15–38%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/multi_agent_collaborative_filtering_orchestrating_users_and_items_for_agent/">Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations</a></strong></dt>
<dd>
<p>将相似用户与相关物品实例化为LLM智能体，由中心编排器动态招募并下发个性化指令，多轮讨论聚合协同信号，三域HR@10最高提升17.9%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/recoworld_building_simulated_environments_for_agentic_recommender_systems/">RecoWorld: Building Simulated Environments for Agentic Recommender Systems</a></strong></dt>
<dd>
<p>提出双视角LLM仿真环境，用户模拟器在多轮交互中发出反思指令，驱动指令遵循推荐器优化留存，并给出三种历史建模方案与多智能体群体仿真蓝图。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/unveiling_and_simulating_short_video_addiction_behaviors_via_economic_addic/">Unveiling and Simulating Short-Video Addiction Behaviors via Economic Addiction Theory</a></strong></dt>
<dd>
<p>用经济学成瘾理论量化短视频成瘾曲线，并以 M2A+GRPO 训练 LLM 用户模拟器 AddictSim，会话级 MAE 较 PPO 降 11%。</p>
</dd>
</dl>

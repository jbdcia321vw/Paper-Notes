# 🗣️ 对话推荐与智能体仿真 (Conversational & Agentic Recommendation)

> **研究范畴**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

当前方向共收录 **60** 篇顶会/顶刊论文解读：

<dl>
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
<dt><strong><a href="../../venues/ICDM/2026/debias_fairness_industrial/an_efficient_and_effective_agentic_group_shilling_attack_on_recommender_sys/">An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems</a></strong></dt>
<dd>
<p>提出协调者驱动的角色切换多智能体群体托攻击框架 AGAS，黑盒免训练、动态策略，HR@10 相对提升 155%–198% 且显著削弱检测器。</p>
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
<dt><strong><a href="../../venues/TKDE/2026/llm_rec/reason4rec_deliberative_user_preference_alignment_of_large_language_models_/">Reason4Rec: Deliberative User Preference Alignment of Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>提出审慎推荐任务与三专家多步推理框架，用评论监督偏好蒸馏-匹配-预测，评分预测与排序全面领先。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/agent_conversational_rec/collab_rec_an_llm_based_agentic_framework_for_balancing_recommendations_in_/">Collab-Rec: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></strong></dt>
<dd>
<p>三角色 LLM 智能体 + 确定性 Moderator 多轮约束精炼，在 900 条旅游查询上兼顾约束满足与长尾多样性。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/llm_rec/towards_next_generation_recommender_systems_a_benchmark_for_personalized_re/">Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs</a></strong></dt>
<dd>
<p>构建 RecBench+ 约3万条复杂用户查询基准，覆盖显式/隐式/误导条件与画像偏好，系统评测7个LLM并揭示SFT+RFT两阶段微调最优。</p>
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

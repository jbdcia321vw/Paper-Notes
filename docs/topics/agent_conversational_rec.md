# 🗣️ 对话推荐与智能体仿真 (Conversational & Agentic Recommendation)

> **研究范畴**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

当前方向共收录 **29** 篇顶会/顶刊论文解读：

<dl>
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
<dt><strong><a href="../../venues/KDD/2026/agent_conversational_rec/hierarchical_long_term_semantic_memory_for_linkedin_s_hiring_agent/">Hierarchical Long-Term Semantic Memory for LinkedIn&#x27;s Hiring Agent</a></strong></dt>
<dd>
<p>HLTM 用 schema 对齐的层级记忆树 + 多视图表示 + 无损增量索引，为 LinkedIn 招聘智能体实现低延迟、隐私隔离的长期个性化记忆，正确率提升 5%+、检索 F1 提升 10%+。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/hoba_hierarchical_on_policy_bidding_agents_for_adaptive_online_advertising/">HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising</a></strong></dt>
<dd>
<p>三层时间尺度解耦出价：LLM 小时级推理超参、SARSA 分钟级因果去偏选专家、专家池逐次拍卖执行，AuctionNet 与线上 A/B 双验证。</p>
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
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/conversational_recommendation_over_live_e_commerce_catalogues_with_self_ref/">Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval</a></strong></dt>
<dd>
<p>面向实时电商商品库的自刷新检索式对话购物助手，以双哈希增量同步替代全量重建，同步开销降至全量 1.8%–12.3%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/llm_as_a_judge_for_evaluating_system_responses_in_conversational_music_reco/">LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation</a></strong></dt>
<dd>
<p>首个以 400 条专家标注实证 LLM-as-a-Judge 在对话音乐推荐响应评估中与人类判断的对齐度，并给出条件化与规模选择指南。</p>
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
<p>在 ReDial 上标准化复现 7 个 CRS 方法，揭示 Recall@1 粒度鸿沟、约 50% 准确率来自重复捷径，且增益多源于 LLM 骨干而非架构创新。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/bridging_passive_and_active_enhancing_conversation_starter_recommendation_v/">Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling</a></strong></dt>
<dd>
<p>用对抗分布对齐+RQ-VAE语义离散化把用户主动查询引入对话开场白推荐，打破曝光-点击回音室，线上活跃天数+0.04%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/conv_finre_a_conversational_and_longitudinal_benchmark_for_utility_grounded/">Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation</a></strong></dt>
<dd>
<p>用逆优化从纵向决策轨迹反推用户风险偏好，构建四视图多视角金融推荐基准，揭示 LLM 效用对齐与行为模仿的根本张力。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/from_top_1_to_top_k_a_reproducibility_study_and_benchmarking_of_counterfact/">From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems</a></strong></dt>
<dd>
<p>统一复现11种反事实解释方法，提出item/list双层与隐式/显式双格式评测协议，揭示效果-稀疏性权衡与图解释器可扩展性瓶颈</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/meituan_merchant_business_diagnosis_via_policy_guided_dual_process_user_sim/">Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation</a></strong></dt>
<dd>
<p>从行为轨迹挖掘决策策略作为共享对齐层，锚定 LLM 推理与 GBDT 拟合双过程分支，群体仿真误差降至 8.80%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/recgpt_mobile_on_device_large_language_models_for_user_intent_understanding/">RecGPT-Mobile: On-Device Large Language Models for User Intent Understanding in Taobao Feed Recommendation</a></strong></dt>
<dd>
<p>将 Qwen3-0.6B 量化后部署于手机端作为意图理解 Agent，把隐式行为翻译为显式搜索意图 query，配合自适应 Prompt 与意图漂移触发机制，线上平均 +1.8% CLICK / +2.7% PAY / +2.5% GMV。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/task_aware_automated_user_profile_generation_for_recommendation_simulation_/">Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models</a></strong></dt>
<dd>
<p>提出免训练三阶段画像生成框架 APG4RecSim，用语义合并+反事实因果映射将交互历史转为任务对齐画像，在判别/排序/评分三任务上取得最佳综合表现。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/trace_a_conversational_framework_for_sustainable_tourism_recommendation_wit/">TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations</a></strong></dt>
<dd>
<p>多智能体 LLM 对话推荐框架，用澄清提问+反事实解释对用户进行可持续旅游软性助推，24 人用户研究验证。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 分层 MoE 架构，显式建模“何时问、怎么问”的阶段感知偏好引导策略，Recall@10 相对最优基线提升 14.6%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/agent_conversational_rec/collab_rec_an_llm_based_agentic_framework_for_balancing_recommendations_in_/">Collab-Rec: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></strong></dt>
<dd>
<p>三角色 LLM 智能体 + 确定性 Moderator 多轮约束精炼，在 900 条旅游查询上兼顾约束满足与长尾多样性。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/investigating_the_effects_of_different_levels_of_user_control_in_an_interac/">Investigating the Effects of Different Levels of User Control in an Interactive Educational Recommender System</a></strong></dt>
<dd>
<p>在 MOOC 平台 CourseMapper 中实现输入/过程/输出三级用户控制的教育推荐系统，通过 N=184 被试间实验揭示输入控制是提升感知控制感的关键，并系统刻画控制感、透明度、信任、满意度与感知质量之间的结构关系。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/llm_rec/towards_next_generation_recommender_systems_a_benchmark_for_personalized_re/">Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs</a></strong></dt>
<dd>
<p>构建 RecBench+ 约3万条复杂用户查询基准，覆盖显式/隐式/误导条件与画像偏好，系统评测7个LLM并揭示SFT+RFT两阶段微调最优。</p>
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

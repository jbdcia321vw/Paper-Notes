# 🧠 大语言模型推荐 (LLM4Rec) (LLM-Enhanced & LLM-as-Recommender)

> **研究范畴**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

当前方向共收录 **86** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/a_behavioral_trait_leaks_into_preferences_diagnosing_trait_interference_in_/">A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators</a></strong></dt>
<dd>
<p>揭示 LLM 用户模拟器中活跃度特质干扰偏好边界并导致评估失效，提出页面级质量锚点 PQA 恢复特质独立性。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/a_dual_expert_strategy_integrating_llms_to_mitigate_negative_transfer_in_cr/">A Dual-Expert Strategy Integrating LLMs to Mitigate Negative Transfer in Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用域门控双专家+物品级注意力掩码+双池负采样对比学习，抑制 LLM 跨域序列推荐中的负迁移，10 域上超越 26 个基线，线上 CTR 相对提升 47.6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/ask_to_be_sure_informative_interactions_for_confident_multi_turn_llm_recomm/">Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation</a></strong></dt>
<dd>
<p>以推荐列表采样熵的下降量作为信息增益奖励，免真值微调 LLM 多轮对话策略，INSPIRED/ReDial 上 Hit 与对话效率双升。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cold_start_rec/castle_contrastive_and_seed_guided_training_for_cold_start_natural_language/">CASTLE: Contrastive and Seed-Guided Training for Cold-Start Natural Language Search</a></strong></dt>
<dd>
<p>用预订会话对比房源对+用户调研种子查询联合生成合成查询与构造式相关性标签，KL 1.01 对齐真实用户，9.2× 优于基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/debias_fairness_industrial/echotrace_diagnosing_recursive_risks_in_llm_powered_recommender_systems/">EchoTrace: Diagnosing Recursive Risks in LLM-Powered Recommender Systems</a></strong></dt>
<dd>
<p>提出角色感知三阶段诊断框架，用受控反馈回路仿真追踪 LLM 偏见与幻觉在 LLM4RS 中的递归放大与生态极化。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/from_counterfactual_edits_to_human_narratives_llm_driven_interpretations_fo/">From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations</a></strong></dt>
<dd>
<p>用 LLM 将冗长反事实编辑列表摘要为高层偏好叙事，并以人类校准指标评估正确性/信息量/语言质量。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/debias_fairness_industrial/llm_based_user_product_relevance_evaluation_for_dynamic_product_advertising/">LLM-based User-Product Relevance Evaluation for Dynamic Product Advertising</a></strong></dt>
<dd>
<p>用 SID 分组意图压缩 + 智能体指南蒸馏小模型，构建无查询的 DPA 召回离线相关性指标，与线上 CP/CR/CVR 强对齐。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/sequential_rec/lsf_sr_latent_semantic_fusion_for_sequential_recommendation_via_flow_based_/">LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders</a></strong></dt>
<dd>
<p>用归一化流增强的 CVAE 将 LLM 语义作为条件概率融合进 ID 协同嵌入，Full-Ranking 下 Recall@20 最高提升 12.98%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/profiling_what_matters_context_aware_item_profiles_from_large_scale_metadat/">Profiling What Matters: Context-Aware Item Profiles from Large-Scale Metadata for LLM Recommenders</a></strong></dt>
<dd>
<p>将海量异构物品元数据结构化并训练轻量选择器，为每个用户-物品对动态生成上下文相关画像，显著提升 LLM 重排效果。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/the_recall_ceiling_of_llm_recommendation_reranking/">The Recall Ceiling of LLM Recommendation Reranking</a></strong></dt>
<dd>
<p>证明闭候选重排器 NDCG 受自身窗口召回率硬上界约束，Oracle 协议高估 92–95%，九类 LLM 与监督重排器在真实检索下均无法显著超越 CF。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/think_to_personalize_unifying_reasoning_and_retrieval_for_user_centric_pers/">Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval</a></strong></dt>
<dd>
<p>用 SFT+GRPO 两阶段让 LLM 显式推理用户历史生成意图增强查询并统一编码，Full-Ranking 下 Broad/LongTail 召回显著提升，线上订单量 +0.46%</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/tracer_balancing_stability_plasticity_cognitivity_trilemma_for_llm_enhanced/">TRACER: Balancing Stability-Plasticity-Cognitivity Trilemma for LLM Enhanced Continual Recommendation</a></strong></dt>
<dd>
<p>揭示 LLM 增强持续推荐中的 SPC 三难困境，用 Procrustes 投影、置信门控凝聚与语义同步三模块协同破局，Full-Ranking 下 ACC 最高提升 14.38%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/training_free_llm_based_recommendation_with_post_llm_item_refinement_using_/">Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals</a></strong></dt>
<dd>
<p>提出 post-LLM 范式 CoRRe，用共购图方向细化与流行度幅度校准改造 LLM 物品表征，免训练即在三域 Amazon 上超越全部免训练基线并多数超越训练式方法。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/agentltv_an_agent_based_unified_search_and_evolution_framework_for_automate/">AgentLTV: An Agent-Based Unified Search-and-Evolution Framework for Automated Lifetime Value Prediction</a></strong></dt>
<dd>
<p>将 LTV 建模视为可执行流水线程序的搜索问题，用 LLM 多智能体驱动 MCTS 广探索 + 岛屿进化精炼，工业与公开数据上 ER 降 41%、Norm GINI 升 15%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/enriching_semantic_profiles_into_knowledge_graph_for_recommender_systems_us/">Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 为 KG 全实体生成语义画像，以可逆加性注入+成对匹配对齐，Full-Ranking 下 Recall@40 最高提升 2.6%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/agent_conversational_rec/hierarchical_long_term_semantic_memory_for_linkedin_s_hiring_agent/">Hierarchical Long-Term Semantic Memory for LinkedIn&#x27;s Hiring Agent</a></strong></dt>
<dd>
<p>HLTM 用 schema 对齐的层级记忆树 + 多视图表示 + 无损增量索引，为 LinkedIn 招聘智能体实现低延迟、隐私隔离的长期个性化记忆，正确率提升 5%+、检索 F1 提升 10%+。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/improving_item_discoverability_in_e_commerce_search_via_related_intent_gene/">Improving Item Discoverability in e-Commerce Search via Related Intent Generation</a></strong></dt>
<dd>
<p>用闭源LLM+蒸馏SLM生成替代/互补/主题三类隐式意图以扩展电商搜索召回，会话购买预测与LLM评审双轨评估，覆盖流量60%→80%、成本仅30%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/llm_as_a_judge_for_reliable_and_explainable_offline_evaluation_in_top_k_rec/">LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation</a></strong></dt>
<dd>
<p>用 LLM 从用户文本行为构建语义偏好代理替代 holdout ID 匹配，以 reasoning-then-scoring 生成可解释 Top-K 评测，模型排序一致性较常规测试提升 40–150%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/llm_rec/mixture_of_experts_knowledge_graph_retrieval_augmented_generation_for_multi/">Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation</a></strong></dt>
<dd>
<p>用 MoE 多粒度 KG 检索智能体按查询复杂度路由检索粒度，配合知识对齐与对比偏好优化，在 20 候选池协议下 Accuracy 相对提升 5.8%–20.4%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/reasoning_over_semantic_ids_enhances_generative_recommendation/">Reasoning over Semantic IDs Enhances Generative Recommendation</a></strong></dt>
<dd>
<p>通过教师模型合成富化语料强化 SID-语言对齐，再用 GRPO 结果驱动强化学习激发 SID 推理，三数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/multimodal_rec/text_guided_visual_representation_learning_for_robust_multimodal_e_commerce/">Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation</a></strong></dt>
<dd>
<p>以结构化元数据初始化语义查询、双流解耦视觉证据并用可靠性双门控校准，在 134K 全库 I2I 检索上 H@100 相对提升 6.04%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/uncertainty_aware_generative_recommendation/">Uncertainty-aware Generative Recommendation</a></strong></dt>
<dd>
<p>将生成置信度与样本难度转化为GRPO奖励与梯度权重，并显式对齐置信度token，稳定训练并支持风险感知推荐。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/a_position_paper_on_recommender_systems_in_the_era_of_autonomous_agents/">A Position Paper on Recommender Systems in the Era of Autonomous Agents</a></strong></dt>
<dd>
<p>提出人-智能体-平台三方混合推荐生态，将智能体视为用户侧独立代理，重构任务定义、评估范式与四大研究方向。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/social_group_rec/consensus_vs_dissent_dynamic_llm_modeling_of_subjective_preferences_in_grou/">Consensus vs. Dissent: Dynamic LLM Modeling of Subjective Preferences in Group Recommenders</a></strong></dt>
<dd>
<p>用人类问卷+DeepSeek-V3.1 蒸馏 CoT 微调 Llama/OLMo 作实时判断模型，动态挑选社会选择聚合策略，用户研究(n=284)中公平与共识得分最高。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/deepaffinity_long_term_aspect_preference_prediction_in_ecommerce_using_smal/">DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models</a></strong></dt>
<dd>
<p>用 135M 小模型+结构化字符串化+多任务预测头，从时序行为预测品牌/尺码/颜色等商品属性偏好，Fashion 微 F1 0.34 超 Gemini 零样本与 3B SFT。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/improving_rare_medication_recommendation_with_counterfactual_data_augmentat/">Improving Rare Medication Recommendation with Counterfactual Data Augmentation and Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 反事实数据增强 + 病历摘要指令微调 + 逐药自回归生成，缓解罕见药长尾失效，Group-1 Jaccard 最高 +30.2%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/llm_reasoning_for_subjective_tasks_failure_modes_mitigation_and_dynamic_rea/">LLM Reasoning for Subjective Tasks: Failure Modes, Mitigation, and Dynamic Reasoning Routing</a></strong></dt>
<dd>
<p>揭示主观验证任务上显式推理与标准 RLVR 会引发推理崩溃，提出条件长度惩罚奖励与人格路由蓝图，Netflix 四任务 macro-F1 最高回升至 0.851。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/llm_as_a_judge_for_evaluating_system_responses_in_conversational_music_reco/">LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation</a></strong></dt>
<dd>
<p>首个以 400 条专家标注实证 LLM-as-a-Judge 在对话音乐推荐响应评估中与人类判断的对齐度，并给出条件化与规模选择指南。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/membership_inference_attacks_on_in_context_examples_in_llm_based_recommende/">Membership Inference Attacks on In-Context Examples in LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>首次针对 ICL 式 LLM 推荐系统提出 ItemMem 与 RecInertia 两种成员推理攻击，黑盒下攻击优势最高近 1.0，现有提示抽取防御基本失效。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/pairwise_ranking_outperforms_single_action_rl_for_offline_explanation_selec/">Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson</a></strong></dt>
<dd>
<p>将 LLM 解释生成离线化为冻结候选池，用 CPU 端 LambdaRank 逐候选打分选择，F1=0.500 超越 G-Refer/XRec 且延迟&lt;100ms。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/position_bias_undermines_preference_consistency_in_listwise_llm_based_reran/">Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking</a></strong></dt>
<dd>
<p>将 LLM 列表重排的候选置换视为诱导偏好系统，提出 PPI/GPI/LOC 三层一致性度量，揭示有效性、曝光均衡与置换一致性三者可背离。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/recap_feedback_driven_streaming_semantic_user_profiles_for_short_video_reco/">RECAP: Feedback-Driven Streaming Semantic User Profiles for Short-Video Recommendation</a></strong></dt>
<dd>
<p>将流式结构化用户画像视为有界记忆状态，用 LLM 语义差分+确定性状态机维护，并以 LLM 判官清洗隐式反馈、双塔评估器打分作 GRPO 奖励闭环优化，快手离线 uAUC +0.0084、线上人均使用时长 +0.139%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/self_evolving_recommendation_system_end_to_end_autonomous_model_optimizatio/">Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</a></strong></dt>
<dd>
<p>用 Gemini 双智能体（快循环提名+慢循环上线）自动改写优化器/架构/奖励函数，在 YouTube 线上 A/B 取得显著北极星指标提升。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/structure_preserving_projection_for_mitigating_modality_bias_in_llm_based_s/">Structure-Preserving Projection for Mitigating Modality Bias in LLM-Based Sequential Recommendation</a></strong></dt>
<dd>
<p>用余弦相似度保持与序列对比保持双损失约束协同嵌入投影，缓解 LLM 推荐中的模态偏置，20 候选协议下 HR@1 最高提升约 12pt。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/a_cascaded_generative_approach_for_e_commerce_recommendations/">A Cascaded Generative Approach for e-Commerce Recommendations</a></strong></dt>
<dd>
<p>级联 LLM 生成货架主题与 RAG 约束关键词驱动检索，教师-学生蒸馏落地，线上加购 +2.7%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 1.5B LLaMA2+BLIP-2 将图像转字幕并 token 化为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%，延迟中性。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在保持语义层级的同时缓解几何失衡，Full-Ranking 下 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/bear_towards_beam_search_aware_optimization_for_recommendation_with_large_l/">BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>揭示 SFT 训练与 beam search 推理的剪枝不一致，提出 token 级 top-B 必要条件的可微正则 BEAR，平均提升 12.50% 且训练开销几乎不变。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/beyond_static_best_of_n_bayesian_list_wise_alignment_for_llm_based_recommen/">Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用贝叶斯后验融合静态先验与在线 rollout 证据，构建自演化 BoN 目标，突破静态对齐的性能上界，NDCG@5 相对提升 23.1%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/dcgl_dual_channel_graph_learning_with_large_language_models_for_knowledge_a/">DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation</a></strong></dt>
<dd>
<p>双通道解耦 LLM 语义与 ID 行为信号，多级对比对齐 + 频率感知门控融合，四数据集全库排序全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong></dt>
<dd>
<p>融合原型意图与分布意图重构表征，以对齐-均匀性为主目标联合优化，三数据集全库排序显著超越15个基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/differentiable_semantic_id_for_generative_recommendation/">Differentiable Semantic ID for Generative Recommendation</a></strong></dt>
<dd>
<p>用 Gumbel 噪声+不确定性衰减实现可微语义ID联合训练，缓解码本坍缩，Full-Ranking 下 NDCG@10 稳定提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/every_preference_has_its_strength_injecting_ordinal_semantics_into_llm_base/">Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders</a></strong></dt>
<dd>
<p>将显式评分建模为LLM词表中的序数语义锚点，以强度感知对齐交互表征，在序列推荐Hit@1与成对偏好准确率上显著超越CF-LLM混合基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/factorized_latent_reasoning_for_llm_based_recommendation/">Factorized Latent Reasoning for LLM-based Recommendation</a></strong></dt>
<dd>
<p>将 LLM 隐空间推理分解为 K 个正交偏好因子，配合 GRPO 隐空间强化学习，在 Amazon 四域上以全库排序取得稳定提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/filling_the_gaps_selective_knowledge_augmentation_for_llm_recommenders/">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></strong></dt>
<dd>
<p>提出 CKP 比较式知识探测量化 LLM 对物品的协同知识缺口，仅对知识贫乏物品选择性注入属性与参考物品，免训练提升 Recall@1 与长尾覆盖。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/from_time_and_place_to_preference_llm_driven_geo_temporal_context_in_recomm/">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></strong></dt>
<dd>
<p>用 LLM 从时间戳+粗粒度位置生成地理-时间上下文嵌入，经轻量信息量测试与两种融合策略注入序列推荐，MovieLens 与生产数据均获增益。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习缓解长尾物品表征不足，在三个数据集多骨干上稳定提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/genrec_a_preference_oriented_generative_framework_for_large_scale_recommend/">GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>以 Page-wise NTP 解一对多歧义、线性 Token Merger 压缩 2× 输入、GRPO-SR 混合奖励对齐偏好，京东线上点击 +9.5%、成交 +8.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/gflowgr_fine_tuning_generative_recommendation_frameworks_with_generative_fl/">GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks</a></strong></dt>
<dd>
<p>将GR重构为多步集合生成任务，用GFlowNets做token级微调对齐物品效用，三数据集双骨干一致提升并落地淘宝搜索广告。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/itemrag_item_based_retrieval_augmented_generation_for_llm_based_recommendat/">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 RAG 从粗粒度用户历史检索转向细粒度物品级检索，融合共购与语义信息，标准与冷启动场景全面超越用户级 RAG 基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/l2rec_towards_dual_view_understanding_of_llms_for_personalized_recommendati/">L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation</a></strong></dt>
<dd>
<p>以参数级双视图 LoRA-MoE 替代输入/输出级融合，个性化路由对齐行为与语义视图，四数据集 N@10 提升 3.87%–8.02%，线上 CTR +9.24%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Models Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong></dt>
<dd>
<p>用 LLM 生成可迁移跨域行为+噪声过滤+两阶段训练+域感知画像，缓解域不平衡与域迁移问题，三数据集全排序 Recall/NDCG 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/lwgr_lagrangian_constrained_personalized_world_knowledge_for_generative_rec/">LWGR: Lagrangian-Constrained Personalized World Knowledge for Generative Recommendation</a></strong></dt>
<dd>
<p>用 OPQ 并行码本生成用户个性化软指令提取 LLM 世界知识，并以拉格朗日原始-对偶约束可控融合进生成式推荐，离线最高 +11.23%，线上广告收入 +1.35%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/meituan_merchant_business_diagnosis_via_policy_guided_dual_process_user_sim/">Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation</a></strong></dt>
<dd>
<p>从行为轨迹挖掘决策策略作为共享对齐层，锚定 LLM 推理与 GBDT 拟合双过程分支，群体仿真误差降至 8.80%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/modular_representation_compression_adapting_llms_for_efficient_and_effectiv/">Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 中间层表征优势 (MRA) 并用模块化解耦压缩修复，128 维表征在 CTR/重排/召回全面超越基线，线上 eCPM +2.82%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/multimodal_large_language_models_with_adaptive_preference_optimization_for_/">Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation</a></strong></dt>
<dd>
<p>面向 MLLM 序列推荐，用难度感知重加权 DPO + 高斯噪声正则分布优化，缓解样本难度失衡与跨模态语义偏差，三数据集 NDCG@3 提升 7.4%–11.0%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层消除候选顺序依赖，单次前向实现近完美排列不变性，nDCG 仅微降。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/promax_exploring_the_potential_of_llm_derived_profiles_with_distribution_sh/">ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems</a></strong></dt>
<dd>
<p>将 LLM 画像视为检索信号而非融合特征，用双路分布重塑（监督+自监督）引导基座模型，LightGCN 上 NDCG@10 最高提升 8.44%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/prompt_unknown_promotion_attacks_against_llm_based_sequential_recommender_s/">Prompt-Unknown Promotion Attacks against LLM-based Sequential Recommender Systems</a></strong></dt>
<dd>
<p>在系统提示词与受害模型均不可见的全黑盒设定下，用进化式提示蒸馏+代理模型+文本与行为双投毒实现长尾商品推广攻击，Beauty 上 HR@10 提升约 3.9 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/recgpt_mobile_on_device_large_language_models_for_user_intent_understanding/">RecGPT-Mobile: On-Device Large Language Models for User Intent Understanding in Taobao Feed Recommendation</a></strong></dt>
<dd>
<p>将 Qwen3-0.6B 量化后部署于手机端作为意图理解 Agent，把隐式行为翻译为显式搜索意图 query，配合自适应 Prompt 与意图漂移触发机制，线上平均 +1.8% CLICK / +2.7% PAY / +2.5% GMV。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/recpfn_prior_fitted_networks_for_in_context_based_recommendations/">RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations</a></strong></dt>
<dd>
<p>纯合成因果先验预训练的 PFN 式上下文学习序列推荐器，单次前向零样本适配新域，8 个基准上超越零样本基线并逼近低算力监督方法</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/rethinking_semantic_collaborative_integration_why_alignment_is_not_enough/">Rethinking Semantic–Collaborative Integration: Why Alignment Is Not Enough</a></strong></dt>
<dd>
<p>论证语义-协同全局对齐假设过强，提出共享+私有潜结构与互补性诊断，最小融合在 Amazon 三域 Full-Ranking 上超越对齐范式最高 41.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/sigma_a_semantic_grounded_instruction_driven_generative_multi_task_recommen/">SIGMA: A Semantic-Grounded Instruction-Driven Generative Multi-Task Recommender at AliExpress</a></strong></dt>
<dd>
<p>多视图语义对齐+混合SID/ID分词+指令多任务SFT+三步生成与自适应概率融合，离线HR@10达33.76%，线上GMV+7.84%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/spectran_spectral_aware_transformer_based_adapter_for_llm_enhanced_sequenti/">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用谱域 Transformer 适配器替代静态 SVD 截断，全谱注意力聚合 LLM 语义嵌入，四数据集三骨干平均提升 9.17%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/sprint_scalable_and_predictive_intent_refinement_for_llm_enhanced_session_b/">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></strong></dt>
<dd>
<p>用全局意图池约束LLM生成、以推荐效用验证意图，并蒸馏为轻量意图预测器，实现免LLM推理的会话推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/task_aware_automated_user_profile_generation_for_recommendation_simulation_/">Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models</a></strong></dt>
<dd>
<p>提出免训练三阶段画像生成框架 APG4RecSim，用语义合并+反事实因果映射将交互历史转为任务对齐画像，在判别/排序/评分三任务上取得最佳综合表现。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/think_when_needed_model_aware_reasoning_routing_for_llm_based_ranking/">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></strong></dt>
<dd>
<p>提出轻量即插即用路由器，基于排序感知特征与模型感知难度信号，逐实例决定是否触发推理，实现精度-效率帕累托最优。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/trace_a_conversational_framework_for_sustainable_tourism_recommendation_wit/">TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations</a></strong></dt>
<dd>
<p>多智能体 LLM 对话推荐框架，用澄清提问+反事实解释对用户进行可持续旅游软性助推，24 人用户研究验证。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/unifying_search_and_recommendation_in_llms_via_gradient_multi_subspace_tuni/">Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning</a></strong></dt>
<dd>
<p>用梯度多子空间分解+零空间投影替代 LoRA，在统一 S&amp;R 生成式建模中消解任务梯度冲突并保住通用知识，Qilin/Amazon 上大幅超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/verifiable_reasoning_for_llm_based_generative_recommendation/">Verifiable Reasoning for LLM-based Generative Recommendation</a></strong></dt>
<dd>
<p>提出 reason-verify-recommend 范式，用混合验证器+代理偏好预测+单调性正则，在四个数据集上显著提升生成式推荐效果。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 分层 MoE 架构，显式建模“何时问、怎么问”的阶段感知偏好引导策略，Recall@10 相对最优基线提升 14.6%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/graph_rec/cpgrec_a_balance_oriented_framework_for_personalized_video_game_recommendat/">CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations</a></strong></dt>
<dd>
<p>以 Fisher 分布符号边重加权缓解过平滑、LLM 生成偏好描述精炼表征，在 Steam 双数据集上兼顾精度与多样性。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/llm_rec/hadsf_aspect_aware_semantic_control_for_explainable_recommendation/">HADSF: Aspect Aware Semantic Control for Explainable Recommendation</a></strong></dt>
<dd>
<p>两阶段受控方面词表+动态个性化抽取，提出ADR/OFR幻觉度量，揭示幻觉-性能非单调关系，小模型可媲美大模型</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/llm_rec/towards_next_generation_recommender_systems_a_benchmark_for_personalized_re/">Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs</a></strong></dt>
<dd>
<p>构建 RecBench+ 约3万条复杂用户查询基准，覆盖显式/隐式/误导条件与画像偏好，系统评测7个LLM并揭示SFT+RFT两阶段微调最优。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/agenticshop_benchmarking_agentic_product_curation_for_personalized_web_shop/">AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping</a></strong></dt>
<dd>
<p>首个开放网络个性化商品策展基准，用真实购买史构建叙事画像与六维清单，LLM-as-a-judge 可验证评估，揭示现有智能体策展得分仅 15–38%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/bridging_semantic_understanding_and_popularity_bias_with_llms/">Bridging Semantic Understanding and Popularity Bias with LLMs</a></strong></dt>
<dd>
<p>将流行度偏差拆解为物品侧与用户侧语义，用结构化指令提示引导 RecLLM 双侧去偏，在 LtC/MRMC 与 MRR/F1 上同时超越多样性提示基线。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/generative_rec/diffusion_generative_recommendation_with_continuous_tokens/">Diffusion Generative Recommendation with Continuous Tokens</a></strong></dt>
<dd>
<p>用 σ-VAE 连续分词器替代量化码本，并以条件扩散+离散散度损失生成用户偏好，四数据集 Full-Ranking 上 HR@10 平均超 TIGER 11.76%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/from_reasoning_llms_to_bert_a_two_stage_distillation_framework_for_search_r/">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></strong></dt>
<dd>
<p>美团搜索广告两阶段框架：领域自适应推理LLM做教师标注，CRSD自蒸馏将推理能力内化进6层BERT，线上AdCTR+0.91%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/sequential_rec/generative_archetype_grounded_item_representations_for_sequential_recommend/">Generative Archetype-Grounded Item Representations for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 生成物品“目标人群原型”文本并单次前向抽取 prefill/decoding 双嵌入，再以行为关联分数调制排斥力做校准，Full-Ranking 下 HR@10 最高 +6.95%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/is_more_context_always_better_examining_llm_reasoning_capability_for_time_i/">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></strong></dt>
<dd>
<p>系统评测 GPT-4o/Claude/Gemini 在复购时间间隔预测上的零样本能力，发现 LLM 仅胜统计基线、全面落后 XGBoost，且中等上下文最优、高上下文反降。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cold_start_rec/llm_reasoning_for_cold_start_item_recommendation/">LLM Reasoning for Cold-Start Item Recommendation</a></strong></dt>
<dd>
<p>在 Netflix 冷启动重排任务上系统对比结构化推理、软自洽与 SFT/GRPO 混合微调，推理微调模型在 Discovery 指标上超越线上生产排序模型最高 8%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/multi_agent_collaborative_filtering_orchestrating_users_and_items_for_agent/">Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations</a></strong></dt>
<dd>
<p>将相似用户与相关物品实例化为LLM智能体，由中心编排器动态招募并下发个性化指令，多轮讨论聚合协同信号，三域HR@10最高提升17.9%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/prototype_aligned_federated_soft_prompts_for_continual_web_personalization/">Prototype-Aligned Federated Soft-Prompts for Continual Web Personalization</a></strong></dt>
<dd>
<p>冻结骨干上注入双时间尺度软提示，长期提示锚定DP联邦原型库，八基准NDCG@10平均+2.9%且降低遗忘与适应步数。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/agent_conversational_rec/recoworld_building_simulated_environments_for_agentic_recommender_systems/">RecoWorld: Building Simulated Environments for Agentic Recommender Systems</a></strong></dt>
<dd>
<p>提出双视角LLM仿真环境，用户模拟器在多轮交互中发出反思指令，驱动指令遵循推荐器优化留存，并给出三种历史建模方案与多智能体群体仿真蓝图。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/llm_rec/token_level_collaborative_alignment_for_llm_based_generative_recommendation/">Token-level Collaborative Alignment for LLM-based Generative Recommendation</a></strong></dt>
<dd>
<p>将CF物品级logits投影为token级分布并与one-hot融合成软标签，以soft-NTP损失显式对齐LLM生成，即插即用提升多骨干推荐效果。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/towards_fair_large_language_model_based_recommender_systems_without_costly_/">Towards Fair Large Language Model-based Recommender Systems without Costly Retraining</a></strong></dt>
<dd>
<p>将 LLM-RS 去偏重构为免重训机器遗忘任务：可学习掩码定位致偏样本 + 影响函数一步参数修正，公平性提升同时精度不降、耗时降 96%。</p>
</dd>
</dl>

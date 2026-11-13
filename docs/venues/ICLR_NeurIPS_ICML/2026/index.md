---
hide:
  - toc
---

# 🏛️ ICLR / NeurIPS / ICML (RecSys) 2026 推荐系统论文解读

41 篇 ICLR / NeurIPS / ICML (RecSys) 2026 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/generalizing_linear_autoencoder_recommenders_with_decoupled_expected_quadra/">Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss</a></strong></dt>
<dd>
<p>将 EDLAE 目标推广为 DEQL，首次给出 b&gt;0 闭式解并用 Miller 矩阵逆定理把复杂度从 O(n^4) 降到 O(n^3)，在 9 个基准上稳定超越 b=0 基线。</p>
</dd>
<dt><strong><a href="collaborative_filtering/two_level_softmax_sampling_done_right_correcting_bias_from_size_imbalance_a/">Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion</a></strong></dt>
<dd>
<p>揭示两级 softmax 采样因簇大小失衡与簇内相似度离散导致的系统性偏差，提出 S-2LS/SD-2LS 校正，KL 保真度提升 1–4 个数量级且开销近乎为零。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/can_recommender_systems_teach_themselves_a_recursive_self_improving_framewo/">Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control</a></strong></dt>
<dd>
<p>提出 RSIR 递归自我改进框架，用模型自生成序列+保真度质量控制稠密化稀疏交互，理论证明其等价于流形切向隐式正则，多骨干多数据集 Full-Ranking 下 NDCG/Recall 稳定提升 3%–14%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/gcib_graph_contrastive_information_bottleneck_for_multi_behavior_recommenda/">GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用图信息瓶颈在结构层剪枝辅助行为噪声边、用跨行为 InfoNCE 在特征层对齐目标语义，Leave-One-Out 全库排序下 NDCG@10 最高提升 37.2%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitatio/">Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy</a></strong></dt>
<dd>
<p>展开 GCF 预测为邻居对加权聚合，揭示 SSM 仅按物品侧相似度更新且忽略邻居对类型，提出类型感知 NT-SSM，NDCG@20 最高提升 21.45%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl>
<dt><strong><a href="sequential_rec/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential/">CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation</a></strong></dt>
<dd>
<p>用可学习全局 K/V 池跨用户共享协同信息，将 KV cache 压缩至 0.8% 且精度不降，解码延迟随 batch 增长近乎平坦。</p>
</dd>
<dt><strong><a href="sequential_rec/cramer_control_via_request_aware_masking_for_editing_recommenders/">CRAMER: Control via Request-Aware Masking for Editing Recommenders</a></strong></dt>
<dd>
<p>将自然语言请求编码为稀疏行列门控掩码，对冻结序列推荐骨干做后验行为编辑，采样100负例排序下全面超越四类请求感知基线且推理仅增0.018s。</p>
</dd>
<dt><strong><a href="sequential_rec/lime_link_based_user_item_interaction_modeling_with_decoupled_xor_attention/">LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling</a></strong></dt>
<dd>
<p>用低秩 link embedding 解耦用户-候选交互并预计算 QK 权重，配合 XOR 线性注意力将长历史建模降至 O(N)，实现 HSTU 级精度与 10× 推理加速。</p>
</dd>
<dt><strong><a href="sequential_rec/massive_memorization_with_hundreds_of_trillions_of_parameters_for_sequentia/">Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders</a></strong></dt>
<dd>
<p>VISTA 将超长用户历史两阶段解耦：离线用准线性注意力压缩为数百种子嵌入并缓存，下游仅做候选-摘要注意力，工业上线推理 GPU 降 94%。</p>
</dd>
<dt><strong><a href="sequential_rec/not_all_is_lost_repairing_lossy_user_preference_states_of_personalization_e/">Not All Is Lost: Repairing Lossy User Preference States of Personalization Encoders</a></strong></dt>
<dd>
<p>冻结宿主编码器与任务头，仅训练轻量 REPAIR 模块，用缓存逐步表征对压缩偏好状态做时序结构化残差校正，12 个推荐宿主 MRR/nDCG@10 全面提升。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (3)</h2>
<dl>
<dt><strong><a href="generative_rec/rethinking_semantic_id_construction_for_generative_recommendation_simhash_w/">Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment</a></strong></dt>
<dd>
<p>揭示LSH分词劣势源于与自回归解码的结构错配，提出免训练SimHash+并行解码+语义对齐的FLASH，全库排序下NDCG@5最高提升14.46%。</p>
</dd>
<dt><strong><a href="generative_rec/syngr_unleashing_the_potential_of_cross_modal_synergy_for_generative_recomm/">SynGR: Unleashing the Potential of Cross-Modal Synergy for Generative Recommendation</a></strong></dt>
<dd>
<p>用 PID 信息分解揭示多模态生成式推荐的协同信息缺失，提出显著性感知掩码+协同对比学习，Full-Ranking 下平均提升约 10%</p>
</dd>
<dt><strong><a href="generative_rec/tracer_token_reassignment_for_concept_erasure_in_generative_recommendation/">TRACER: Token ReAssignment for Concept ERasure in Generative Recommendation</a></strong></dt>
<dd>
<p>首个面向生成式推荐的概念遗忘框架，用可微 token 重分配替代直接抑制共享 SID，在 Amazon 三域上遗忘 HR@5 较 PISCES 再降 20%+ 且保留 97% 效用。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (5)</h2>
<dl>
<dt><strong><a href="llm_rec/harp_training_free_dual_profile_agentic_communication_for_llm_based_recomme/">HARP: Training-Free Dual-Profile Agentic Communication for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 LLM 推荐重构为四阶段智能体通信，用双画像、同质性门控检索与单次反思三个免训练算子补齐缺失环节，Hit@1 相对提升 6.9%–24.0%。</p>
</dd>
<dt><strong><a href="llm_rec/learning_evidence_highlighting_for_frozen_llms/">Learning Evidence Highlighting for Frozen LLMs</a></strong></dt>
<dd>
<p>用轻量 Emphasis Actor 以 RL 在冻结 LLM 的原始上下文中插入高亮标签，解耦证据选择与推理，序列推荐 HR@10 相对最优基线提升 5.38%。</p>
</dd>
<dt><strong><a href="llm_rec/principled_synthetic_data_enables_the_first_scaling_laws_for_llms_in_recomm/">Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation</a></strong></dt>
<dd>
<p>用分层合成数据课程（Item-Text/CF/UIH）替代噪声交互日志，首次在 0.6B–8B LLM CPT 上拟合出推荐域幂律缩放律，SasRec 在合成数据上 Recall@100 提升 130%。</p>
</dd>
<dt><strong><a href="llm_rec/rejuvenating_cross_entropy_loss_in_knowledge_distillation_for_recommender_s/">Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems</a></strong></dt>
<dd>
<p>揭示CE蒸馏损失与NDCG的闭包假设矛盾，提出师生协同子集拆分与自适应加权的RCE-KD，全库排序下NDCG@20最高提升11.76%。</p>
</dd>
<dt><strong><a href="llm_rec/token_efficient_item_representation_via_images_for_llm_recommender_systems/">Token-Efficient Item Representation via Images for LLM Recommender Systems</a></strong></dt>
<dd>
<p>用单张物品图像替代冗长文本描述作为 LLM 推荐输入，配合 RISA 对齐与 RERI 检索式推荐，在 Amazon 四域上兼顾效率与效果。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/agentselect_benchmark_for_narrative_query_to_agent_recommendation/">AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation</a></strong></dt>
<dd>
<p>将异构评测产物统一为叙事查询到可部署智能体配置的推荐基准，揭示长尾一次性监督下内容感知匹配的必要性。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/propersim_developing_proactive_and_personalized_ai_assistants_through_user_/">ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation</a></strong></dt>
<dd>
<p>提出 ProPerSim 仿真任务与 ProPerAssistant 基线，用 LLM 用户智能体+四维评分准则+DPO 偏好学习，让助手在时机与内容上同时个性化，14 天内平均满意度从 2.2 升至 3.3/4。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/siturecbench_evaluating_frontier_models_as_situated_recommendation_assistan/">SituRecBench: Evaluating Frontier Models as Situated Recommendation Assistants in 3D Environments</a></strong></dt>
<dd>
<p>构建 UE5 三维购物环境与共享第一人称视角的交互式基准，统一评测 VLM 助手的对话质量、推荐效用与导航有效性。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (3)</h2>
<dl>
<dt><strong><a href="multimodal_rec/can_complementary_signals_bridge_similarity_islands_manifold_augmented_grap/">Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation</a></strong></dt>
<dd>
<p>提出 MAGE，用离线 Soft Conceptor 残差注入 + 在线非对称双流 GNN 桥接互补品“相似性孤岛”，四个 Amazon 数据集 Full-Ranking 全面领先。</p>
</dd>
<dt><strong><a href="multimodal_rec/controllable_and_content_based_recommendations/">Controllable and Content-Based Recommendations</a></strong></dt>
<dd>
<p>用多模态 LLM 将图像/音频/视频条目转述为文本画像，经标签监督与 InfoNCE 对齐冻结 CF 骨干，实现可读可编辑的文本瓶颈控制。</p>
</dd>
<dt><strong><a href="multimodal_rec/intentlens_grounding_underspecified_multimodal_queries_for_recommendation_v/">IntentLens: Grounding Underspecified Multimodal Queries for Recommendation via Tool-Augmented Reasoning</a></strong></dt>
<dd>
<p>用 MLLM 工具增强推理将欠指定图文查询接地为结构化意图，再以轻量双塔排序，构建两个新基准并显著超越 MLLM 与检索基线。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl>
<dt><strong><a href="cold_start_rec/adaptive_querying_with_ai_persona_priors/">Adaptive Querying with AI Persona Priors</a></strong></dt>
<dd>
<p>用 LLM 人物画像构建有限混合贝叶斯先验，实现闭式后验更新与自适应提问，在冷启动用户/题目场景下优于 CAT/IRT 基线。</p>
</dd>
<dt><strong><a href="cold_start_rec/eating_for_a_sustainable_planet_personalized_sustainable_diet_recommendatio/">Eating for a Sustainable Planet: Personalized Sustainable Diet Recommendation via Constraint-Aware Decision-Making Modeling</a></strong></dt>
<dd>
<p>将可持续饮食推荐重构为约束感知决策问题，用可学习用户约束阈值替代偏好建模，在自建 SusDiet 上兼顾偏好精度与四维可持续性。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/model_monotonicity_in_autobidding_auctions_when_do_better_predictions_lead_/">Model Monotonicity in Autobidding Auctions: When Do Better Predictions Lead to Better Outcomes?</a></strong></dt>
<dd>
<p>用划分细化刻画模型改进，系统证明自动出价拍卖中仅三种设定保证平台指标单调，其余均可反例击穿。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/on_the_coordination_of_value_maximizing_bidders/">On the Coordination of Value-Maximizing Bidders</a></strong></dt>
<dd>
<p>证明价值最大化自动出价方在“最高价值者出价”协调机制下，RoS 合规与联盟总价值均优于独立出价，并给出充要条件与渐近最优性。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (2)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/goal_conditioned_supervised_learning_for_multi_objective_recommendation/">Goal-Conditioned Supervised Learning for Multi-Objective Recommendation</a></strong></dt>
<dd>
<p>将 GCSL 的标量目标扩展为多维向量目标，用 Transformer 编码器 + 交叉熵直接做多目标下一物品预测，无需复杂架构与权重调参，在 RetailRocket/Challenge15 上购买 HR@10 相对提升约 4-17%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/goalrank_group_relative_optimization_for_a_large_ranking_model/">GoalRank: Group-Relative Optimization for a Large Ranking Model</a></strong></dt>
<dd>
<p>证明单生成器排序器近似误差严格优于多生成器-评估器，提出组相对优化训练大排序模型，离线 H@6 最高 +25.4%，线上全量部署。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/beyond_match_maximization_and_fairness_retention_optimized_two_sided_matchi/">Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching</a></strong></dt>
<dd>
<p>将双边匹配推荐重构为留存最大化问题，用凹性 Jensen 下界把 NP-hard 排序松弛为 O(NlogN) 的 argsort 打分，在合成与真实约会平台数据上以约 70% 匹配量取得最高留存率。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/incentivized_exploration_with_stochastic_covariates_a_two_stage_mechanism_d/">Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System</a></strong></dt>
<dd>
<p>面向在线随机用户协变量的激励相容探索，提出两阶段 RCB 算法（冷启动补贴 + 逆比例间隙采样），在满足 ε-DBIC 约束下达到 $\tilde{O}(\sqrt{KdT})$ 遗憾。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/optimal_and_efficient_contextual_combinatorial_semi_bandits_with_general_fu/">Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation</a></strong></dt>
<dd>
<p>提出 SquareCB.Comb，用回归 oracle + 对数障碍边际优化实现通用函数逼近下 CCSB 的极小极大最优 regret，并在网页推荐与最短路任务上验证。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/prorl_effective_reinforcement_learning_for_proactive_recommendation_via_rec/">ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation</a></strong></dt>
<dd>
<p>揭示主动推荐中策略梯度的长度捷径与高方差缺陷，提出逐步奖励中心化与位置特定优势估计，三数据集全面超越 SOTA。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/alm_mta_front_door_causal_multi_touch_attribution_method_for_creator_ecosys/">ALM-MTA: Front-Door Causal Multi-Touch Attribution Method for Creator-Ecosystem Optimization</a></strong></dt>
<dd>
<p>以前门识别+对抗代理中介实现大规模推荐系统多触点因果归因，离线 AUC 0.907、线上创作者 +0.6%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/causal_representation_learning_for_generalisable_recommendation/">Causal Representation Learning for Generalisable Recommendation</a></strong></dt>
<dd>
<p>提出信息论解耦准则 J=I(g(T);Y|Z)−λI(g(T);Z)，用 InfoNCE+梯度反转实现，零推理开销，Spotify 千万级 A/B 提升 +0.75% 播放。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/copf_an_online_framework_for_deployment_stable_counterfactual_fairness_in_e/">COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs</a></strong></dt>
<dd>
<p>以曝光反事实定义群体机会差，用倾向日志+GA-DR残差+Residual-OI证书在线审计并控制演化图链路推荐的部署稳定公平性。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/kuairecv2_benchmarking_large_scale_continual_learning_for_diversified_and_m/">KuaiRecV2: Benchmarking Large-Scale Continual Learning for Diversified and Multi-task Recommendation</a></strong></dt>
<dd>
<p>快手 30 天 4.84 亿交互动态物品池基准，统一评测持续学习、Token 级多样性与多任务检索。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/low_pass_personalized_subgraph_federated_recommendation/">Low-pass Personalized Subgraph Federated Recommendation</a></strong></dt>
<dd>
<p>用低通谱滤波提取跨子图稳定的结构信号做个性化联邦聚合，并以局部流行度偏差感知 margin 抑制长尾偏置，五数据集 NDCG@20 最高提升 24.2%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/obliviate_efficient_unlearning_in_recommender_systems/">Obliviate: Efficient Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>两阶段低秩适配器+局部校准的推荐系统机器遗忘框架，20%删除率下兼顾遗忘完整性与效用，比全量重训快约 3 倍。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/user_embeddings_are_superpositions_of_interpretable_behavioral_modes/">User Embeddings are Superpositions of Interpretable Behavioral Modes</a></strong></dt>
<dd>
<p>将机制可解释性的叠加假设迁移到用户嵌入，用 SAE 事后分解出可命名行为模式，支撑零样本自然语言受众检索与画像。</p>
</dd>
</dl>
</div>

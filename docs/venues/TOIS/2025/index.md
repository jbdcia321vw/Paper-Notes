---
hide:
  - toc
---

# 📖 ACM TOIS 2025 推荐系统论文解读

30 篇 ACM TOIS 2025 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/lightweight_embeddings_with_graph_rewiring_for_collaborative_filtering/">Lightweight Embeddings with Graph Rewiring for Collaborative Filtering</a></strong></dt>
<dd>
<p>量化组合码本+图重连剪枝，在严格存储预算下兼顾 GNN 推荐精度与边缘端推理效率，iFashion 存储降 97.7%。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl>
<dt><strong><a href="graph_rec/kg4receval_does_knowledge_graph_really_matter_for_recommender_systems/">KG4RecEval: Does Knowledge Graph Really Matter for Recommender Systems?</a></strong></dt>
<dd>
<p>提出 KGER 指标与 KG4RecEval 框架，系统扰动/删除 KG 知识，发现 KG 对推荐精度贡献常为负或近零，挑战 KG 必有益的主流认知。</p>
</dd>
<dt><strong><a href="graph_rec/position_aware_graph_transformer_for_recommendation/">Position-aware Graph Transformer for Recommendation</a></strong></dt>
<dd>
<p>用四种为交互图定制的节点位置编码把 Transformer 全局建模能力注入 GCN 推荐骨干，在四个数据集上以 Full-Ranking 协议取得最高 27.69% 的 Recall@20 相对提升。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/symmetric_graph_contrastive_learning_against_noisy_views_for_recommendation/">Symmetric Graph Contrastive Learning against Noisy Views for Recommendation</a></strong></dt>
<dd>
<p>定义图增广产生的噪声视图，用对称指数对比损失替代 InfoNCE，使 GCL 对噪声视图鲁棒，全库排序下 NDCG@20 最高提升 12.25%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl>
<dt><strong><a href="sequential_rec/learning_robust_sequential_recommenders_through_confident_soft_labels/">Learning Robust Sequential Recommenders through Confident Soft Labels</a></strong></dt>
<dd>
<p>用模型级/数据级/训练级三种教师模块生成高置信软标签，与 one-hot 联合训练学生序列推荐器，全库排序下 Recall@10 最高提升约 11%。</p>
</dd>
<dt><strong><a href="sequential_rec/privacy_preserving_sequential_recommendation_with_collaborative_confusion/">Privacy-Preserving Sequential Recommendation with Collaborative Confusion</a></strong></dt>
<dd>
<p>用相似序列引导的逐项修改器在推荐前混淆交互序列，66% 修改率下推荐精度仍超 99% SOTA。</p>
</dd>
<dt><strong><a href="sequential_rec/ptf_fsr_a_parameter_transmission_free_federated_sequential_recommender_syst/">PTF-FSR: A Parameter Transmission-Free Federated Sequential Recommender System</a></strong></dt>
<dd>
<p>用指数机制生成扰动序列替代参数传输，实现模型隐私与轻量通信的联邦序列推荐框架，通信降至 KB 级。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (9)</h2>
<dl>
<dt><strong><a href="llm_rec/enhancing_id_based_recommendation_with_large_language_models/">Enhancing ID-based Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 LoRA 微调 LLM 仅凭纯 ID 序列生成伪交互数据，经双重过滤后注入任意 ID 基推荐模型，全库排序下平均提升 3%–9%。</p>
</dd>
<dt><strong><a href="llm_rec/fellas_enhancing_federated_sequential_recommendation_with_llm_as_external_s/">FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services</a></strong></dt>
<dd>
<p>将 LLM 作为外部服务为联邦序列推荐提供物品级与序列级表征，配合 dX-privacy 扰动与对比学习，在 Full-Ranking 下显著超越 FedSeqRec 并逼近集中式训练。</p>
</dd>
<dt><strong><a href="llm_rec/how_can_recommender_systems_benefit_from_large_language_models_a_survey/">How Can Recommender Systems Benefit from Large Language Models: A Survey</a></strong></dt>
<dd>
<p>以“WHERE×HOW”双正交视角系统梳理 LLM 融入推荐全链路的五阶段角色与四象限适配范式，并给出效率/效果/伦理三大挑战。</p>
</dd>
<dt><strong><a href="llm_rec/mitigating_propensity_bias_of_large_language_models_for_recommender_systems/">Mitigating Propensity Bias of Large Language Models for Recommender Systems</a></strong></dt>
<dd>
<p>用谱条件 iVAE 编码 LLM 侧信息避免维度坍缩，并以反事实推断消除 LLM 倾向性偏差，在 6 个 CF 骨干上全库排序一致提升。</p>
</dd>
<dt><strong><a href="llm_rec/one_model_for_all_large_language_models_are_domain_agnostic_recommendation_/">One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems</a></strong></dt>
<dd>
<p>用 LLM 编码跨域混合标题序列实现域无关推荐，采样 1000 负例评测，SPIAO 上较最优基线相对提升约 19%。</p>
</dd>
<dt><strong><a href="llm_rec/pept_expert_finding_meets_personalized_pre_training/">PEPT: Expert Finding Meets Personalized Pre-training</a></strong></dt>
<dd>
<p>将专家历史问答、投票分与专家ID统一进BERT预训练，用问题级MLM与投票预测双任务学习个性化兴趣与专长，六域CQA全库排序MRR全面领先。</p>
</dd>
<dt><strong><a href="llm_rec/recranker_instruction_tuning_large_language_model_as_ranker_for_top_k_recom/">RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation</a></strong></dt>
<dd>
<p>以自适应用户采样、位置偏移与检索信号增强构建三类排序指令微调 LLM 重排器，并提出混合排序集成，Leave-One-Out 全库排序下 HR/NDCG 最高提升 213%。</p>
</dd>
<dt><strong><a href="llm_rec/reinforced_prompt_personalization_for_recommendation_with_large_language_mo/">Reinforced Prompt Personalization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>将提示词拆为四类句式模式，用多智能体强化学习为每个用户逐句选择并组合个性化提示，在三个数据集上 NDCG@10 相对任务级提示提升 107%–132%。</p>
</dd>
<dt><strong><a href="llm_rec/thoroughly_modeling_multi_domain_pre_trained_recommendation_as_language/">Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language</a></strong></dt>
<dd>
<p>将多域序列推荐彻底转为文本生成任务，用五项指标指导行为-文本双向适配，7 数据集微调与零样本均最优，NDCG@1 最高提升 23%。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/recommender_ai_agent_integrating_large_language_models_for_interactive_reco/">Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations</a></strong></dt>
<dd>
<p>以 LLM 为大脑、传统推荐模型为工具，配合候选总线、长短期画像、计划优先执行与反思机制，在 Steam/MovieLens/Beauty 上显著超越 GPT-4 与 Chat-Rec。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/user_behavior_simulation_with_large_language_model_based_agents/">User Behavior Simulation with Large Language Model based Agents</a></strong></dt>
<dd>
<p>提出 LLM Agent 用户行为模拟器 RecAgent，以画像-记忆-动作三模块与沙盒环境复现信息茧房与从众现象，行为可信度逼近真人。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/vague_preference_policy_learning_for_conversational_recommendation/">Vague Preference Policy Learning for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 VPMCR 模糊偏好场景与 VPPL 框架，用软估计替代二值过滤，SR@15 最高提升 22%，平均轮次显著下降。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/id_centric_pre_training_for_recommendation/">ID-centric Pre-training for Recommendation</a></strong></dt>
<dd>
<p>提出以 ID 嵌入为中心的多域预训练范式 IDP，用行为感知跨域 ID 匹配器 CDIM 将预训练 ID 嵌入迁移至新域物品，冷/热启动均显著超越 UniSRec 等基线。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (3)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/causal_deconfounding_via_confounder_disentanglement_for_dual_target_cross_d/">Causal Deconfounding via Confounder Disentanglement for Dual-Target Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>显式解耦单域/跨域观测混杂因子，经后门调整保留正向效应并消除负向偏差，七数据集 HR@10 平均提升 6.17%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/coherence_guided_preference_disentanglement_for_cross_domain_recommendation/">Coherence-guided Preference Disentanglement for Cross-domain Recommendations</a></strong></dt>
<dd>
<p>用全局-局部图 + 一致性约束抽取跨域共享物品属性，并以流行度加权 Softmax 将共享用户偏好解耦为兴趣/从众，六组双目标 CDR 任务 HR@10 最高提升约 37%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/federated_semantic_learning_for_privacy_preserving_cross_domain_recommendat/">Federated Semantic Learning for Privacy-preserving Cross-domain Recommendation</a></strong></dt>
<dd>
<p>提出 FFMSR，用多层 PLM+MoE 提取语义、服务器端聚类融合跨域语义、FFT 滤波去噪，在完全非重叠联邦场景下 Full-Ranking 提升 1.4%–5.6%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/a_unified_framework_for_multi_domain_ctr_prediction_via_large_language_mode/">A Unified Framework for Multi-Domain CTR Prediction via Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 分层语义表征作跨域共享底座、可插拔 DSN 建模域特性，掩码损失解耦梯度，AUC 相对提升 24%–63% 并支持零样本新域。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (2)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/efficient_multi_task_prompt_tuning_for_recommendation/">Efficient Multi-task Prompt Tuning for Recommendation</a></strong></dt>
<dd>
<p>提出两阶段 MPT-Rec：GAN 解耦任务共享/特有信息后冻结主干，仅用任务感知 Prompt 微调新任务，参数降至 10% 且性能保持 95%+。</p>
</dd>
<dt><strong><a href="multi_task_reranking/user_invariant_preference_learning_for_multi_behavior_recommendation/">User Invariant Preference Learning for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用 IRM 范式 + VAE 从多行为交互中抽取用户不变偏好以抑制辅助行为噪声，Leave-One-Out + Full-Ranking 下 HR@10 最高提升 3.98%。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/a_thorough_performance_benchmarking_on_lightweight_embedding_based_recommen/">A Thorough Performance Benchmarking on Lightweight Embedding-based Recommender Systems</a></strong></dt>
<dd>
<p>统一协议横评 7 类轻量嵌入压缩方法，提出 MagPrune 强基线，并首次量化跨任务迁移性与树莓派端侧效率瓶颈。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/causality_inspired_fair_representation_learning_for_multimodal_recommendati/">Causality-Inspired Fair Representation Learning for Multimodal Recommendation</a></strong></dt>
<dd>
<p>因果启发的模态解耦与公平/不公平关系挖掘，实现多模态推荐的反事实公平，公平性显著优于 SOTA 且精度可比。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/multi_agent_attacks_for_black_box_social_recommendations/">Multi-agent Attacks for Black-box Social Recommendations</a></strong></dt>
<dd>
<p>用多智能体强化学习协同生成冷启动物品画像与跨社区社交画像，对黑盒社交推荐发起无目标投毒/规避攻击，全库排序下 NDCG@10 相对下降约 19%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/post_training_attribute_unlearning_in_recommender_systems/">Post-Training Attribute Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>提出锚点分布MMD与函数空间排序正则的PoT-AU双损失，多类属性遗忘复杂度降至O(T)且推荐性能近无损。</p>
</dd>
</dl>
</div>

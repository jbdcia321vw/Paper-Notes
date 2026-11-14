---
hide:
  - toc
---

# 🏛️ WSDM 2025 推荐系统论文解读

26 篇 WSDM 2025 论文解读，覆盖 13 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/s_diff_an_anisotropic_diffusion_model_for_collaborative_filtering_in_spectr/">S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain</a></strong></dt>
<dd>
<p>将扩散前向过程搬到物品图拉普拉斯谱域，用特征值参数化各向异性噪声保低频，配合 FiLM 条件去噪，全库排序下 Recall@20 最高提升约 5.5%。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl>
<dt><strong><a href="graph_rec/an_aspect_performance_aware_hypergraph_neural_network_for_review_based_reco/">An Aspect Performance-aware Hypergraph Neural Network for Review-based Recommendation</a></strong></dt>
<dd>
<p>用四元组超图建模用户-物品-方面-情感极性，以用户偏好加权聚合冲突情感推断物品方面表现，六数据集 MSE 平均降 2.30%。</p>
</dd>
<dt><strong><a href="graph_rec/lightgnn_simple_graph_neural_network_for_recommendation/">LightGNN: Simple Graph Neural Network for Recommendation</a></strong></dt>
<dd>
<p>用可学习边/嵌入剪枝+分层知识蒸馏压缩 GNN 推荐模型，80% 边与 90% 嵌入裁剪下仍超越 SOTA。</p>
</dd>
<dt><strong><a href="graph_rec/mixrec_heterogeneous_graph_collaborative_filtering/">MixRec: Heterogeneous Graph Collaborative Filtering</a></strong></dt>
<dd>
<p>参数化异质超图解耦多行为意图，配合节点/图级自适应对比学习，Leave-One-Out 全库排序下 HR@10 最高提升 9.8%。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/a_contrastive_framework_with_user_item_and_review_alignment_for_recommendat/">A Contrastive Framework with User, Item and Review Alignment for Recommendation</a></strong></dt>
<dd>
<p>将评论从“特征”升级为“对比信号”，用评论增广+三方对齐双对比任务缓解稀疏与评论缺失，Full-Ranking 下 NDCG@5 最高提升 7.7%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/scone_a_novel_stochastic_sampling_to_generate_contrastive_views_and_hard_ne/">SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation</a></strong></dt>
<dd>
<p>用分数生成模型统一生成用户-物品特定对比视图与难负样本，6 数据集全库排序 Recall@20 最高提升 1.70%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl>
<dt><strong><a href="sequential_rec/facet_aware_multi_head_mixture_of_experts_model_for_sequential_recommendati/">Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation</a></strong></dt>
<dd>
<p>用多头子嵌入独立预测+头内 MoE 解耦用户偏好，在 4 个 Amazon/ML-20m 数据集 Full-Ranking 下 NDCG@20 最高提升 8.6%。</p>
</dd>
<dt><strong><a href="sequential_rec/oracle_guided_dynamic_user_preference_modeling_for_sequential_recommendatio/">Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation</a></strong></dt>
<dd>
<p>用未来交互作为后验“Oracle”引导过去编码器学习前瞻表征，配无参低通滤波去噪与两阶段训练，六数据集采样排序全面领先。</p>
</dd>
<dt><strong><a href="sequential_rec/personalised_outfit_recommendation_via_history_aware_transformers/">Personalised Outfit Recommendation via History-aware Transformers</a></strong></dt>
<dd>
<p>双层 Transformer 堆叠建模套装内部兼容性与用户购买史，配合监督对比与自适应间隔损失实现个性化套装打分，CP-Hard AUC 提升 15.7%/19.4%。</p>
</dd>
<dt><strong><a href="sequential_rec/temporal_linear_item_item_model_for_sequential_recommendation/">Temporal Linear Item-Item Model for Sequential Recommendation</a></strong></dt>
<dd>
<p>在闭式解线性 item-item 模型上引入时间间隔加权与趋势感知归一化，Full-Ranking 下平均 NDCG@10 提升 9.07%，训练快 6.9–57 倍。</p>
</dd>
<dt><strong><a href="sequential_rec/your_causal_self_attentive_recommender_hosts_a_lonely_neighborhood/">Your Causal Self-Attentive Recommender Hosts a Lonely Neighborhood</a></strong></dt>
<dd>
<p>从稀疏局部归纳偏置与低秩近似两角度理论剖析 AE/AR 自注意力，并在 5 个基准上以 ModSAR 流水线实证 AR 全面占优。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl>
<dt><strong><a href="llm_rec/dlcrec_a_novel_approach_for_managing_diversity_in_llm_based_recommender_sys/">DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems</a></strong></dt>
<dd>
<p>将多样性可控推荐拆解为类型预测/填充/物品预测三子任务级联，配合两类数据增广，在 Movie/Steam 上 MAE_Cov@10 大幅领先。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/crs_arena_crowdsourced_benchmarking_of_conversational_recommender_systems/">CRS Arena: Crowdsourced Benchmarking of Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出基于成对对战与真实用户反馈的众包 CRS 评测平台，发布 474 段对话数据集与 Elo 榜单，揭示 R@10 与用户满意度负相关。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/reindex_then_adapt_improving_large_language_models_for_conversational_recom/">Reindex-Then-Adapt: Improving Large Language Models for Conversational Recommendation</a></strong></dt>
<dd>
<p>将多 token 电影标题重索引为单 token，再用偏置/门控适配目标平台分布，Llama2-7b 上 H@10 相对提升 59.37%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl>
<dt><strong><a href="multimodal_rec/spectrum_based_modality_representation_fusion_graph_convolutional_network_f/">Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation</a></strong></dt>
<dd>
<p>将多模态特征投影至频域做滤波去噪与点积融合，配合模态专属/融合双图学习与模态感知偏好模块，在 Amazon 三数据集全库排序下 Recall@20 相对最优基线提升约 5%。</p>
</dd>
<dt><strong><a href="multimodal_rec/teach_me_how_to_denoise_a_universal_framework_for_denoising_multi_modal_rec/">Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration</a></strong></dt>
<dd>
<p>揭示 ID 空间去噪假设在多模态空间失效，提出 AMSC 重校准 + DBPR + OT 蒸馏的即插即用去噪框架，全库排序下 Recall@20 最高提升约 11%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/large_language_model_simulator_for_cold_start_recommendation/">Large Language Model Simulator for Cold-Start Recommendation</a></strong></dt>
<dd>
<p>用 LLM 模拟冷物品的用户交互行为，并以耦合漏斗（FAISS 过滤 + LLM 精炼）把十亿级候选压到数百，离线冷召回 NDCG 提升 37%+，线上冷启 GMV +16.9%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/review_based_hyperbolic_cross_domain_recommendation/">Review-Based Hyperbolic Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>将评论文本嵌入双曲空间做跨域推荐，用度数归一化与尺度对齐避免域对齐引发的层级坍塌，NDCG@10 较 SER 提升约 10.4%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/towards_personalized_federated_multi_scenario_multi_task_recommendation/">Towards Personalized Federated Multi-Scenario Multi-Task Recommendation</a></strong></dt>
<dd>
<p>首个联邦多场景多任务推荐框架 PF-MSMTrec，通过专家参数解耦、联邦批归一化与冲突协调缓解数据/模型/任务三级冲突，AliExpress 与 Tenrec 上 CTR/CTCVR AUC 全面超越 SOTA。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/explainable_ctr_prediction_via_llm_reasoning/">Explainable CTR Prediction via LLM Reasoning</a></strong></dt>
<dd>
<p>用 PPO 双奖励（LC 用户意图对齐 + IC 与 ID-CTR 模型对齐）把 LLM 解释生成嵌入 CTR 流程，免解释数据集，AUC 最高提升 18.2%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/fusion_matters_learning_fusion_in_deep_click_through_rate_prediction_models/">Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models</a></strong></dt>
<dd>
<p>将CTR模型的融合连接与融合算子统一建模为可微搜索空间，用one-shot联合选择替代手工stacked/parallel，三数据集AUC稳定超越EDCN与NASRec。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/sequential_diversification_with_provable_guarantees/">Sequential Diversification with Provable Guarantees</a></strong></dt>
<dd>
<p>将多样性从集合度量升级为序列期望度量，证明 MaxSSD 的 NP 难性并给出 BτI/SBτI 等常数近似算法，在 7 个数据集上稳定超越 EXPLORE/MMR/DPP。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/choosing_the_best_of_both_worlds_diverse_and_novel_recommendations_through_/">Choosing the Best of Both Worlds: Diverse and Novel Recommendations through Multi-Objective Reinforcement Learning</a></strong></dt>
<dd>
<p>用标量化多目标 RL 头正则化序列推荐主干，同时优化准确率/多样性/新颖性，聚合覆盖度提升约 20% 且准确率不降。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/large_language_model_driven_policy_exploration_for_recommender_systems/">Large Language Model driven Policy Exploration for Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 蒸馏用户偏好离线预训练 A2C 推荐策略，并提出冻结+可学习双策略自适应在线适配，显著提升冷启动回报与长期收益。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/supervised_advantage_actor_critic_for_recommender_systems/">Supervised Advantage Actor-Critic for Recommender Systems</a></strong></dt>
<dd>
<p>为 RL 头引入负采样构造 SNQN，并用采样优势 A2C 重加权监督头，在 RC15/RetailRocket 上显著超越 SQN/SAC。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/privacy_preserving_orthogonal_aggregation_for_guaranteeing_gender_fairness_/">Privacy-Preserving Orthogonal Aggregation for Guaranteeing Gender Fairness in Federated Recommendation</a></strong></dt>
<dd>
<p>用正交映射+安全聚合让各性别组独立聚合，兼顾隐私与组公平，ML-100K 整体 NDCG@10 提升 7.30%。</p>
</dd>
</dl>
</div>

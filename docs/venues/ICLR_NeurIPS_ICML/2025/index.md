---
hide:
  - toc
---

# 🏛️ ICLR / NeurIPS / ICML (RecSys) 2025 推荐系统论文解读

38 篇 ICLR / NeurIPS / ICML (RecSys) 2025 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/covariate_moderated_empirical_bayes_matrix_factorization/">Covariate-moderated Empirical Bayes Matrix Factorization</a></strong></dt>
<dd>
<p>用协变量调制的经验贝叶斯先验统一矩阵分解，MovieLens 100K 上 RMSE 全面优于 EBMF/MFAI/VAE 系基线。</p>
</dd>
<dt><strong><a href="collaborative_filtering/recommendations_with_sparse_comparison_data_provably_fast_convergence_for_n/">Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization</a></strong></dt>
<dd>
<p>将成对比较反馈建模为低秩矩阵分解下的离散选择似然，证明稀疏数据下投影梯度下降在热启动邻域内指数收敛，样本复杂度近最优。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl>
<dt><strong><a href="graph_rec/contextgnn_beyond_two_tower_recommendation_systems/">ContextGNN: Beyond Two-Tower Recommendation Systems</a></strong></dt>
<dd>
<p>提出局部成对表征+全局双塔回退的混合GNN架构，在RelBench十个任务上平均MAP较最强成对基线提升20%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl>
<dt><strong><a href="sequential_rec/long_sequence_recommendation_models_need_decoupled_embeddings/">Long-Sequence Recommendation Models Need Decoupled Embeddings</a></strong></dt>
<dd>
<p>揭示长序列推荐中注意力与表征共享嵌入导致的梯度支配与冲突，提出双嵌入表 DARE，AUC 最高提升 9‰ 并加速检索 50%。</p>
</dd>
<dt><strong><a href="sequential_rec/panther_generative_pretraining_beyond_language_for_sequential_user_behavior/">PANTHER: Generative Pretraining Beyond Language for Sequential User Behavior Modeling</a></strong></dt>
<dd>
<p>将支付交易视为行为 token 做生成式预训练，SPRM 卷积+原型交叉注意力捕捉周期模式，用户画像对比学习，微信支付线上欺诈召回 +38.6%。</p>
</dd>
<dt><strong><a href="sequential_rec/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommenda/">Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义检索构建用户/物品候选池并配可学习样本合成器，替代随机增广与 K-Means 硬规则，全库排序下 HR@20 最高提升 11.82%。</p>
</dd>
<dt><strong><a href="sequential_rec/tv_rec_time_variant_convolutional_filter_for_sequential_recommendation/">TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation</a></strong></dt>
<dd>
<p>用图信号处理中的时变图滤波器替代固定卷积核与自注意力，在6个基准上全库排序平均提升7.49%且推理更快。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (6)</h2>
<dl>
<dt><strong><a href="generative_rec/actionpiece_contextually_tokenizing_action_sequences_for_generative_recomme/">ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation</a></strong></dt>
<dd>
<p>首个上下文感知的动作序列分词器，将动作建模为无序特征集，用加权共现合并+集合置换正则实现同一动作在不同上下文下差异化分词，NDCG@10 相对提升 6.0%–12.8%。</p>
</dd>
<dt><strong><a href="generative_rec/efficient_inference_for_large_language_model_based_generative_recommendatio/">Efficient Inference for Large Language Model-based Generative Recommendation</a></strong></dt>
<dd>
<p>将投机解码引入 LLM 生成式推荐，提出 N-to-K 验证下的 AtSpeed-S/R 对齐目标与松弛采样验证，实现近 2×–2.5× 无损/近似无损加速。</p>
</dd>
<dt><strong><a href="generative_rec/listwise_preference_diffusion_optimization_for_user_behavior_trajectories_p/">Listwise Preference Diffusion Optimization for User Behavior Trajectories Prediction</a></strong></dt>
<dd>
<p>将 Plackett-Luce 列表级排序信号注入扩散 ELBO，实现多步用户行为轨迹的连贯生成，并提出 SeqMatch 轨迹级指标，Full-Ranking 下 SeqMatch@50 最高提升 36.8%。</p>
</dd>
<dt><strong><a href="generative_rec/multimodal_quantitative_language_for_generative_recommendation/">Multimodal Quantitative Language for Generative Recommendation</a></strong></dt>
<dd>
<p>用 RQ-VAE 将多域多模态物品翻译为统一量化语言，配合多任务预训练微调实现知识迁移，全库排序下 NDCG@10 最高提升 14.82%。</p>
</dd>
<dt><strong><a href="generative_rec/on_efficiency_effectiveness_trade_off_of_diffusion_based_recommenders/">On Efficiency-Effectiveness Trade-off of Diffusion-based Recommenders</a></strong></dt>
<dd>
<p>用时间一致性正则平滑去噪函数实现一步生成，再用自适应偏好对齐修正轨迹偏差，兼顾扩散推荐效率与效果。</p>
</dd>
<dt><strong><a href="generative_rec/preference_diffusion_for_recommendation/">Preference Diffusion for Recommendation</a></strong></dt>
<dd>
<p>将 BPR 改写为对数似然排序目标并推导变分上界，为扩散推荐器注入多负样本偏好信号，Full-Ranking 下 R@5 最高提升 19.35%。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (10)</h2>
<dl>
<dt><strong><a href="llm_rec/can_llms_outshine_conventional_recommenders_a_comparative_evaluation/">Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation</a></strong></dt>
<dd>
<p>RecBench 系统对比 17 个 LLM 与 DLRM 在 4 种物品表征、CTR/SeqRec 双任务上的精度-效率权衡，LLM 精度领先但推理慢千倍。</p>
</dd>
<dt><strong><a href="llm_rec/context_steering_controllable_personalization_at_inference_time/">Context Steering: Controllable Personalization at Inference Time</a></strong></dt>
<dd>
<p>免训练解码期个性化：用有/无上下文两次前向的似然差线性缩放上下文影响力，实现可控个性化推荐与贝叶斯意图反演。</p>
</dd>
<dt><strong><a href="llm_rec/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_int/">FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens</a></strong></dt>
<dd>
<p>用解耦投影+RQ-VAE 将 CF 嵌入映射为冻结 LLM 词表中的离散 token，配合对比对齐实现免微调的语义可解释与性能增益。</p>
</dd>
<dt><strong><a href="llm_rec/fine_grained_list_wise_alignment_for_generative_medication_recommendation/">Fine-grained List-wise Alignment for Generative Medication Recommendation</a></strong></dt>
<dd>
<p>将用药推荐重构为逐药序列决策，提出逐步GRPO与势函数奖励塑形实现细粒度信用分配，在MIMIC-III上Jaccard达0.4836并支持安全-精度可控权衡。</p>
</dd>
<dt><strong><a href="llm_rec/improving_llms_for_recommendation_with_out_of_vocabulary_tokens/">Improving LLMs for Recommendation with Out-Of-Vocabulary Tokens</a></strong></dt>
<dd>
<p>用元路径聚类生成 OOV 层级 token 构建用户/物品 META ID，兼顾记忆性与多样性，序列推荐 HR@10 最高提升 30%+。</p>
</dd>
<dt><strong><a href="llm_rec/language_representations_can_be_what_recommenders_need_findings_and_potenti/">Language Representations Can be What Recommenders Need: Findings and Potentials</a></strong></dt>
<dd>
<p>揭示语言空间与推荐行为空间同构，提出纯语言表征的 AlphaRec，在 Amazon 三数据集 Full-Ranking 上超越 ID-based CF 6.8%–9.8%。</p>
</dd>
<dt><strong><a href="llm_rec/r_2_ec_towards_large_recommender_models_with_reasoning/">R$^2$ec: Towards Large Recommender Models with Reasoning</a></strong></dt>
<dd>
<p>双头架构统一推理生成与物品打分，RecPO 融合奖励 RL 联合优化，Full-Ranking 下 NDCG@10 最高提升 72%。</p>
</dd>
<dt><strong><a href="llm_rec/slmrec_distilling_large_language_models_into_small_for_sequential_recommend/">SLMRec: Distilling Large Language Models into Small for Sequential Recommendation</a></strong></dt>
<dd>
<p>发现 LLM 中间层冗余，用逐层分块特征蒸馏将 8 层教师压缩为 4 层学生，仅 13% 参数达 6.6x/8.0x 加速且性能不降。</p>
</dd>
<dt><strong><a href="llm_rec/think_before_recommendation_autonomous_reasoning_enhanced_recommender/">Think before Recommendation: Autonomous Reasoning-enhanced Recommender</a></strong></dt>
<dd>
<p>用纯 GRPO 强化学习替代教师蒸馏，让单个 LLM 自主习得「用户分析→物品分析→匹配→打分」推理链，评分预测 MAE 最高降 29.9%。</p>
</dd>
<dt><strong><a href="llm_rec/tree_of_preferences_for_diversified_recommendation/">Tree of Preferences for Diversified Recommendation</a></strong></dt>
<dd>
<p>用 LLM 构建粗到细偏好树挖掘欠探索兴趣并合成交互去偏，在 Twitter/Weibo/Amazon 上同时提升多样性与相关性。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/agentrecbench_benchmarking_llm_agent_based_personalized_recommender_systems/">AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems</a></strong></dt>
<dd>
<p>首个 LLM 智能体推荐系统基准，构建 U-R-I 文本交互环境与三类场景，揭示结构化工作流智能体显著优于传统与朴素 Agent。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/enhancing_personalized_multi_turn_dialogue_with_curiosity_reward/">Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward</a></strong></dt>
<dd>
<p>将用户模型信念增益作为 PBRS 内在奖励注入多轮 RLHF，使 LLM 主动探询用户属性，习题推荐成功率 54%→87.5%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl>
<dt><strong><a href="multimodal_rec/structured_spectral_reasoning_for_frequency_adaptive_multimodal_recommendat/">Structured Spectral Reasoning for Frequency-Adaptive Multimodal Recommendation</a></strong></dt>
<dd>
<p>将多模态图信号分解为等能量频带，用频带掩码一致性训练、低秩超谱算子与谱对比正则实现频率自适应融合，全库排序下冷启动 Recall@20 提升约 10%。</p>
</dd>
<dt><strong><a href="multimodal_rec/visuallens_personalization_through_task_agnostic_visual_history/">VisualLens: Personalization through Task-Agnostic Visual History</a></strong></dt>
<dd>
<p>用任务无关的日常照片历史构建“图像-描述-方面词”谱系画像，经检索+网格化+联合微调实现个性化推荐，Hit@3 超 UniMP 5-10%。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/spot_trip_dual_preference_driven_out_of_town_trip_recommendation/">SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation</a></strong></dt>
<dd>
<p>首次形式化异地行程推荐，用知识图谱静态偏好+Neural ODE动态偏好解耦建模，F1最高提升17.01%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/optimal_auction_design_in_the_joint_advertising/">Optimal Auction Design in the Joint Advertising</a></strong></dt>
<dd>
<p>联合广告单槽位Myerson最优机制+多槽位BundleNet束级IC约束神经网络，逼近理论最优并显著提升平台收益</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/learning_personalized_ad_impact_via_contextual_reinforcement_learning_under/">Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards</a></strong></dt>
<dd>
<p>将自动出价建模为带延迟泊松奖励的CMDP，提出两阶段MLE与数据分割，实现个性化出价近最优遗憾界。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/mtrec_learning_to_align_with_user_preferences_via_mental_reward_models/">MTRec: Learning to Align with User Preferences via Mental Reward Models</a></strong></dt>
<dd>
<p>用分布化逆强化学习(QR-IQL)从行为轨迹反推用户心理奖励分布，作为辅助对齐信号注入CTR/RL推荐模型，离线AUC/NCIS全面提升，工业短视频A/B观看时长+7%。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/personalized_exercise_recommendation_with_semantically_grounded_knowledge_t/">Personalized Exercise Recommendation with Semantically-Grounded Knowledge Tracing</a></strong></dt>
<dd>
<p>ExRec 用 LLM 自动标注知识点+对比学习语义表征+KT 校准，并以 KT 模型做 Model-Based Value Estimation 稳定 Q-learning，在四个教育推荐任务上显著提升知识状态增益。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/bridging_jensen_gap_for_max_min_group_fairness_optimization_in_recommendati/">Bridging Jensen Gap for Max-Min Group Fairness Optimization in Recommendation</a></strong></dt>
<dd>
<p>揭示MMF约束下小批量采样引发Jensen Gap，提出对偶重加权算法FairDual，在三个数据集六种骨干上同时提升准确率与公平性。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/facter_fairness_aware_conformal_thresholding_and_prompt_engineering_for_ena/">FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems</a></strong></dt>
<dd>
<p>用保形分位数阈值+违规触发提示工程，在黑盒 LLM 推荐器上闭环降低人口统计偏见，MovieLens 违规数降 95.5% 且精度基本不掉。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/orbit_open_recommendation_benchmark_for_reproducible_research_with_hidden_t/">ORBIT: Open Recommendation Benchmark for Reproducible Research with Hidden Tests</a></strong></dt>
<dd>
<p>统一可复现推荐基准，含5个公开数据集标准化划分与ClueWeb-Reco隐私安全网页推荐隐藏测试集，评测12个模型并引入LLM-QueryGen基线。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/recflow_an_industrial_full_flow_recommendation_dataset/">RecFlow: An Industrial Full Flow Recommendation Dataset</a></strong></dt>
<dd>
<p>首个覆盖工业六阶段漏斗全流程的推荐数据集，含1.9B阶段样本，揭示训练-服务分布偏移并验证阶段样本作为难负例与级联蒸馏的显著收益。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/secemb_sparsity_aware_secure_federated_learning_of_on_device_recommender_sy/">SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding</a></strong></dt>
<dd>
<p>用 FSS 点函数对稀疏物品嵌入更新做无损安全聚合，通信降 90x、端侧计算降 70x 且不泄露评分物品索引。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/truthful_aggregation_of_llms_with_an_application_to_online_advertising/">Truthful Aggregation of LLMs with an Application to Online Advertising</a></strong></dt>
<dd>
<p>MOSAIC 用重要性采样+循环单调性支付实现 LLM 广告拍卖的真实报价与最优分布收敛，20 候选即近最优。</p>
</dd>
</dl>
</div>

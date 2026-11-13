---
hide:
  - toc
---

# 🏛️ RecSys 2026 推荐系统论文解读

46 篇 RecSys 2026 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/personalizing_incremental_video_search_with_hybrid_text_and_id_embeddings/">Personalizing Incremental Video Search with Hybrid Text and ID Embeddings</a></strong></dt>
<dd>
<p>面向 Apple TV 逐键增量搜索，用文本语义与 ID 协同双嵌入注入 XGBoost 排序器，短前缀 NDCG@10 提升 8.63%，线上 CTR +1.14%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl>
<dt><strong><a href="sequential_rec/do_sequential_recommendation_benchmarks_really_require_higher_order_sequenc/">Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?</a></strong></dt>
<dd>
<p>用两个不学高阶序列表征的近期加权成对探针（SeqRules/PCTM）在 eSASRec 协议下反超 Transformer，质疑主流基准的容量测量效度。</p>
</dd>
<dt><strong><a href="sequential_rec/recrec_latent_interests_recursive_reasoning_for_sequential_recommendation/">RecRec: Latent Interests Recursive Reasoning for Sequential Recommendation</a></strong></dt>
<dd>
<p>解耦推理态与预测态、以 R 个潜在兴趣做递归推理的免 RL 序列推荐框架，Full-Ranking 下 NDCG@10 最高提升 14.8%。</p>
</dd>
<dt><strong><a href="sequential_rec/zero_observation_user_reactivation_with_gap_driven_dimensional_gating/">Zero-Observation User Reactivation with Gap-Driven Dimensional Gating</a></strong></dt>
<dd>
<p>定义零观测用户唤醒问题，提出冻结骨干的逐维门控插件 DeltaGate，以 Δt 与历史表征联合路由个性化历史与全局先验，&gt;365d 桶 Hit@10 提升 33%–84%。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (8)</h2>
<dl>
<dt><strong><a href="generative_rec/addressing_cross_stage_decoupling_of_semantic_and_collaborative_signals_in_/">Addressing Cross-Stage Decoupling of Semantic and Collaborative Signals in Generative Recommendation</a></strong></dt>
<dd>
<p>提出 SCRec，通过协同增强分词、语义引导生成与双曲流形对齐三模块双向补全语义与协同信号，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 相对提升 18.7%–26.6%。</p>
</dd>
<dt><strong><a href="generative_rec/cofirec_coarse_to_fine_tokenization_for_generative_recommendation/">CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation</a></strong></dt>
<dd>
<p>将物品元数据按类目→标题→描述→CF信号分层独立量化，自回归由粗到细解码，Leave-One-Out全库排序下平均超越SOTA约6.4%。</p>
</dd>
<dt><strong><a href="generative_rec/egr_embedding_native_generative_retrieval_with_a_shared_llm/">EGR: Embedding-Native Generative Retrieval with a Shared LLM</a></strong></dt>
<dd>
<p>用单一共享 LLM 同时编码物品与用户历史，以双对比损失联合训练，免量化免 grounding，线上 CVR +2.91%</p>
</dd>
<dt><strong><a href="generative_rec/empowering_cross_domain_sequential_recommendation_with_hybrid_tokenization_/">Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding</a></strong></dt>
<dd>
<p>跨域共享-专属多塔 RQ-VAE 混合分词 + 两段式串并解码，Leave-One-Out 全库排序下精度 +1.5%、时延 -85.1%。</p>
</dd>
<dt><strong><a href="generative_rec/genpage_towards_end_to_end_generative_homepage_construction_at_netflix/">GenPage: Towards End-to-End Generative Homepage Construction at Netflix</a></strong></dt>
<dd>
<p>用单个 Transformer 将用户上下文 token 化后自回归生成整页多行首页，WBC/RL 后训练，线上 +0.24% 核心参与度且延迟降 20%。</p>
</dd>
<dt><strong><a href="generative_rec/tokens_are_all_you_need_dual_purpose_semantic_ids_for_achieving_llm_level_i/">Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in Recommendation Systems</a></strong></dt>
<dd>
<p>用 RQ-VAE 语义ID 同时承担协同身份与内容重建双职责，以 SiDec 在线解码替代稠密向量 I/O，检索 HitRate@100 提升至 0.2910 且吞吐较原始稠密嵌入快 20.4%。</p>
</dd>
<dt><strong><a href="generative_rec/topology_aware_tokenization_for_generative_recommendation/">Topology-Aware Tokenization for Generative Recommendation</a></strong></dt>
<dd>
<p>揭示 RQ-VAE 语义 ID 量化中的拓扑失真问题，提出由粗到细的三级拓扑蒸馏 TopoTok，在 Amazon 三数据集 Full-Ranking 下 Recall@5 最高提升 9.42%。</p>
</dd>
<dt><strong><a href="generative_rec/unirec_bridging_the_expressive_gap_between_generative_and_discriminative_re/">UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute</a></strong></dt>
<dd>
<p>用贝叶斯等价性证明生成式推荐差距源于特征覆盖，以属性链 CoA 前缀解码 + 容量约束 SID + 条件解码上下文补齐，Shopee 线上 GMV +5.60%。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (7)</h2>
<dl>
<dt><strong><a href="llm_rec/deepaffinity_long_term_aspect_preference_prediction_in_ecommerce_using_smal/">DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models</a></strong></dt>
<dd>
<p>用 135M 小模型+结构化字符串化+多任务预测头，从时序行为预测品牌/尺码/颜色等商品属性偏好，Fashion 微 F1 0.34 超 Gemini 零样本与 3B SFT。</p>
</dd>
<dt><strong><a href="llm_rec/improving_rare_medication_recommendation_with_counterfactual_data_augmentat/">Improving Rare Medication Recommendation with Counterfactual Data Augmentation and Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 反事实数据增强 + 病历摘要指令微调 + 逐药自回归生成，缓解罕见药长尾失效，Group-1 Jaccard 最高 +30.2%。</p>
</dd>
<dt><strong><a href="llm_rec/llm_reasoning_for_subjective_tasks_failure_modes_mitigation_and_dynamic_rea/">LLM Reasoning for Subjective Tasks: Failure Modes, Mitigation, and Dynamic Reasoning Routing</a></strong></dt>
<dd>
<p>揭示主观验证任务上显式推理与标准 RLVR 会引发推理崩溃，提出条件长度惩罚奖励与人格路由蓝图，Netflix 四任务 macro-F1 最高回升至 0.851。</p>
</dd>
<dt><strong><a href="llm_rec/pairwise_ranking_outperforms_single_action_rl_for_offline_explanation_selec/">Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson</a></strong></dt>
<dd>
<p>将 LLM 解释生成离线化为冻结候选池，用 CPU 端 LambdaRank 逐候选打分选择，F1=0.500 超越 G-Refer/XRec 且延迟&lt;100ms。</p>
</dd>
<dt><strong><a href="llm_rec/recap_feedback_driven_streaming_semantic_user_profiles_for_short_video_reco/">RECAP: Feedback-Driven Streaming Semantic User Profiles for Short-Video Recommendation</a></strong></dt>
<dd>
<p>将流式结构化用户画像视为有界记忆状态，用 LLM 语义差分+确定性状态机维护，并以 LLM 判官清洗隐式反馈、双塔评估器打分作 GRPO 奖励闭环优化，快手离线 uAUC +0.0084、线上人均使用时长 +0.139%。</p>
</dd>
<dt><strong><a href="llm_rec/self_evolving_recommendation_system_end_to_end_autonomous_model_optimizatio/">Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</a></strong></dt>
<dd>
<p>用 Gemini 双智能体（快循环提名+慢循环上线）自动改写优化器/架构/奖励函数，在 YouTube 线上 A/B 取得显著北极星指标提升。</p>
</dd>
<dt><strong><a href="llm_rec/structure_preserving_projection_for_mitigating_modality_bias_in_llm_based_s/">Structure-Preserving Projection for Mitigating Modality Bias in LLM-Based Sequential Recommendation</a></strong></dt>
<dd>
<p>用余弦相似度保持与序列对比保持双损失约束协同嵌入投影，缓解 LLM 推荐中的模态偏置，20 候选协议下 HR@1 最高提升约 12pt。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (6)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/rec_a_verifiable_benchmark_for_agentic_recommender_systems/">$τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems</a></strong></dt>
<dd>
<p>用可验证目录谓词奖励、reveal-tagged 信息揭示机制与 pass^k 可靠性指标替代 LLM-as-judge，揭示智能体推荐在隐藏约束下的可靠性悬崖。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/a_position_paper_on_recommender_systems_in_the_era_of_autonomous_agents/">A Position Paper on Recommender Systems in the Era of Autonomous Agents</a></strong></dt>
<dd>
<p>提出人-智能体-平台三方混合推荐生态，将智能体视为用户侧独立代理，重构任务定义、评估范式与四大研究方向。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/conversational_recommendation_over_live_e_commerce_catalogues_with_self_ref/">Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval</a></strong></dt>
<dd>
<p>面向实时电商商品库的自刷新检索式对话购物助手，以双哈希增量同步替代全量重建，同步开销降至全量 1.8%–12.3%</p>
</dd>
<dt><strong><a href="agent_conversational_rec/llm_as_a_judge_for_evaluating_system_responses_in_conversational_music_reco/">LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation</a></strong></dt>
<dd>
<p>首个以 400 条专家标注实证 LLM-as-a-Judge 在对话音乐推荐响应评估中与人类判断的对齐度，并给出条件化与规模选择指南。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/shape_your_feed_an_llm_based_agentic_system_for_conversational_recommendati/">Shape Your Feed: An LLM-based Agentic System for Conversational Recommendation</a></strong></dt>
<dd>
<p>三层 Agentic 架构（感知/服务/自进化）实现多模态实时用户可控重排，离线对齐打分 98.85% 准确率，线上 dislike 降 2.74%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/the_utility_of_llms_in_recommender_systems_explanation_evaluation/">The Utility of LLMs in Recommender Systems Explanation Evaluation</a></strong></dt>
<dd>
<p>系统评测 14 个 LLM 作为推荐解释评审员与 216 名人类评分的一致性，发现排序相关中等但绝对评分一致性低，并给出四条实用准则。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/binge_watch_reproducible_multimodal_benchmarks_datasets_for_large_scale_mov/">Binge Watch: Reproducible Multimodal Benchmark Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M</a></strong></dt>
<dd>
<p>为 ML-10M/20M 构建可复现多模态基准 M3L（剧情/海报/预告片文本-图像-视频-音频），保留&gt;90%交互并验证音频模态的关键增益。</p>
</dd>
</dl>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl>
<dt><strong><a href="social_group_rec/are_we_really_making_progress_in_group_recommendation_unmasking_the_tie_bre/">Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion</a></strong></dt>
<dd>
<p>揭示群组推荐中训练期 sigmoid 分数压缩与评估期确定性破平交互导致的系统性评估偏差，提出并列感知 HR/NDCG 精确期望协议，使多个 SOTA 方法报告增益大幅缩水。</p>
</dd>
<dt><strong><a href="social_group_rec/consensus_vs_dissent_dynamic_llm_modeling_of_subjective_preferences_in_grou/">Consensus vs. Dissent: Dynamic LLM Modeling of Subjective Preferences in Group Recommenders</a></strong></dt>
<dd>
<p>用人类问卷+DeepSeek-V3.1 蒸馏 CoT 微调 Llama/OLMo 作实时判断模型，动态挑选社会选择聚合策略，用户研究(n=284)中公平与共识得分最高。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (3)</h2>
<dl>
<dt><strong><a href="cold_start_rec/idproxy_ctr_prediction_with_multimodal_llms_for_cold_start_recommendation_a/">IDProxy: CTR Prediction with Multimodal LLMs for Cold-Start Recommendation at Xiaohongshu</a></strong></dt>
<dd>
<p>用 MLLM 生成代理 ID 嵌入，经粗到细两阶段对齐工业 CTR 排序器，新笔记在线 AUC 提升约 2 倍于全局流量。</p>
</dd>
<dt><strong><a href="cold_start_rec/learning_sparse_representations_of_multimodal_content_for_enhanced_cold_ite/">Learning Sparse Representations of Multimodal Content for Enhanced Cold Item Recommendation</a></strong></dt>
<dd>
<p>将多模态内容冷启动表征改为 top-K 稀疏嵌入，并用 α-entmax 预稀疏激活诱导锐化与去噪，冷启动 NDCG@20 提升 16.6%–75.5% 且存储更低。</p>
</dd>
<dt><strong><a href="cold_start_rec/predicting_custom_feed_returns_for_new_bluesky_posts_a_prospective_study/">Predicting Custom-Feed Returns for New Bluesky Posts: A Prospective Study</a></strong></dt>
<dd>
<p>将新帖作为查询、信息流作为候选的冷启动路由任务，构建 collect-first/label-later 基准，LambdaRank 达 cR@10 0.7361。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/hierarchical_gtv_estimation_bridging_the_gap_between_container_level_rankin/">Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion</a></strong></dt>
<dd>
<p>面向容器式电商的层级 GTV 预估框架 HGE，用集合感知编码器 + 代理标签分解 + 残差融合解决粒度错配，离线 XAUC +0.63pp、线上 RPS +1.42%。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (5)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/embedding_subspace_partitioning_for_dynamic_multi_objective_retrieval/">Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval</a></strong></dt>
<dd>
<p>将双塔嵌入切分为任务感知正交子空间，用服务时可调权重的加权内积替代单一点积，单模型即可在线遍历多目标 Pareto 前沿。</p>
</dd>
<dt><strong><a href="multi_task_reranking/learned_cross_task_relationships_in_multi_task_models/">Learned Cross-Task Relationships in Multi-Task Models</a></strong></dt>
<dd>
<p>用交叉标签/无条件辅助头显式逼近任务两两联合分布，在 YouTube 三大场景线上 A/B 验证迁移与 RL 状态增强双重收益。</p>
</dd>
<dt><strong><a href="multi_task_reranking/lightweight_ranking_heads_accelerating_multi_task_experimentation_in_produc/">Lightweight Ranking Heads: Accelerating Multi-Task Experimentation in Production Recommender Systems</a></strong></dt>
<dd>
<p>以 stop-gradient + 无状态日训的轻量头动态注入新任务，免骨干重训，将多任务实验周期从数周压缩至数天。</p>
</dd>
<dt><strong><a href="multi_task_reranking/multi_objective_ranking_for_live_streaming_balancing_fresh_and_delayed_sign/">Multi-Objective Ranking for Live-Streaming: Balancing Fresh and Delayed Signals with Segment-Aware Targeting</a></strong></dt>
<dd>
<p>Twitch 直播精排用延迟窗口+新鲜/延迟双模型分离+分段推理加权+MMoE，DAV +0.09%、D 段 ARPU +0.56%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/position_bias_undermines_preference_consistency_in_listwise_llm_based_reran/">Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking</a></strong></dt>
<dd>
<p>将 LLM 列表重排的候选置换视为诱导偏好系统，提出 PPI/GPI/LOC 三层一致性度量，揭示有效性、曝光均衡与置换一致性三者可背离。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/long_term_user_engagement_optimization_through_model_agnostic_downstream_re/">Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning</a></strong></dt>
<dd>
<p>以 P2P 兔子洞深度、浅层关闭负奖励与新用例采纳三类下游奖励替代稀疏留存标签，在 Pinterest 多场景线上 A/B 中稳定提升会话与留存指标。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (8)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/decoupled_learning_and_selection_in_slate_recommendation_for_privacy_and_st/">Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores</a></strong></dt>
<dd>
<p>将列表推荐拆为随机打分器+确定性选择器，给出隐私后处理契约与可审计的 margin 稳定性证书，锚点加权使 Flip@K 降 53–81%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/do_we_care_about_personalization_and_explainability_an_interview_study_with/">Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers</a></strong></dt>
<dd>
<p>对 9 家新闻机构 15 位工程师的半结构化访谈，揭示个性化采纳受编辑控制/隐私/资源约束，可解释性几乎只用于内部调试而非终端用户。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/membership_inference_attacks_on_in_context_examples_in_llm_based_recommende/">Membership Inference Attacks on In-Context Examples in LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>首次针对 ICL 式 LLM 推荐系统提出 ItemMem 与 RecInertia 两种成员推理攻击，黑盒下攻击优势最高近 1.0，现有提示抽取防御基本失效。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/on_the_convergent_validity_of_offline_evaluation_designs_for_recommender_sy/">On the Convergent Validity of Offline Evaluation Designs for Recommender Systems</a></strong></dt>
<dd>
<p>以稠密真值反馈为参照，用 Kendall&#x27;s τ 检验稀疏离线评测设计的收敛效度，发现无普适最优设计且 KuaiRec 上排序近乎反转。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/probabilistic_residual_learning_for_online_recommendations/">Probabilistic Residual Learning for Online Recommendations</a></strong></dt>
<dd>
<p>以即插即用的层次贝叶斯残差模型对固定基座推荐器做概率用户聚类与 do-calculus 混杂去偏，冷启动跨域 Recall@20 最高提升约 7.6 倍。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/training_seeds_and_model_selection_stability_in_recommender_system_evaluati/">Training seeds and model-selection stability in recommender-system evaluation</a></strong></dt>
<dd>
<p>固定数据划分与超参、仅变训练种子，从用户级指标敏感性、验证选模稳定性、验证-测试一致性与推荐列表一致性四层量化种子噪声，主张种子应纳入评估协议。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/watchlens_a_configurable_platform_for_online_video_recommendation_experimen/">WatchLens: A Configurable Platform for Online Video Recommendation Experiments</a></strong></dt>
<dd>
<p>开源视频推荐在线实验平台，解耦界面/内容池/双表面策略并原生记录曝光上下文，30人被试验证观看页策略对会话留存的影响。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/whose_posts_get_ranked_identical_text_exposure_gaps_in_bluesky_custom_feeds/">Whose Posts Get Ranked: Identical-Text Exposure Gaps in Bluesky Custom Feeds</a></strong></dt>
<dd>
<p>基于自然重复帖的曝光审计：同文本跨作者在250个Bluesky自定义Feed中曝光差异达33%，作者历史主导排名</p>
</dd>
</dl>
</div>

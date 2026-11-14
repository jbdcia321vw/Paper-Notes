---
hide:
  - toc
---

# 📖 ACM TORS 2025 推荐系统论文解读

17 篇 ACM TORS 2025 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/balancing_embedding_spectrum_for_recommendation/">Balancing Embedding Spectrum for Recommendation</a></strong></dt>
<dd>
<p>从谱分析视角揭示成对学习中的嵌入坍缩，提出 DirectSpec 全通滤波直接均衡奇异值谱，MF/LightGCN 上 nDCG@10 最高提升 52.6%/42.4%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/enhancing_item_level_bundle_representation_for_bundle_recommendation/">Enhancing Item-level Bundle Representation for Bundle Recommendation</a></strong></dt>
<dd>
<p>用 B-U-I 高阶关联与预训练伪交互增强捆绑包物品级表征，全库排序下平均提升 7.5%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl>
<dt><strong><a href="sequential_rec/a_survey_on_intent_aware_recommender_systems/">A Survey on Intent-aware Recommender Systems</a></strong></dt>
<dd>
<p>系统梳理意图感知推荐(IARS)的三大技术路线与评估实践，指出离线精度导向、可复现性不足与显式意图建模缺失三大空白。</p>
</dd>
<dt><strong><a href="sequential_rec/recommending_target_actions_outside_sessions_in_the_data_poor_insurance_dom/">Recommending Target Actions Outside Sessions in the Data-poor Insurance Domain</a></strong></dt>
<dd>
<p>面向保险数据稀缺场景的跨会话 RNN 推荐，预测会话外购买目标，Full-Ranking 下 HR@3 较 SKNN_EB 提升 3.11%。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl>
<dt><strong><a href="llm_rec/a_bi_step_grounding_paradigm_for_large_language_models_in_recommendation_sy/">A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems</a></strong></dt>
<dd>
<p>提出 BIGRec 两步接地范式：先指令微调 LLM 生成物品语义 token，再以 L2 距离对齐真实物品空间，全库排序下 1024 样本即超越全量训练基线。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/impact_of_tone_aware_explanations_in_recommender_systems/">Impact of Tone-Aware Explanations in Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 生成六种语气解释做在线成对用户研究，序数逻辑回归揭示语气对 10 项感知指标、领域与人格年龄的差异化影响。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/topic_centric_explanations_for_news_recommendation/">Topic-Centric Explanations for News Recommendation</a></strong></dt>
<dd>
<p>用双层注意力主题模型 BATM 同时做新闻推荐与主题级解释，MIND 全库排序 AUC 提升约 0.4-0.5 个点，并用 NPMI/W2V 量化主题可解释性。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/formalizing_multimedia_recommendation_through_multimodal_deep_learning/">Formalizing Multimedia Recommendation through Multimodal Deep Learning</a></strong></dt>
<dd>
<p>首次将多媒体推荐对齐多模态深度学习五步流水线，提出 Which/How/When 三问形式化 Schema，并在 Elliot 上统一复现 10 个模型、兼顾精度与超越精度指标。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/data_scarcity_in_recommendation_systems_a_survey/">Data Scarcity in Recommendation Systems: A Survey</a></strong></dt>
<dd>
<p>系统梳理数据稀缺下五类缓解范式（增广/自监督/迁移/广度学习/知识图谱），并给出挑战与未来方向。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/towards_automated_model_design_on_recommender_systems/">Towards Automated Model Design on Recommender Systems</a></strong></dt>
<dd>
<p>用权重共享超网 NAS 在稠密连接+异构算子+维度+量化空间上自动搜索 CTR 模型，Criteo/KDD 降 0.001~0.003 LogLoss，FLOPs 降 8.5×。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/a_privacy_preserving_system_for_movie_recommendations_using_federated_learn/">A Privacy Preserving System for Movie Recommendations Using Federated Learning</a></strong></dt>
<dd>
<p>端到端联邦电影推荐系统，提出 FedQ 队列式联邦训练缓解非独立同分布与小样本问题，结合 NNC 压缩降低通信开销，在 MovieLens 25M 上验证 16 万客户端规模。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/a_survey_on_trustworthy_recommender_systems/">A Survey on Trustworthy Recommender Systems</a></strong></dt>
<dd>
<p>系统梳理可解释、公平、隐私、鲁棒、可控五大可信推荐维度及其交叉关系，提出跨视角关系矩阵与评估局限反思。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/dynamic_fairness_aware_recommendation_through_multi_agent_social_choice/">Dynamic Fairness-aware Recommendation through Multi-agent Social Choice</a></strong></dt>
<dd>
<p>将多利益相关方公平推荐建模为「分配+聚合」两阶段社会选择问题，SCRUF-D 框架动态调度多个公平性智能体，在真实与合成数据上实现多维权衡。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/evaluating_impact_of_user_cluster_targeted_attacks_in_matrix_factorisation_/">Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders</a></strong></dt>
<dd>
<p>系统分析 MF 推荐中面向用户簇的投毒攻击，揭示 V 矩阵是攻击传播主通道，簇内真实评分量决定脆弱性。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/evaluation_measures_of_individual_item_fairness_for_recommender_systems_a_c/">Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study</a></strong></dt>
<dd>
<p>系统批判 8 种单品曝光公平度量，修正 5 类理论缺陷并给出归一化方案与选用指南。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/impression_aware_recommender_systems/">Impression-Aware Recommender Systems</a></strong></dt>
<dd>
<p>首篇曝光感知推荐系统(IARS)系统性综述，提出统一理论框架与三维分类体系，厘清模型、数据集与评估协议。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/recommendation_unlearning_via_influence_function/">Recommendation Unlearning via Influence Function</a></strong></dt>
<dd>
<p>用影响函数一步更新模型实现推荐遗忘，联合直接损失影响与图卷积溢出影响，剪枝加速后较重训提速 250 倍以上。</p>
</dd>
</dl>
</div>

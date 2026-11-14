# 🛡️ 去偏公平、可信隐私与工业架构

**CIKM 2025** · **9** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [CIKM 2026 (4)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与工业架构 ×5 · 召回 ×4 · 精排 ×2

<dl>
<dt><strong><a href="a_universal_framework_for_offline_serendipity_evaluation_in_recommender_sys/">A Universal Framework for Offline Serendipity Evaluation in Recommender Systems via Large Language Models</a></strong></dt>
<dd>
<p>用 LLM-as-a-Judge 给推荐列表逐项打 1-5 级惊喜度分，绕开人工真值与阈值定义，实现跨数据集跨模型的离线惊喜度统一评测。</p>
</dd>
<dt><strong><a href="causality_aware_graph_aggregation_weight_estimator_for_popularity_debiasing/">Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation</a></strong></dt>
<dd>
<p>将图聚合证明为后门调整，用 VAE 式编码器-解码器估计无偏聚合权重，配合动量更新，小众物品 Recall@20 提升约 20%。</p>
</dd>
<dt><strong><a href="content_agnostic_moderation_for_stance_neutral_recommendation/">Content-Agnostic Moderation for Stance-Neutral Recommendation</a></strong></dt>
<dd>
<p>提出不依赖内容特征的立场中立化后处理审核，用双聚类离散化打散用户-物品共簇，仿真闭环中显著降低立场偏度且 CTR 仅降约 11%。</p>
</dd>
<dt><strong><a href="evaluating_and_addressing_fairness_across_user_groups_in_negative_sampling_/">Evaluating and Addressing Fairness Across User Groups in Negative Sampling for Recommender Systems</a></strong></dt>
<dd>
<p>实证揭示负采样比 K 增大使活跃用户增益、非活跃用户受损，提出分组特定负采样比并用带先验 Hyperband 优化，全库排序下 NDCG 提升 2–10%。</p>
</dd>
<dt><strong><a href="federated_continual_recommendation/">Federated Continual Recommendation</a></strong></dt>
<dd>
<p>提出 FCRec 新任务与 F3CRec 框架，客户端自适应回放+服务端物品级时序均值，在四个数据集三个 FedRec 骨干上 N@20 平均提升 5.6%–32.1%。</p>
</dd>
<dt><strong><a href="improving_recommendation_fairness_via_graph_structure_and_representation_au/">Improving Recommendation Fairness via Graph Structure and Representation Augmentation</a></strong></dt>
<dd>
<p>用双先验假设定位敏感边与敏感特征，做图剪枝+特征掩码双增广，HSIC去偏+对比学习保效用，ML-1M 上 DP@10 降 3.4%、EO@10 降 7.6%。</p>
</dd>
<dt><strong><a href="leveraging_watch_time_feedback_for_short_video_recommendations_a_causal_lab/">Leveraging Watch-time Feedback for Short-Video Recommendations: A Causal Labeling Framework</a></strong></dt>
<dd>
<p>用分位数把观看时长转成 WPR 与二值标签，并在标注阶段按视频时长分组做因果去偏，离线全指标最优、线上观看时长 +2.23%。</p>
</dd>
<dt><strong><a href="marm_unlocking_the_future_of_recommendation_systems_through_memory_augmenta/">MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation and Scalable Complexity</a></strong></dt>
<dd>
<p>以外部缓存替代多层自注意力，将序列兴趣建模复杂度从 O(n²d) 降至 O(n·d)，离线 GAUC +0.43%、线上人均时长 +2.079%。</p>
</dd>
<dt><strong><a href="what_data_is_really_necessary_a_feasibility_study_of_inference_data_minimiz/">What Data is Really Necessary? A Feasibility Study of Inference Data Minimization for Recommender Systems</a></strong></dt>
<dd>
<p>提出性能约束式推理数据最小化问题与 Greedy Removal 算法，在三个隐式反馈数据集上实现最高 84.5% 数据削减且 PRR 保持 98%。</p>
</dd>
</dl>

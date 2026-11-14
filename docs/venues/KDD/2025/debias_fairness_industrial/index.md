# 🛡️ 去偏公平、可信隐私与工业架构

**KDD 2025** · **10** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [KDD 2026 (7)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与工业架构 ×6 · 召回 ×3 · 精排 ×2 · 重排 ×2

<dl>
<dt><strong><a href="addressing_correlated_latent_exogenous_variables_in_debiased_recommender_sy/">Addressing Correlated Latent Exogenous Variables in Debiased Recommender Systems</a></strong></dt>
<dd>
<p>放松外生变量独立假设，用双变量正态潜变量建模选择与偏好相关性，蒙特卡洛似然估计实现去偏，Coat/Yahoo!R3/KuaiRec 全指标领先。</p>
</dd>
<dt><strong><a href="breaker_removing_shortcut_cues_with_user_clustering_for_single_slot_recomme/">Breaker: Removing Shortcut Cues with User Clustering for Single-slot Recommendation System</a></strong></dt>
<dd>
<p>用用户表征聚类辅助任务+多塔簇专属偏好建模，破除单槽推荐中用户固有倾向的捷径偏置，离线Recall@1提升7%并上线美团支付。</p>
</dd>
<dt><strong><a href="data_watermarking_for_sequential_recommender_systems/">Data Watermarking for Sequential Recommender Systems</a></strong></dt>
<dd>
<p>首次定义序列推荐数据集/单用户双场景数据水印，用感受野约束插入连续物品序列，5 模型 3 数据集上水印 NDCG 近 1.0 且效用无损。</p>
</dd>
<dt><strong><a href="flexible_generation_of_preference_data_for_recommendation_analysis/">Flexible Generation of Preference Data for Recommendation Analysis</a></strong></dt>
<dd>
<p>HYDRA 以用户-物品匹配、物品流行度、用户活跃度三因子混合分布生成可控合成偏好数据，复现真实基准的度分布与算法排序。</p>
</dd>
<dt><strong><a href="forward_once_for_all_structural_parameterized_adaptation_for_efficient_clou/">Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation</a></strong></dt>
<dd>
<p>用结构控制器+结构引导映射器为每台设备动态组装子网络，一次前向即完成端侧适配，兼顾隐私、带宽与算力。</p>
</dd>
<dt><strong><a href="gigl_large_scale_graph_neural_networks_at_snapchat/">GiGL: Large-Scale Graph Neural Networks at Snapchat</a></strong></dt>
<dd>
<p>Snapchat 开源十亿级图学习库 GiGL，以表格化子图采样+PyG 解耦实现 900M 节点/16.8B 边图 12 小时内端到端训练推理，支撑 35+ 线上发布。</p>
</dd>
<dt><strong><a href="gradients_as_an_action_towards_communication_efficient_federated_recommende/">Gradients as an Action: Towards Communication-Efficient Federated Recommender Systems via Adaptive Action Sharing</a></strong></dt>
<dd>
<p>将物品梯度聚类为少量“动作”共享传输，配合自适应聚类与带宽感知压缩，在 ML-100K/1M/Lastfm-2K 上以最高 96.88% 压缩率不损性能。</p>
</dd>
<dt><strong><a href="harnessing_the_power_of_interleaving_and_counterfactual_evaluation_for_airb/">Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking</a></strong></dt>
<dd>
<p>提出竞争对式 Team Draft 交错实验与在线反事实评估，用双层随机化与位置差增益估计器将 Airbnb 搜索排序实验灵敏度提升 50–100 倍。</p>
</dd>
<dt><strong><a href="merrec_a_large_scale_multipurpose_mercari_dataset_for_consumer_to_consumer_/">MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems</a></strong></dt>
<dd>
<p>首个大规模 C2C 推荐数据集 MerRec（5.56M 用户/83M 商品/1.27B 交互），并配套三塔 Mercatran 模型与四任务基准。</p>
</dd>
<dt><strong><a href="taming_recommendation_bias_with_causal_intervention_on_evolving_personal_po/">Taming Recommendation Bias with Causal Intervention on Evolving Personal Popularity</a></strong></dt>
<dd>
<p>提出演化个人流行度度量与因果去混杂训练，并在推理期用移动平均趋势外推做干预，兼顾用户-物品演化一致性，Full-Ranking 下 Recall@20 最高提升 20.4%。</p>
</dd>
</dl>

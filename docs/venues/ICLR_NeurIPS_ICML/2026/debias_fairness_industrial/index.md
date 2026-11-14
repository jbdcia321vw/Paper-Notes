# 🛡️ 去偏公平、可信隐私与工业架构

**ICLR / NeurIPS / ICML (RecSys) 2026** · **7** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** NeurIPS ×3 · 去偏公平与工业架构 ×3 · ICML ×2 · 召回 ×2 · 精排 ×2

<dl>
<dt><strong><a href="alm_mta_front_door_causal_multi_touch_attribution_method_for_creator_ecosys/">ALM-MTA: Front-Door Causal Multi-Touch Attribution Method for Creator-Ecosystem Optimization</a></strong></dt>
<dd>
<p>以前门识别+对抗代理中介实现大规模推荐系统多触点因果归因，离线 AUC 0.907、线上创作者 +0.6%。</p>
</dd>
<dt><strong><a href="causal_representation_learning_for_generalisable_recommendation/">Causal Representation Learning for Generalisable Recommendation</a></strong></dt>
<dd>
<p>提出信息论解耦准则 J=I(g(T);Y|Z)−λI(g(T);Z)，用 InfoNCE+梯度反转实现，零推理开销，Spotify 千万级 A/B 提升 +0.75% 播放。</p>
</dd>
<dt><strong><a href="copf_an_online_framework_for_deployment_stable_counterfactual_fairness_in_e/">COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs</a></strong></dt>
<dd>
<p>以曝光反事实定义群体机会差，用倾向日志+GA-DR残差+Residual-OI证书在线审计并控制演化图链路推荐的部署稳定公平性。</p>
</dd>
<dt><strong><a href="kuairecv2_benchmarking_large_scale_continual_learning_for_diversified_and_m/">KuaiRecV2: Benchmarking Large-Scale Continual Learning for Diversified and Multi-task Recommendation</a></strong></dt>
<dd>
<p>快手 30 天 4.84 亿交互动态物品池基准，统一评测持续学习、Token 级多样性与多任务检索。</p>
</dd>
<dt><strong><a href="low_pass_personalized_subgraph_federated_recommendation/">Low-pass Personalized Subgraph Federated Recommendation</a></strong></dt>
<dd>
<p>用低通谱滤波提取跨子图稳定的结构信号做个性化联邦聚合，并以局部流行度偏差感知 margin 抑制长尾偏置，五数据集 NDCG@20 最高提升 24.2%。</p>
</dd>
<dt><strong><a href="obliviate_efficient_unlearning_in_recommender_systems/">Obliviate: Efficient Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>两阶段低秩适配器+局部校准的推荐系统机器遗忘框架，20%删除率下兼顾遗忘完整性与效用，比全量重训快约 3 倍。</p>
</dd>
<dt><strong><a href="user_embeddings_are_superpositions_of_interpretable_behavioral_modes/">User Embeddings are Superpositions of Interpretable Behavioral Modes</a></strong></dt>
<dd>
<p>将机制可解释性的叠加假设迁移到用户嵌入，用 SAE 事后分解出可命名行为模式，支撑零样本自然语言受众检索与画像。</p>
</dd>
</dl>

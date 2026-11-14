# 🛡️ 去偏公平、可信隐私与工业架构

**ECIR 2025** · **6** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ECIR 2026 (2)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与工业架构 ×4 · 召回 ×4 · 重排 ×2

<dl>
<dt><strong><a href="are_representation_disentanglement_and_interpretability_linked_in_recommend/">Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study</a></strong></dt>
<dd>
<p>复现5个推荐模型×4数据集，量化解耦度与可解释性，发现解耦与效果无关但与表示可解释性强正相关。</p>
</dd>
<dt><strong><a href="embedding_cultural_diversity_in_prototype_based_recommender_systems/">Embedding Cultural Diversity in Prototype-based Recommender Systems</a></strong></dt>
<dd>
<p>在 ProtoMF 原型矩阵分解上引入 K-近邻原型过滤与原型正交分布正则，以物品国别为文化代理缓解流行度偏差，全库排序下长尾平均排名降 27%、HR@10 反升 2%。</p>
</dd>
<dt><strong><a href="how_to_diversify_any_personalized_recommender/">How to Diversify any Personalized Recommender?</a></strong></dt>
<dd>
<p>提出用户画像级预处理：用逻辑回归识别用户代表类别、userKNN 生成候选，再按 λ% 增删交互，使任意推荐器在保持精度的同时提升多样性与少数类曝光公平。</p>
</dd>
<dt><strong><a href="improving_minimax_group_fairness_in_sequential_recommendation/">Improving Minimax Group Fairness in Sequential Recommendation</a></strong></dt>
<dd>
<p>将 CVaR-DRO 首次引入 SASRec 序列推荐，无需组标注即可提升小众/短序列用户的最差组 NDCG@20，Retailrocket 整体提升约 6%。</p>
</dd>
<dt><strong><a href="okra_an_explainable_heterogeneous_multi_stakeholder_job_recommender_system/">OKRA: an Explainable, Heterogeneous, Multi-Stakeholder Job Recommender System</a></strong></dt>
<dd>
<p>异构知识图谱+多头GATv2构建求职者/企业双侧嵌入，谐波均值融合打分，nDCG@10较六基线最高提升约18%，并揭示城乡地域偏差。</p>
</dd>
<dt><strong><a href="repeat_bias_aware_optimization_of_beyond_accuracy_metrics_for_next_basket_r/">Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation</a></strong></dt>
<dd>
<p>用 MILP 重排同时优化多样性/物品公平与重复偏差，在三个零售数据集上以可接受 Recall 损失显著改善 DS、logDP 与 RepBias。</p>
</dd>
</dl>

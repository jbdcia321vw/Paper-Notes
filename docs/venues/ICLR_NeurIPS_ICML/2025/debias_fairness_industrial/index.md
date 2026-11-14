# 🛡️ 去偏公平、可信隐私与工业架构

**ICLR / NeurIPS / ICML (RecSys) 2025** · **6** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与工业架构 ×3

<dl>
<dt><strong><a href="bridging_jensen_gap_for_max_min_group_fairness_optimization_in_recommendati/">Bridging Jensen Gap for Max-Min Group Fairness Optimization in Recommendation</a></strong></dt>
<dd>
<p>揭示MMF约束下小批量采样引发Jensen Gap，提出对偶重加权算法FairDual，在三个数据集六种骨干上同时提升准确率与公平性。</p>
</dd>
<dt><strong><a href="facter_fairness_aware_conformal_thresholding_and_prompt_engineering_for_ena/">FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems</a></strong></dt>
<dd>
<p>用保形分位数阈值+违规触发提示工程，在黑盒 LLM 推荐器上闭环降低人口统计偏见，MovieLens 违规数降 95.5% 且精度基本不掉。</p>
</dd>
<dt><strong><a href="orbit_open_recommendation_benchmark_for_reproducible_research_with_hidden_t/">ORBIT: Open Recommendation Benchmark for Reproducible Research with Hidden Tests</a></strong></dt>
<dd>
<p>统一可复现推荐基准，含5个公开数据集标准化划分与ClueWeb-Reco隐私安全网页推荐隐藏测试集，评测12个模型并引入LLM-QueryGen基线。</p>
</dd>
<dt><strong><a href="recflow_an_industrial_full_flow_recommendation_dataset/">RecFlow: An Industrial Full Flow Recommendation Dataset</a></strong></dt>
<dd>
<p>首个覆盖工业六阶段漏斗全流程的推荐数据集，含1.9B阶段样本，揭示训练-服务分布偏移并验证阶段样本作为难负例与级联蒸馏的显著收益。</p>
</dd>
<dt><strong><a href="secemb_sparsity_aware_secure_federated_learning_of_on_device_recommender_sy/">SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding</a></strong></dt>
<dd>
<p>用 FSS 点函数对稀疏物品嵌入更新做无损安全聚合，通信降 90x、端侧计算降 70x 且不泄露评分物品索引。</p>
</dd>
<dt><strong><a href="truthful_aggregation_of_llms_with_an_application_to_online_advertising/">Truthful Aggregation of LLMs with an Application to Online Advertising</a></strong></dt>
<dd>
<p>MOSAIC 用重要性采样+循环单调性支付实现 LLM 广告拍卖的真实报价与最优分布收敛，20 候选即近最优。</p>
</dd>
</dl>

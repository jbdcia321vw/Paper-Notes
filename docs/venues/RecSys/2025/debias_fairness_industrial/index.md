# 🛡️ 去偏公平、可信隐私与工业架构

**RecSys 2025** · **9** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [RecSys 2026 (14)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 重排 ×5 · 去偏公平与工业架构 ×3 · 流行度去偏 ×3

<dl>
<dt><strong><a href="a_reproducibility_study_of_product_side_fairness_in_bundle_recommendation/">A Reproducibility Study of Product-side Fairness in Bundle Recommendation</a></strong></dt>
<dd>
<p>首个捆绑推荐商品侧公平性复现研究，揭示捆绑级与物品级曝光分布不一致、指标间结论分歧，并发现用户捆绑偏好越强曝光越公平。</p>
</dd>
<dt><strong><a href="leveraging_geometric_insights_in_hyperbolic_triplet_loss_for_improved_recom/">Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations</a></strong></dt>
<dd>
<p>洛伦兹双曲三元组损失 TriplH，以物品-物品项自适应间隔替代固定 margin，兼顾精度、去流行度偏差与 3 倍推理加速。</p>
</dd>
<dt><strong><a href="mitigating_popularity_bias_in_counterfactual_explanations_using_large_langu/">Mitigating Popularity Bias in Counterfactual Explanations using Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 生成用户画像并贪心剔除“出格”历史项，作为 ACCENT 反事实解释的即插即用预处理，使解释流行度分布更贴合用户偏好。</p>
</dd>
<dt><strong><a href="opening_the_black_box_interpretable_remedies_for_popularity_bias_in_recomme/">Opening the Black Box: Interpretable Remedies for Popularity Bias in Recommender Systems</a></strong></dt>
<dd>
<p>用稀疏自编码器定位流行度神经元并做激活转向，在 ML-1M/Last.fm 上以极小精度损失换取显著曝光公平提升。</p>
</dd>
<dt><strong><a href="privacy_risks_of_llm_empowered_recommender_systems_an_inversion_attack_pers/">Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective</a></strong></dt>
<dd>
<p>首次系统研究 LLM 推荐系统的 logits 反演攻击，提出相似度引导精炼，可恢复 65% 交互物品与 87% 年龄性别。</p>
</dd>
<dt><strong><a href="recps_privacy_risk_scoring_for_recommender_systems/">RecPS: Privacy Risk Scoring for Recommender Systems</a></strong></dt>
<dd>
<p>基于差分隐私推导交互级隐私分数，用 RecLiRA 成员推断攻击估计 TPR/FPR，指导细粒度交互移除以兼顾隐私与效用。</p>
</dd>
<dt><strong><a href="rethinking_the_privacy_of_text_embeddings_a_reproducibility_study_of_text_e/">Rethinking the Privacy of Text Embeddings: A Reproducibility Study of \&quot;Text Embeddings Reveal (Almost) As Much As Text\&quot;</a></strong></dt>
<dd>
<p>复现 Vec2Text 嵌入反演攻击，揭示其长度敏感性，并提出免调参 8-bit 量化作为轻量隐私防御。</p>
</dd>
<dt><strong><a href="ricciflowrec_a_geometric_root_cause_recommender_using_ricci_curvature_on_fi/">RicciFlowRec: A Geometric Root Cause Recommender Using Ricci Curvature on Financial Graphs</a></strong></dt>
<dd>
<p>用 Ollivier/Forman-Ricci 曲率与 Ricci 流在动态金融图上做根因归因，以曲率风险惩罚重排资产，NDCG@10 提升 4.9%、Top-10 波动降 28%。</p>
</dd>
<dt><strong><a href="stairway_to_fairness_connecting_group_and_individual_fairness/">Stairway to Fairness: Connecting Group and Individual Fairness</a></strong></dt>
<dd>
<p>首次用同一族度量（SD/Gini/Atk）在 3 数据集 8 组 LLMRec 运行上实证组公平与个体公平的错位，揭示组间公平掩盖组内与个体不公。</p>
</dd>
</dl>

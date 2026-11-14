# 🛡️ 去偏公平、可信隐私与工业架构

**SIGIR 2025** · **13** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [SIGIR 2026 (13)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 召回 ×5 · 去偏公平与工业架构 ×4 · 精排 ×2

<dl>
<dt><strong><a href="a_worrying_reproducibility_study_of_intent_aware_recommendation_models/">A Worrying Reproducibility Study of Intent-Aware Recommendation Models</a></strong></dt>
<dd>
<p>复现 5 个意图感知推荐模型，发现 2 个不可复现、全部被调优后的传统非神经基线超越，揭示方法学顽疾。</p>
</dd>
<dt><strong><a href="datarec_a_python_library_for_standardized_and_reproducible_data_management_/">DataRec: A Python Library for Standardized and Reproducible Data Management in Recommender Systems</a></strong></dt>
<dd>
<p>基于 55 篇顶会论文调研构建的统一数据管理库，提供可复现的过滤/划分/版本校验与跨框架导出，覆盖 18 个内置数据集。</p>
</dd>
<dt><strong><a href="diversity_aware_dual_promotion_poisoning_attack_on_sequential_recommendatio/">Diversity-aware Dual-promotion Poisoning Attack on Sequential Recommendation</a></strong></dt>
<dd>
<p>揭示攻击与推荐目标梯度冲突，提出双提升MSE攻击目标与多样性重排序列生成，在保推荐精度的同时显著提升目标物品曝光。</p>
</dd>
<dt><strong><a href="exploring_the_escalation_of_source_bias_in_user_data_and_recommender_system/">Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop</a></strong></dt>
<dd>
<p>揭示 AIGC 在用户-数据-系统反馈回路中被逐步放大的来源偏见，并提出基于 L1 损失的黑盒去偏方法维持内容生态平衡。</p>
</dd>
<dt><strong><a href="extending_movielens_32m_to_provide_new_evaluation_objectives/">Extending MovieLens-32M to Provide New Evaluation Objectives</a></strong></dt>
<dd>
<p>用 TREC 池化法为 ML-32M 构建 51 用户偏好型 qrels，揭示传统 train/test 评测的流行度偏置并给出新评测目标。</p>
</dd>
<dt><strong><a href="id_free_not_risk_free_llm_powered_agents_unveil_risks_in_id_free_recommende/">ID-Free Not Risk-Free: LLM-Powered Agents Unveil Risks in ID-Free Recommender Systems</a></strong></dt>
<dd>
<p>揭示无ID推荐器可被LLM多智能体改写文本攻击，提出TextSimu攻击与RewriteDetection检测，Hit@50最高提升数倍。</p>
</dd>
<dt><strong><a href="llm_generated_fake_news_induces_truth_decay_in_news_ecosystem_a_case_study_/">LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation</a></strong></dt>
<dd>
<p>构建约 7.5 万条人机混合新闻推荐数据集，模拟 LLM 生成假新闻在候选、历史、训练集三阶段渗透，揭示真实新闻排序优势被侵蚀的 Truth Decay 现象并以困惑度给出解释。</p>
</dd>
<dt><strong><a href="pre_training_for_recommendation_unlearning/">Pre-training for Recommendation Unlearning</a></strong></dt>
<dd>
<p>提出预训练式影响编码器 UnlearnRec，将遗忘请求映射为嵌入漂移，免重训实现高效图推荐遗忘，10x 加速且保持精度。</p>
</dd>
<dt><strong><a href="recgaze_the_first_eye_tracking_and_user_interaction_dataset_for_carousel_in/">RecGaze: The First Eye Tracking and User Interaction Dataset for Carousel Interfaces</a></strong></dt>
<dd>
<p>首个轮播界面眼动+点击+光标+选择解释数据集，87用户3477次交互，揭示F型浏览与位置偏置。</p>
</dd>
<dt><strong><a href="revisiting_algorithmic_audits_of_tiktok_poor_reproducibility_and_short_term/">Revisiting Algorithmic Audits of TikTok: Poor Reproducibility and Short-term Validity of Findings</a></strong></dt>
<dd>
<p>复现 TikTok 推荐审计发现可复现性极差、结论仅短期有效，观看行为才是最强个性化信号，且结论高度依赖度量选择。</p>
</dd>
<dt><strong><a href="toward_holistic_evaluation_of_recommender_systems_powered_by_generative_mod/">Toward Holistic Evaluation of Recommender Systems Powered by Generative Models</a></strong></dt>
<dd>
<p>提出 Gen-RecSys 评估挑战二分法与 WH-六问 + 三维目标(D1–D5)整体评估框架，用场景化案例揭示传统 NDCG 无法覆盖的幻觉、偏见与政策合规风险。</p>
</dd>
<dt><strong><a href="unbiased_collaborative_filtering_with_fair_sampling/">Unbiased Collaborative Filtering with Fair Sampling</a></strong></dt>
<dd>
<p>通过对称补充采样使每个用户/物品等概率成为正负样本，无需估计倾向分即可消除流行度偏置，全库排序下 Recall/NDCG 与 ARP 双优。</p>
</dd>
<dt><strong><a href="why_is_normalization_necessary_for_linear_recommenders/">Why is Normalization Necessary for Linear Recommenders?</a></strong></dt>
<dd>
<p>理论剖析线性自编码器归一化对流行度/邻域偏差的作用，提出数据自适应归一化 DAN，六数据集全库排序下长尾最高提升 128.57%。</p>
</dd>
</dl>

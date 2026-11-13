# 🛡️ 去偏公平、可信隐私与工业架构

**KDD 2026** · **5** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [CIKM 2026 (2)](../../../CIKM/2026/debias_fairness_industrial/) · [RecSys 2026 (8)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2026 (11)](../../../SIGIR/2026/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2026 (4)](../../../TORS/2026/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 精排 ×4 · 去偏公平与工业架构 ×3

<dl>
<dt><strong><a href="canniuplift_a_holistic_framework_for_mitigating_seller_and_incentive_cannib/">CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling</a></strong></dt>
<dd>
<p>用平台级GMV一致性约束与核销路径分解，缓解多卖家电商uplift建模中的卖家级与激励级蚕食，线上ΔGMV相对提升4.08%。</p>
</dd>
<dt><strong><a href="isotonic_layer_a_unified_framework_for_recommendation_calibration_and_debia/">Isotonic Layer: A Unified Framework for Recommendation Calibration and Debiasing</a></strong></dt>
<dd>
<p>将非负桶权重参数化为上下文嵌入，用单个可微分段线性层统一校准与位置去偏，LinkedIn 线上 A/B 显著提升。</p>
</dd>
<dt><strong><a href="llm_as_a_judge_for_reliable_and_explainable_offline_evaluation_in_top_k_rec/">LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation</a></strong></dt>
<dd>
<p>用 LLM 从用户文本行为构建语义偏好代理替代 holdout ID 匹配，以 reasoning-then-scoring 生成可解释 Top-K 评测，模型排序一致性较常规测试提升 40–150%。</p>
</dd>
<dt><strong><a href="rethinking_popularity_bias_in_collaborative_filtering_via_analytical_vector/">Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition</a></strong></dt>
<dd>
<p>揭示 BPR 优化在嵌入空间刻出“流行度方向”的几何偏差，提出非对称方向分解校正 DDC，在 5 个骨干上 MRR@10 提升 6%–29% 且 AvgPop@10 降 21%–47%。</p>
</dd>
<dt><strong><a href="the_pitfall_of_scaling_up_uncovering_and_mitigating_popularity_bias_amplifi/">The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders</a></strong></dt>
<dd>
<p>揭示 Transformer 序列推荐随深度加深发生谱坍缩并放大流行度偏差，提出逐层注意力列和与 FFN 谱范数正则 SPRINT，在 6 数据集上同时提升准确率与长尾公平性。</p>
</dd>
</dl>

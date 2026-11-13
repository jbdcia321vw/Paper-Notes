# 🎯 CTR/CVR 预估与特征交叉

**WWW (TheWebConf) 2026** · **5** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×5 · 精排 ×5 · WWW 2026 ×3 · WWW ×2

<dl>
<dt><strong><a href="delayed_feedback_modeling_for_post_click_gross_merchandise_volume_predictio/">Delayed Feedback Modeling for Post-Click Gross Merchandise Volume Prediction: Benchmark, Insights and Approaches</a></strong></dt>
<dd>
<p>首个后点击GMV延迟反馈基准TRACE，提出复购感知双分支READER，在线流式训练下ACC相对提升2.19%</p>
</dd>
<dt><strong><a href="genci_generative_modeling_of_user_interest_shift_via_cohort_based_intent_le/">GenCI: Generative Modeling of User Interest Shift via Cohort-based Intent Learning for CTR Prediction</a></strong></dt>
<dd>
<p>用RQ-VAE语义ID+NTP生成兴趣群组，再以分层候选感知注意力注入精排，弥合召回-排序意图鸿沟，三数据集AUC全面SOTA。</p>
</dd>
<dt><strong><a href="modeling_cascaded_delay_feedback_for_online_net_conversion_rate_prediction_/">Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Insights and Solutions</a></strong></dt>
<dd>
<p>首个淘宝净转化率流式基准 CASCADE，提出 CVR-RFR 级联+分阶段去偏+延迟感知排序损失的 TESLA，NetCVR 的 RI-AUC 绝对提升 12.41%。</p>
</dd>
<dt><strong><a href="onetrans_unified_feature_interaction_and_sequence_modeling_with_one_transfo/">OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</a></strong></dt>
<dd>
<p>用单一因果 Transformer 统一序列建模与特征交叉，配合金字塔剪枝与跨请求 KV 缓存，线上 GMV/u 提升 5.68%</p>
</dd>
<dt><strong><a href="unleashing_the_potential_of_sparse_attention_on_long_term_behaviors_for_ctr/">Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction</a></strong></dt>
<dd>
<p>面向长行为序列 CTR 的个性化时间分块 + 三分支稀疏注意力 SparseCTR，AUC 相对提升 8.49%，线上 CTR +1.72%。</p>
</dd>
</dl>

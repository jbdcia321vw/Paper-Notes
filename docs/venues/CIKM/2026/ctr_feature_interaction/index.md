# 🎯 CTR/CVR 预估与特征交叉

**CIKM 2026** · **4** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [KDD 2026 (3)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2026 (1)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2026 (10)](../../../SIGIR/2026/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (2)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×3 · 精排 ×3

<dl>
<dt><strong><a href="bidding_aware_retrieval_for_multi_stage_consistency_in_online_advertising/">Bidding-Aware Retrieval for Multi-Stage Consistency in Online Advertising</a></strong></dt>
<dd>
<p>将竞价信号注入召回打分，用单调性约束+多任务蒸馏+任务注意力精炼实现召回-精排 eCPM 一致性，线上收入 +4.32%。</p>
</dd>
<dt><strong><a href="funnelcausalnet_funnel_aware_joint_conversion_revenue_uplift_for_multi_tier/">FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation</a></strong></dt>
<dd>
<p>以漏斗恒等式 μ_gmv=μ_conv·μ_val 耦合双头 uplift 估计，配 RCT 锚定拉格朗日预算分配与 Bonferroni 联合保形审计带，工业多档券 RCT 上 7/7 锚点 ΔROI 最优。</p>
</dd>
<dt><strong><a href="jointly_optimizing_debiased_ctr_and_uplift_for_coupons_marketing_a_unified_/">Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing: A Unified Causal Framework</a></strong></dt>
<dd>
<p>UniMVT 用 MoE 解耦混淆表征 + logit 空间单调线性假设，统一估计去偏 Base CTR 与多值券面额 Uplift，线上收入 +18.14%。</p>
</dd>
<dt><strong><a href="uniscale_synergistic_entire_space_data_and_model_scaling_for_search_ranking/">UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking</a></strong></dt>
<dd>
<p>提出数据-架构协同缩放框架：ES³全空间样本系统扩展训练信号，HHSFT异构分层Transformer吸收跨域分布，离线AUC+1.14%、线上GMV+2.04%。</p>
</dd>
</dl>

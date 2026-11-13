# 🌐 跨域与多场景推荐

**CIKM 2026** · **3** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [KDD 2026 (1)](../../../KDD/2026/cross_domain_multi_scenario/) · [SIGIR 2026 (4)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/cross_domain_multi_scenario/)

<dl>
<dt><strong><a href="a_dual_expert_strategy_integrating_llms_to_mitigate_negative_transfer_in_cr/">A Dual-Expert Strategy Integrating LLMs to Mitigate Negative Transfer in Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用域门控双专家+物品级注意力掩码+双池负采样对比学习，抑制 LLM 跨域序列推荐中的负迁移，10 域上超越 26 个基线，线上 CTR 相对提升 47.6%。</p>
</dd>
<dt><strong><a href="fedcgr_federated_cross_domain_generative_recommendation/">FedCGR: Federated Cross-Domain Generative Recommendation</a></strong></dt>
<dd>
<p>以固定 SID 词表为跨域通信接口，用可靠性感知 CF 残差与原型个性化聚合解决联邦生成式 CDR 的语义瓶颈与负迁移，全库排序下显著超越联邦生成基线。</p>
</dd>
<dt><strong><a href="lfm4ads_large_foundation_model_for_ads_recommendation/">LFM4Ads: Large Foundation Model for Ads Recommendation</a></strong></dt>
<dd>
<p>腾讯广告提出预训练-迁移范式，联合迁移用户/物品/交叉表征，以时间感知EMA聚合样本级CR解耦上游推理，三级粒度下游复用，离线AUC+0.0189、线上GMV+2.45%。</p>
</dd>
</dl>

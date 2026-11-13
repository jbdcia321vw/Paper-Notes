# 🌐 跨域与多场景推荐

**WWW (TheWebConf) 2026** · **4** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [CIKM 2026 (4)](../../../CIKM/2026/cross_domain_multi_scenario/) · [KDD 2026 (4)](../../../KDD/2026/cross_domain_multi_scenario/) · [SIGIR 2026 (4)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/)

🔥 **高频主题：** WWW 2026 ×4 · 精排 ×2

<dl>
<dt><strong><a href="fedecider_an_llm_based_framework_for_federated_cross_domain_recommendation/">FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>联邦跨域 LLM 推荐中，服务器只广播 LoRA 更新的方向分量以消除尺度噪声，客户端用可学习个性化权重做数据感知融合，Goodreads 平均 Hit@5 提升 3.8%。</p>
</dd>
<dt><strong><a href="generative_data_transformation_from_mixed_to_unified_data/">Generative Data Transformation: From Mixed to Unified Data</a></strong></dt>
<dd>
<p>提出数据侧范式 Taesar，用三模型对比解码把跨域混合序列重生成纯目标域序列，免改架构即插即用，Books HR@10 提升 29.4%。</p>
</dd>
<dt><strong><a href="s2cdr_smoothing_sharpening_process_model_for_cross_domain_recommendation/">S2CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用无噪声热方程平滑+锐化ODE替代扩散模型做跨域冷启动，免训练即在三场景六跨域对上 HR/NDCG 平均提升 7.07%/7.42%</p>
</dd>
<dt><strong><a href="the_double_edged_sword_of_knowledge_transfer_diagnosing_and_curing_fairness/">The Double-Edged Sword of Knowledge Transfer: Diagnosing and Curing Fairness Pathologies in Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>理论揭示跨域推荐中“差异迁移”与“信息增益不均”两类公平性病理，提出模型无关的 CDFA 框架，在 Tenrec QB/QK 上 UGF 最高降 98.97% 且精度不降。</p>
</dd>
</dl>

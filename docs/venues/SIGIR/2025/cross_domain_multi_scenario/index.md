# 🌐 跨域与多场景推荐

**SIGIR 2025** · **6** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [SIGIR 2026 (4)](../../2026/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (4)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../../KDD/2025/cross_domain_multi_scenario/) · [KDD 2026 (4)](../../../KDD/2026/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cross_domain_multi_scenario/)

🔥 **高频主题：** 召回 ×3 · 跨域序列推荐 ×3 · 精排 ×2

<dl>
<dt><strong><a href="adaptive_domain_scaling_for_personalized_sequential_modeling_in_recommender/">Adaptive Domain Scaling for Personalized Sequential Modeling in Recommenders</a></strong></dt>
<dd>
<p>用域特征动态生成序列物品的个性化MLP权重与候选物品的多查询表示，改造目标注意力实现多域序列建模，抖音双场景线上GMV+0.79%、ADVV+1.00%。</p>
</dd>
<dt><strong><a href="adaptive_graph_integration_for_cross_domain_recommendation_via_heterogeneou/">Adaptive Graph Integration for Cross-Domain Recommendation via Heterogeneous Graph Coordinators</a></strong></dt>
<dd>
<p>用异质自适应图协调器把多域图拼成统一结构做多域预训练，再以图提示迁移到目标域，Full-Ranking 下全面超越 9 个基线。</p>
</dd>
<dt><strong><a href="bridge_the_domains_large_language_models_enhanced_cross_domain_sequential_r/">Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义嵌入+分层画像桥接跨域物品与用户偏好，破解重叠困境与转移复杂性，三数据集 H@10 最高提升 31.7%。</p>
</dd>
<dt><strong><a href="cdc_causal_domain_clustering_for_multi_domain_recommendation/">CDC: Causal Domain Clustering for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用孤立/混合双亲和矩阵+因果发现系数自适应融合，联合优化目标域簇与训练源域集，50+域上DomainAUC提升4.14‰/6.76‰，线上eCPM+4.9%</p>
</dd>
<dt><strong><a href="learning_universal_user_representations_leveraging_cross_domain_user_intent/">Learning Universal User Representations Leveraging Cross-domain User Intent at Snapchat</a></strong></dt>
<dd>
<p>将 Snapchat 多业务面事件拼接为统一序列训练 Transformer 用户塔，产出跨域通用用户表征，线上多场景 A/B 全面正向。</p>
</dd>
<dt><strong><a href="x_cross_dynamic_integration_of_language_models_for_cross_domain_sequential_/">X-Cross: Dynamic Integration of Language Models for Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>逐层动态集成多个 LoRA 源域语言模型，仅用 25% 参数与 50-75% 更少微调数据即达/超 LoRA 跨域序列推荐效果。</p>
</dd>
</dl>

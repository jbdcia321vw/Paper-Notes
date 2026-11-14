# ❄️ 冷启动与长尾推荐

**CIKM 2025** · **4** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [CIKM 2026 (1)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/cold_start_rec/) · [RecSys 2026 (4)](../../../RecSys/2026/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/cold_start_rec/)

🔥 **高频主题：** 召回 ×2 · 精排 ×2

<dl>
<dt><strong><a href="csrm_llm_embracing_multilingual_llms_for_cold_start_relevance_matching_in_e/">CSRM-LLM: Embracing Multilingual LLMs for Cold-Start Relevance Matching in Emerging E-commerce Markets</a></strong></dt>
<dd>
<p>多语言 LLM 联合翻译任务、检索式查询增强与多轮自蒸馏，解决新兴市场零标注相关性匹配，离线 F1 0.965、线上缺陷率降 45.8%。</p>
</dd>
<dt><strong><a href="harnessing_light_for_cold_start_recommendations_leveraging_epistemic_uncert/">Harnessing Light for Cold-Start Recommendations: Leveraging Epistemic Uncertainty to Enhance Performance in User-Item Interactions</a></strong></dt>
<dd>
<p>用集成 CVAE 生成暖化嵌入，并以 Sinkhorn 散度版 PaiDE 估计认知不确定性做优化，在冷启动 CTR 上稳定超越 CVAR 等基线。</p>
</dd>
<dt><strong><a href="multi_modal_adaptive_mixture_of_experts_for_cold_start_recommendation/">Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation</a></strong></dt>
<dd>
<p>用双层 MoE（模态专属专家+动态门控融合）替代拼接/平均融合，在 Amazon 三域冷启动全库排序上 Recall@10 最高提升 16.17%。</p>
</dd>
<dt><strong><a href="shallow_autoencoding_recommender_with_cold_start_handling_via_side_features/">Shallow AutoEncoding Recommender with Cold Start Handling via Side Features</a></strong></dt>
<dd>
<p>将用户/物品侧特征拼入 EASE 输入矩阵，并用内容相似度矩阵作冷物品权重先验，闭式解统一处理双冷启动。</p>
</dd>
</dl>

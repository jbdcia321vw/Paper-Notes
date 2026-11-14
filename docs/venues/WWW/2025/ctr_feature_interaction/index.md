# 🎯 CTR/CVR 预估与特征交叉

**WWW (TheWebConf) 2025** · **7** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (5)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/)

🔥 **高频主题：** WWW 2025 ×7 · CTR预估与特征交叉 ×6 · 精排 ×4

<dl>
<dt><strong><a href="bat_benchmark_for_auto_bidding_task/">BAT: Benchmark for Auto-bidding Task</a></strong></dt>
<dd>
<p>发布 Avito 双拍卖格式（VCG/FP）真实自动出价数据集，配套 ALM/TA-PID 等 5 个基线，覆盖预算平滑与 CPC 约束两类 RTB 任务。</p>
</dd>
<dt><strong><a href="ctr_driven_advertising_image_generation_with_multimodal_large_language_mode/">CTR-Driven Advertising Image Generation with Multimodal Large Language Models</a></strong></dt>
<dd>
<p>用 MLLM 双分支奖励模型把 CTR 相对比较转化为偏好信号，结合产品中心偏好优化 PCPO 驱动广告图生成，线上 CTR 提升 2%。</p>
</dd>
<dt><strong><a href="efficient_and_practical_approximation_algorithms_for_advertising_in_content/">Efficient and Practical Approximation Algorithms for Advertising in Content Feeds</a></strong></dt>
<dd>
<p>针对信息流原生广告位分配，提出反向贪心 2-近似算法，突破前人 4-近似瓶颈，并在合成与真实数据上首次系统评测。</p>
</dd>
<dt><strong><a href="middleman_bias_in_advertising_aligning_relevance_of_keyphrase_recommendatio/">Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search</a></strong></dt>
<dd>
<p>揭示广告关键词相关性过滤中的“中间人偏差”，用Search相关性判定重构训练标签，并以轻量Cross-Encoder在eBay线上取得CVR+11.6%的收益。</p>
</dd>
<dt><strong><a href="on_the_practice_of_deep_hierarchical_ensemble_network_for_ad_conversion_rat/">On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction</a></strong></dt>
<dd>
<p>Pinterest 工业实践：两层 DHEN 多任务骨干 + 五类行为序列端到端建模 + 未来行为自监督辅助损失，线上 CPA 降 2.15%。</p>
</dd>
<dt><strong><a href="prectr_a_synergistic_framework_for_integrating_personalized_search_relevanc/">PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction</a></strong></dt>
<dd>
<p>用条件概率融合把相关性分级与CTR统一进单模型，配两阶段训练、语义一致性正则与个性化相关性激励，线上CTR+0.4%、GMV+1.1%。</p>
</dd>
<dt><strong><a href="rtbagent_a_llm_based_agent_system_for_real_time_bidding/">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></strong></dt>
<dd>
<p>首个 LLM Agent 出价系统，用两步决策+多类记忆+专家策略工具在 iPinYou 上以调整因子方式超越规则/RL/扩散基线。</p>
</dd>
</dl>

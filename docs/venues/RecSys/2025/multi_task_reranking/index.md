# ⚖️ 多任务学习与列表级重排

**RecSys 2025** · **5** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [RecSys 2026 (7)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 重排 ×3

<dl>
<dt><strong><a href="d_rdw_diversity_driven_random_walks_for_news_recommender_systems/">D-RDW: Diversity-Driven Random Walks for News Recommender Systems</a></strong></dt>
<dd>
<p>将规范目标分布(NTD)作为线性约束嵌入随机游走采样，实现可解释、低能耗的新闻多样化重排，能耗降低一个数量级。</p>
</dd>
<dt><strong><a href="non_parametric_graph_convolution_for_re_ranking_in_recommendation_systems/">Non-parametric Graph Convolution for Re-ranking in Recommendation Systems</a></strong></dt>
<dd>
<p>将图卷积从训练期搬到测试期，用相似用户/物品构造增强对并加权聚合打分，重排平均提升 8.1% 而额外开销仅 0.5%。</p>
</dd>
<dt><strong><a href="paragon_parameter_generation_for_controllable_multi_task_recommendation/">Paragon: Parameter Generation for Controllable Multi-Task Recommendation</a></strong></dt>
<dd>
<p>用扩散模型按偏好权重条件生成适配器参数，免重训实现测试时可控多任务推荐，响应时间降低 94.6%+。</p>
</dd>
<dt><strong><a href="practical_multi_task_learning_for_rare_conversions_in_ad_tech/">Practical Multi-Task Learning for Rare Conversions in Ad Tech</a></strong></dt>
<dd>
<p>以历史CVR统计自动划分软/硬转化任务，共享底层+双塔MTL提升稀有转化预估，离线AUC+0.69%、线上CPA降2%。</p>
</dd>
<dt><strong><a href="radar_recall_augmentation_through_deferred_asynchronous_retrieval/">RADAR: Recall Augmentation through Deferred Asynchronous Retrieval</a></strong></dt>
<dd>
<p>用离线异步全复杂度精排模型对 50 倍候选池预打分并缓存 Top-200，线上绕过召回与粗排直接注入精排，Recall@200 翻倍、线上 +0.8%。</p>
</dd>
</dl>

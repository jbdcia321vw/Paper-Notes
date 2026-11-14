---
hide:
  - toc
---

# 🏛️ DASFAA 2025 推荐系统论文解读

9 篇 DASFAA 2025 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/a_situation_aware_enhancer_for_personalized_recommendation/">A Situation-aware Enhancer for Personalized Recommendation</a></strong></dt>
<dd>
<p>将情境视为交互前置条件，用可插拔双塔增强器 SARE 建模个性化情境感知与偏好影响，在 MIND/KuaiRand 上显著超越骨干与 SOTA。</p>
</dd>
<dt><strong><a href="collaborative_filtering/learning_recommender_systems_with_soft_target_a_decoupled_perspective/">Learning Recommender Systems with Soft Target: A Decoupled Perspective</a></strong></dt>
<dd>
<p>将 Softmax 软标签目标解耦为目标置信度与未观测项兴趣分布两项，配合 K-means 邻居标签传播生成软标签，在四个骨干模型三个数据集上稳定提升。</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl>
<dt><strong><a href="graph_rec/genet_unleashing_the_power_of_side_information_for_recommendation_via_hyper/">GENET: Unleashing the Power of Side Information for Recommendation via Hypergraph Pre-training</a></strong></dt>
<dd>
<p>将异构边信息统一构建为超图，通过超边预测与双层自监督对比预训练，再以 LightGCN/GRU 微调，Top-N 与序列推荐最高提升 38%，冷启动提升 175%+。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl>
<dt><strong><a href="sequential_rec/empowering_sequential_recommendation_from_collaborative_signals_and_semanti/">Empowering Sequential Recommendation from Collaborative Signals and Semantic Relatedness</a></strong></dt>
<dd>
<p>双流 Transformer 分别编码 ID 与内容序列，用用户级/物品级分层对比对齐跨模态语义鸿沟，全库排序下 NDCG@10 最高提升 15.44%。</p>
</dd>
<dt><strong><a href="sequential_rec/reformulating_sequential_recommendation_learning_dynamic_user_interest_with/">Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling</a></strong></dt>
<dd>
<p>用知识提示+推理提示将序列推荐改写为文本生成任务，GPT-2 冻结主干下在三个数据集全排序协议上取得 Recall@5 相对提升约 5%–12%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/truthsr_trustworthy_sequential_recommender_systems_via_user_generated_multi/">TruthSR: Trustworthy Sequential Recommender Systems via User-generated Multimodal Content</a></strong></dt>
<dd>
<p>用评论-图片跨模态一致性降噪 + 主观逻辑证据融合做可信序列推荐，四数据集 Full-Ranking 全面超越 MMSRec。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/a_unified_framework_for_adaptive_representation_enhancement_and_inversed_le/">A Unified Framework for Adaptive Representation Enhancement and Inversed Learning in Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>解耦共享/专属表征，用 LightGCN 域内增强 + 自注意力跨域自适应融合 + GRL 逆向约束，全库排序下 Recall@20 最高提升 8.31%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/cross_domain_causal_preference_learning_for_out_of_distribution_recommendat/">Cross-Domain Causal Preference Learning for Out-of-Distribution Recommendation</a></strong></dt>
<dd>
<p>用域对抗抽取域共享偏好、DAG 学习潜属性因果结构，缓解目标域稀疏下的 OOD 泛化，OOD 测试 HR@10 提升约 2-6%。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/separating_and_learning_latent_confounders_to_enhancing_user_preferences_mo/">Separating and Learning Latent Confounders to Enhancing User Preferences Modeling</a></strong></dt>
<dd>
<p>将前序推荐系统作为未观测混杂代理，用变分推断在隐空间解耦用户偏好与混杂，实现通用去偏，五数据集 Recall@10 全面领先。</p>
</dd>
</dl>
</div>

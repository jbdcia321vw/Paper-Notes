---
hide:
  - toc
---

# 🏛️ SIGIR-AP 2025 推荐系统论文解读

4 篇 SIGIR-AP 2025 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/limitations_of_current_evaluation_practices_for_conversational_recommender_/">Limitations of Current Evaluation Practices for Conversational Recommender Systems and the Potential of User Simulation</a></strong></dt>
<dd>
<p>批判静态数据集与 Recall@N 评估失真，提出奖励/成本效用框架与 RDL 指标，并用 LLM 用户模拟器替代真人评估。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/marco_a_cooperative_knowledge_transfer_framework_for_personalized_cross_dom/">MARCO: A Cooperative Knowledge Transfer Framework for Personalized Cross-domain Recommendations</a></strong></dt>
<dd>
<p>将多源跨域推荐建模为协作式多智能体强化学习，每源域一个 Agent 估计贡献并加熵正则，冷启动 MAE 最高降 32.6%。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/accurate_and_diverse_recommendations_via_propensity_weighted_linear_autoenc/">Accurate and Diverse Recommendations via Propensity-Weighted Linear Autoencoders</a></strong></dt>
<dd>
<p>用对数 Sigmoid 重定义 IPS 倾向分，避免幂律对热门物品过度惩罚，在 EASE 类线性自编码器上以闭式解实现覆盖率大幅提升且精度不降。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/synthetic_prefixes_to_mitigate_bias_in_real_time_neural_query_autocomplete/">Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomplete</a></strong></dt>
<dd>
<p>用无 QAC 会话的完整查询合成前缀训练数据，缓解展示偏差，并以 O(n) 成对近似实现列表级损失，线上 MRR 提升 0.92%–1.49%。</p>
</dd>
</dl>
</div>

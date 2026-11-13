---
hide:
  - toc
---

# 🏛️ ICDM 2026 推荐系统论文解读

10 篇 ICDM 2026 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl>
<dt><strong><a href="sequential_rec/hierarchical_exponential_gaussian_mixtures_for_watch_time_distribution_pred/">Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction</a></strong></dt>
<dd>
<p>用分层 skip-watch 分解与结构化初始化修复 EGMN 的方差坍缩与组件冗余，在固定骨干换头协议下提升 XAUC 与阈值事件预测，线上会话深度 +9.26%。</p>
</dd>
<dt><strong><a href="sequential_rec/revisiting_self_attentive_sequential_recommendation_beyond_the_llm_paradigm/">Revisiting Self-Attentive Sequential Recommendation Beyond the LLM Paradigm</a></strong></dt>
<dd>
<p>以 SASRec 为锚点对比推荐与语言建模的熵结构差异，诊断四条继承假设并给出数据优先议程，含同模型双视图实证。</p>
</dd>
<dt><strong><a href="sequential_rec/user_representation_via_cross_multi_source_behavior_pre_training_for_mobile/">User Representation via Cross Multi-source Behavior Pre-training for Mobile Games</a></strong></dt>
<dd>
<p>面向设备级多源多粒度行为日志，提出跨源级联掩码预训练 CM-PTM，以源→类目→App→操作逐级预测统一建模跨源依赖，离线 AUC 提升 1.05%–2.85%，线上 A/B 提升 1.28%。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (2)</h2>
<dl>
<dt><strong><a href="generative_rec/controllable_generative_recommendation_via_guided_token_refinement/">Controllable Generative Recommendation via Guided Token Refinement</a></strong></dt>
<dd>
<p>将语义 ID 生成从自回归改为并行掩码填充，用 logit 引导+置信度精炼+前缀树约束实现免训练实时多样性控制，Diversity@10 提升约 15% 且有效率保持 99.97%。</p>
</dd>
<dt><strong><a href="generative_rec/hyprq_vae_hyperbolic_item_indexing_for_long_tail_aware_generative_recommend/">HypRQ-VAE: Hyperbolic Item Indexing for Long-Tail-Aware Generative Recommender Systems</a></strong></dt>
<dd>
<p>将 RQ-VAE 语义 ID 量化搬进 Poincaré 双曲球，用 Möbius 加减替代欧氏残差，长尾 Hit@10 提升 52.7%，全库排序验证。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (3)</h2>
<dl>
<dt><strong><a href="llm_rec/autoresearch_at_production_scale_failure_modes_and_a_multi_agent_framework/">AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework</a></strong></dt>
<dd>
<p>将 LLM 自主科研范式落地生产级推荐表征学习，归纳五大失效模式并提出 prevent/persist/redirect 三智能体脚手架，Recall@6 提升 1.82 倍。</p>
</dd>
<dt><strong><a href="llm_rec/empowering_compact_llms_with_fusion_of_layer_wise_exits_for_recommendation/">Empowering Compact LLMs with Fusion of Layer-wise Exits for Recommendation</a></strong></dt>
<dd>
<p>在紧凑 LLM 每层插入预测出口并用连续 ReLU 路由器自适应融合多层打分分布，target-k hinge 损失约束激活出口数，全库排序下精度超越大模型基线且延迟仅增 &lt;1ms。</p>
</dd>
<dt><strong><a href="llm_rec/memory_has_geometry_non_uniform_geometric_memory_for_long_horizon_personali/">Memory Has Geometry: Non-Uniform Geometric Memory for Long-Horizon Personalized AI</a></strong></dt>
<dd>
<p>将长程个性化记忆重构为用户特定非均匀几何状态空间上的轨迹条件状态估计，替代扁平记录检索范式。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/an_efficient_and_effective_agentic_group_shilling_attack_on_recommender_sys/">An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems</a></strong></dt>
<dd>
<p>提出协调者驱动的角色切换多智能体群体托攻击框架 AGAS，黑盒免训练、动态策略，HR@10 相对提升 155%–198% 且显著削弱检测器。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/explain_then_forget_recommendation_unlearning_under_asymmetric_access_const/">Explain, Then Forget: Recommendation Unlearning under Asymmetric Access Constraint</a></strong></dt>
<dd>
<p>面向 MLaaS 非对称访问约束的免参数推荐遗忘框架，用因果解释定位最小支持集并做输出级分数干预，遗忘效果逼近全量重训且延迟降低 9–55 倍。</p>
</dd>
</dl>
</div>

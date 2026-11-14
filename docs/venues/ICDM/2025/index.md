---
hide:
  - toc
---

# 🏛️ ICDM 2025 推荐系统论文解读

11 篇 ICDM 2025 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl>
<dt><strong><a href="sequential_rec/efficient_sequential_recommendation_for_long_term_user_interest_via_persona/">Efficient Sequential Recommendation for Long Term User Interest Via Personalization</a></strong></dt>
<dd>
<p>用可学习 token 将长历史分段压缩为个性化专家，配合定制注意力掩码与 KV 缓存，在 HSTU/HLLM 上以约 1/4 推理开销逼近全序列精度。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl>
<dt><strong><a href="llm_rec/effectiveness_of_llms_in_temporal_user_profiling_for_recommendation/">Effectiveness of LLMs in Temporal User Profiling for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 分别生成短期/长期自然语言画像，经 SBERT 编码与注意力融合做内容推荐，Movies&amp;TV 上 Recall@10 相对提升 17%。</p>
</dd>
<dt><strong><a href="llm_rec/scaling_multimodal_search_and_recommendation_with_small_language_models_via/">Scaling Multimodal Search and Recommendation with Small Language Models via Upside-Down Reinforcement Learning</a></strong></dt>
<dd>
<p>用 Llama-3 合成 5200 万意图-提示对蒸馏进 104M nanoGPT，并以倒置强化学习实现长度/模态可控的多任务提示生成，性能逼近 8B 模型而显存仅 500MB。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/bi_nas_towards_effective_and_personalized_explanation_for_recommender_syste/">Bi-NAS: Towards Effective and Personalized Explanation for Recommender Systems via Bi-Level Neural Architecture Search</a></strong></dt>
<dd>
<p>用双层 NAS 自动搜索交叉注意力与特征交互算子，再以零样本 LLM 提示生成个性化解释，四数据集上推荐与解释指标双优。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/persopilot_an_adaptive_ai_copilot_for_transparent_contextualized_persona_cl/">PersoPilot: An Adaptive AI-Copilot for Transparent Contextualized Persona Classification and Personalized Response Generation</a></strong></dt>
<dd>
<p>双模式智能体 Copilot：PersoAgent 做上下文人格抽取与社区推荐，Analytic Tools 用 Phi-4-mini+TF-IDF 主动学习闭环做可解释人格分类，UniEval 四维均分 0.84–1.0。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/reasoning_over_user_preferences_knowledge_graph_augmented_llms_for_explaina/">Reasoning over User Preferences: Knowledge Graph-Augmented LLMs for Explainable Conversational Recommendations</a></strong></dt>
<dd>
<p>COMPASS 用图实体描述预训练+知识感知指令微调，让 LLM 生成可解释文本偏好摘要并以门控注入现有 CRS，ReDial/INSPIRED 上 HR@10 最高提升 44.76%。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/fuse_failure_aware_usage_of_subagent_evidence_for_multimodal_search_and_rec/">FUSE: Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation</a></strong></dt>
<dd>
<p>用紧凑 GDR 替代原始图像提示，系统对比 7 种上下文预算策略并以五阶段归因层定位失败，Context Compression 在意图/路由/召回/NDCG@5 全面最优且 p95 延迟降 45%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/adversarial_alignment_and_disentanglement_for_cross_domain_ctr_prediction_w/">Adversarial Alignment and Disentanglement for Cross-Domain CTR Prediction with Domain-Encompassing Features</a></strong></dt>
<dd>
<p>提出 A²DCDR，用 DC-MMD 对抗对齐 + CLUB 互信息解耦 + 特征重构，保留域不变与有价值非对齐特征，四组 Amazon 场景 HR@10 最高 +6.96，线上 CTR +7.18%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/on_predicting_post_click_conversion_rate_via_counterfactual_inference/">On Predicting Post-Click Conversion Rate via Counterfactual Inference</a></strong></dt>
<dd>
<p>用 SCM 对未点击样本做 do(C=1) 反事实推断生成转化伪标签，全空间训练 CVR 模型，离线 AUC 平均 +1.01%，线上 CVR +17.35%。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/beyond_static_llm_policies_imitation_enhanced_reinforcement_learning_for_re/">Beyond Static LLM Policies: Imitation-Enhanced Reinforcement Learning for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 轨迹作次优示范，经对抗逆强化学习提取奖励并加权模仿，离线 RL 策略在 Steam/Amazon 上累积奖励较 BiLLP 提升 13%+。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/customized_retrieval_augmented_generation_with_llm_for_debiasing_recommenda/">Customized Retrieval-Augmented Generation with LLM for Debiasing Recommendation Unlearning</a></strong></dt>
<dd>
<p>将推荐遗忘重构为 RAG 检索端过滤问题，用三种检索策略实现用户级原子遗忘，平均加速 4.5× 且保留约 90% 原模型效用。</p>
</dd>
</dl>
</div>

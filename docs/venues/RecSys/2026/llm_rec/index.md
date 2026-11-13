# 🧠 大语言模型推荐 (LLM4Rec)

**RecSys 2026** · **7** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [CIKM 2026 (5)](../../../CIKM/2026/llm_rec/) · [KDD 2026 (3)](../../../KDD/2026/llm_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/llm_rec/) · [WSDM 2026 (2)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×7 · 精排 ×2

<dl>
<dt><strong><a href="deepaffinity_long_term_aspect_preference_prediction_in_ecommerce_using_smal/">DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models</a></strong></dt>
<dd>
<p>用 135M 小模型+结构化字符串化+多任务预测头，从时序行为预测品牌/尺码/颜色等商品属性偏好，Fashion 微 F1 0.34 超 Gemini 零样本与 3B SFT。</p>
</dd>
<dt><strong><a href="improving_rare_medication_recommendation_with_counterfactual_data_augmentat/">Improving Rare Medication Recommendation with Counterfactual Data Augmentation and Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 反事实数据增强 + 病历摘要指令微调 + 逐药自回归生成，缓解罕见药长尾失效，Group-1 Jaccard 最高 +30.2%。</p>
</dd>
<dt><strong><a href="llm_reasoning_for_subjective_tasks_failure_modes_mitigation_and_dynamic_rea/">LLM Reasoning for Subjective Tasks: Failure Modes, Mitigation, and Dynamic Reasoning Routing</a></strong></dt>
<dd>
<p>揭示主观验证任务上显式推理与标准 RLVR 会引发推理崩溃，提出条件长度惩罚奖励与人格路由蓝图，Netflix 四任务 macro-F1 最高回升至 0.851。</p>
</dd>
<dt><strong><a href="pairwise_ranking_outperforms_single_action_rl_for_offline_explanation_selec/">Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson</a></strong></dt>
<dd>
<p>将 LLM 解释生成离线化为冻结候选池，用 CPU 端 LambdaRank 逐候选打分选择，F1=0.500 超越 G-Refer/XRec 且延迟&lt;100ms。</p>
</dd>
<dt><strong><a href="recap_feedback_driven_streaming_semantic_user_profiles_for_short_video_reco/">RECAP: Feedback-Driven Streaming Semantic User Profiles for Short-Video Recommendation</a></strong></dt>
<dd>
<p>将流式结构化用户画像视为有界记忆状态，用 LLM 语义差分+确定性状态机维护，并以 LLM 判官清洗隐式反馈、双塔评估器打分作 GRPO 奖励闭环优化，快手离线 uAUC +0.0084、线上人均使用时长 +0.139%。</p>
</dd>
<dt><strong><a href="self_evolving_recommendation_system_end_to_end_autonomous_model_optimizatio/">Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</a></strong></dt>
<dd>
<p>用 Gemini 双智能体（快循环提名+慢循环上线）自动改写优化器/架构/奖励函数，在 YouTube 线上 A/B 取得显著北极星指标提升。</p>
</dd>
<dt><strong><a href="structure_preserving_projection_for_mitigating_modality_bias_in_llm_based_s/">Structure-Preserving Projection for Mitigating Modality Bias in LLM-Based Sequential Recommendation</a></strong></dt>
<dd>
<p>用余弦相似度保持与序列对比保持双损失约束协同嵌入投影，缓解 LLM 推荐中的模态偏置，20 候选协议下 HR@1 最高提升约 12pt。</p>
</dd>
</dl>

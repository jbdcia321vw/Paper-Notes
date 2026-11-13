# 🗣️ 对话推荐与智能体仿真

**SIGIR 2026** · **8** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** 重排 ×4 · 端到端生成式 ×2

<dl>
<dt><strong><a href="a_standardized_re_evaluation_of_conversational_recommender_systems_on_the_r/">A Standardized Re-evaluation of Conversational Recommender Systems on the ReDial Dataset</a></strong></dt>
<dd>
<p>统一复现 7 个 CRS 模型，揭示 Recall@1 复现粒度鸿沟、约 50% 精度来自重复捷径、增益多源于 LLM 骨干而非架构创新。</p>
</dd>
<dt><strong><a href="bridging_passive_and_active_enhancing_conversation_starter_recommendation_v/">Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling</a></strong></dt>
<dd>
<p>用对抗分布对齐+RQ-VAE语义离散化把用户主动查询引入对话开场白推荐，打破曝光-点击回音室，线上活跃天数+0.04%。</p>
</dd>
<dt><strong><a href="conv_finre_a_conversational_and_longitudinal_benchmark_for_utility_grounded/">Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation</a></strong></dt>
<dd>
<p>用逆优化从纵向决策轨迹反推用户风险偏好，构建四视图多视角金融推荐基准，揭示 LLM 效用对齐与行为模仿的根本张力。</p>
</dd>
<dt><strong><a href="meituan_merchant_business_diagnosis_via_policy_guided_dual_process_user_sim/">Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation</a></strong></dt>
<dd>
<p>从行为轨迹挖掘决策策略作为共享对齐层，锚定 LLM 推理与 GBDT 拟合双过程分支，群体仿真误差降至 8.80%。</p>
</dd>
<dt><strong><a href="recquest_towards_estimating_user_domain_knowledge_in_conversational_recomme/">RecQuest: Towards Estimating User Domain Knowledge in Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出游戏化数据采集协议 RecQuest 与知识估计新任务，用客观问卷替代自评建立 ground truth，并给出 LLM 零/少样本与微调基线。</p>
</dd>
<dt><strong><a href="task_aware_automated_user_profile_generation_for_recommendation_simulation_/">Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models</a></strong></dt>
<dd>
<p>提出免训练三阶段画像生成框架 APG4RecSim，用语义合并+反事实因果映射将交互历史转为任务对齐画像，在判别/排序/评分三任务上取得最佳综合表现。</p>
</dd>
<dt><strong><a href="trace_a_conversational_framework_for_sustainable_tourism_recommendation_wit/">TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations</a></strong></dt>
<dd>
<p>多智能体 LLM 框架以澄清提问+反事实解释实现可持续旅游推荐的非强制绿色引导，24 人用户研究验证。</p>
</dd>
<dt><strong><a href="when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 分层 MoE 架构，显式建模“何时问、怎么问”的阶段感知偏好引导策略，Recall@10 相对最优基线提升 14.6%。</p>
</dd>
</dl>

# 🗣️ 对话推荐与智能体仿真

**ACL / EMNLP (RecSys & CRS) 2025** · **8** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../2026/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** ACL 2025 ×4 · 端到端生成式 ×3 · LLM Agent ×2 · 重排 ×2

<dl>
<dt><strong><a href="a_survey_on_llm_powered_agents_for_recommender_systems/">A Survey on LLM-powered Agents for Recommender Systems</a></strong></dt>
<dd>
<p>系统综述 LLM Agent 推荐的三范式（推荐/交互/仿真）与四模块架构，梳理数据集与评测协议并指出统一评测缺失。</p>
</dd>
<dt><strong><a href="empowering_retrieval_based_conversational_recommendation_with_contrasting_u/">Empowering Retrieval-based Conversational Recommendation with Contrasting User Preferences</a></strong></dt>
<dd>
<p>用 LLM 抽取并扩增 like/dislike 对比偏好，以三路相似度线性打分做稠密检索式对话推荐，INSPIRED 上 Recall@10 相对提升 99.72%。</p>
</dd>
<dt><strong><a href="enhancing_recommendation_explanations_through_user_centric_refinement/">Enhancing Recommendation Explanations through User-Centric Refinement</a></strong></dt>
<dd>
<p>提出 RefineX 多智能体推理期精炼框架，以 plan-then-refine 与分层反思将 PETER 解释在事实性/个性化/连贯性三维度大幅提升。</p>
</dd>
<dt><strong><a href="exploring_the_potential_of_llms_as_personalized_assistants_dataset_evaluati/">Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis</a></strong></dt>
<dd>
<p>提出 HiCUPID 个性化助手基准（1.5k 合成用户、17k token 长对话、单/多跳 QA），并蒸馏 Llama-3.2 评估器对齐人类偏好，揭示 SFT+DPO 最优而纯 DPO 崩溃。</p>
</dd>
<dt><strong><a href="iagent_llm_agent_as_a_shield_between_user_and_recommender_systems/">iAgent: LLM Agent as a Shield between User and Recommender Systems</a></strong></dt>
<dd>
<p>提出 user-agent-platform 新范式，用个体指令感知 LLM Agent 在用户与平台间做重排护盾，四数据集平均提升 16.6%。</p>
</dd>
<dt><strong><a href="muse_a_multimodal_conversational_recommendation_dataset_with_scenario_groun/">Muse: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles</a></strong></dt>
<dd>
<p>首个多模态对话推荐数据集，用场景锚定用户画像的多智能体框架合成 7K 段图文对话，并验证三种 MLLM 的可学习性。</p>
</dd>
<dt><strong><a href="refining_text_generation_for_realistic_conversational_recommendation_via_di/">Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization</a></strong></dt>
<dd>
<p>用 DPO 对齐 LLM 生成的对话摘要与物品推荐信息，使评分器更准，Tabidachi 上 HR@1 提升 21.3%。</p>
</dd>
<dt><strong><a href="towards_personalized_conversational_sales_agents_contextual_user_profiling_/">Towards Personalized Conversational Sales Agents: Contextual User Profiling for Strategic Action</a></strong></dt>
<dd>
<p>提出 CSALES 对话销售任务与 CSUSER 真实画像用户模拟器，CSI 智能体通过上下文画像驱动统一动作空间，SR 提升约 15-20pt、SWR 提升约 5-10pt。</p>
</dd>
</dl>

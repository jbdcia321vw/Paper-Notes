# 🗣️ 对话推荐与智能体仿真

**WWW (TheWebConf) 2025** · **6** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (5)](../../2026/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/)

🔥 **高频主题：** WWW 2025 ×6 · 端到端生成式 ×2 · 重排 ×2

<dl>
<dt><strong><a href="a_llm_based_controllable_scalable_human_involved_user_simulator_framework_f/">A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>以插件管理器分阶段管控 LLM 用户模拟器，通过记忆隔离与偏好二分抑制数据泄漏，在 ReDial/OpenDialKG/MovieLens 上显著提升 CRS 评估可信度。</p>
</dd>
<dt><strong><a href="agentsociety_challenge_designing_llm_agents_for_user_modeling_and_recommend/">AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms</a></strong></dt>
<dd>
<p>首个 WWW LLM Agent 双赛道竞赛：Yelp/Amazon/Goodreads 三源数据 + 交互式模拟器，295 队参赛，Track1/2 最终阶段提升 9.1%/15.9%，并验证模拟 groundtruth 与真实分布强相关。</p>
</dd>
<dt><strong><a href="bridging_conversational_and_collaborative_signals_for_conversational_recomm/">Bridging Conversational and Collaborative Signals for Conversational Recommendation</a></strong></dt>
<dd>
<p>构建 Reddit-ML32M 跨源数据集，用 SASRec 协同嵌入对 GPT-3.5 零样本推荐做相似度重排，H@5 提升 12.32%。</p>
</dd>
<dt><strong><a href="collaborative_retrieval_for_large_language_model_based_conversational_recom/">Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems</a></strong></dt>
<dd>
<p>用 EASE 协同检索 + 两步 LLM 反思重排为黑盒 LLM 注入 CF 知识，Reddit-v2/Redial 全库排序 Recall@20 显著超越 Zero-shot LLM。</p>
</dd>
<dt><strong><a href="disentangling_likes_and_dislikes_in_personalized_generative_explainable_rec/">Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation</a></strong></dt>
<dd>
<p>用 LLM 重构正负观点分离的可解释推荐数据集，提出情感匹配与正负内容相似度指标，并证明将预测评分作为输入可显著提升情感感知解释生成。</p>
</dd>
<dt><strong><a href="recusersim_a_realistic_and_diverse_user_simulator_for_evaluating_conversati/">RecUserSim: A Realistic and Diverse User Simulator for Evaluating Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出四模块 LLM Agent 用户模拟器，以有限理性三层 Rating-Action-Response 机制实现真实角色扮演、多样人群覆盖与显式打分，并落地华为 Celia 餐饮助手。</p>
</dd>
</dl>

# 🗣️ 对话推荐与智能体仿真

**SIGIR 2025** · **7** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [SIGIR 2026 (8)](../../2026/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** 重排 ×3 · 端到端生成式 ×2

<dl>
<dt><strong><a href="agentcf_memory_enhanced_llm_based_agents_for_popularity_aware_cross_domain_/">AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations</a></strong></dt>
<dd>
<p>以双层记忆+两步融合+兴趣组共享记忆，令 LLM 用户智能体在跨域场景显式建模流行度影响，采样排序 MRR 平均超 AgentCF 14.6%。</p>
</dd>
<dt><strong><a href="agentic_feedback_loop_modeling_improves_recommendation_and_user_simulation/">Agentic Feedback Loop Modeling Improves Recommendation and User Simulation</a></strong></dt>
<dd>
<p>用记忆驱动的推荐智能体-用户智能体多轮反馈闭环，同时提升下一物品推荐与用户行为模拟，并验证不放大流行度/位置偏差。</p>
</dd>
<dt><strong><a href="mscrs_multi_modal_semantic_graph_prompt_learning_framework_for_conversation/">MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>构建协同/文本/图像三模态语义图并以共享嵌入融合，图提示注入 DialoGPT，ReDial 与 INSPIRED 全指标 SOTA。</p>
</dd>
<dt><strong><a href="pscon_product_search_through_conversations/">PSCon: Product Search Through Conversations</a></strong></dt>
<dd>
<p>提出人-人教练式采集协议构建双市场双语真实 CPS 数据集 PSCon，定义六子任务流水线并给出 Transformer 统一基准模型。</p>
</dd>
<dt><strong><a href="pub_an_llm_enhanced_personality_driven_user_behaviour_simulator_for_recomme/">PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation</a></strong></dt>
<dd>
<p>用大五人格驱动 LLM Agent 生成合成交互日志，Jaccard 相似度 0.31 逼近真实序列，并复现 7 种推荐算法的 nDCG 排序趋势。</p>
</dd>
<dt><strong><a href="search_based_interaction_for_conversation_recommendation_via_generative_rew/">Search-Based Interaction For Conversation Recommendation via Generative Reward Model Based Simulated User</a></strong></dt>
<dd>
<p>用生成式奖励模型构建无标签模拟用户，通过属性评论与生成式打分驱动束搜索交互，INSPIRED 上 Recall@10 相对最优基线提升 83%。</p>
</dd>
<dt><strong><a href="simulating_before_planning_constructing_intrinsic_user_world_model_for_user/">Simulating Before Planning: Constructing Intrinsic User World Model for User-Tailored Dialogue Policy Planning</a></strong></dt>
<dd>
<p>用扩散模型刻画用户画像、布朗桥预判用户反馈，构建内在用户世界模型以定制对话策略，P4G/ESConv 上 SR 显著提升。</p>
</dd>
</dl>

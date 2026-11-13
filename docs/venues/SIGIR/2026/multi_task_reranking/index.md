# ⚖️ 多任务学习与列表级重排

**SIGIR 2026** · **4** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [CIKM 2026 (4)](../../../CIKM/2026/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2026 (5)](../../../RecSys/2026/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/)

🔥 **高频主题：** 重排 ×2

<dl>
<dt><strong><a href="equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong></dt>
<dd>
<p>用残差量化码本近似 Target Attention，将交互式建模引入粗排，AUC 0.799、线上 GMV +7.57%。</p>
</dd>
<dt><strong><a href="one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层消除候选顺序依赖，单次前向实现近完美排列不变性，nDCG 仅微降。</p>
</dd>
<dt><strong><a href="towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong></dt>
<dd>
<p>提出 EASQ，用 LoRA 独立参数通路 + MoE 解耦多任务 + 在线 DPO，将极稀疏问卷满意度信号端到端对齐进短视频精排在线学习。</p>
</dd>
<dt><strong><a href="uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong></dt>
<dd>
<p>用后验价值对齐把抽象混合分映射到有效完播率语义空间，再以独立线性 boosting 解耦多业务加权，线上 A/B 全指标正向。</p>
</dd>
</dl>

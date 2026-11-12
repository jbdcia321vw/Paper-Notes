# ⚖️ 多任务学习与列表级重排

**SIGIR 2026** · **4** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

🔥 **高频主题：** 重排 ×2

<dl>
<dt><strong><a href="equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong></dt>
<dd>
<p>用残差量化把 Target Attention 近似搬进粗排，以码本查表替代逐候选交互，淘宝线上 CTR/CVR/GMV 全面显著提升</p>
</dd>
<dt><strong><a href="one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层面消除 LLM 列表重排的候选顺序敏感性，单次前向即达近完美排列稳健性</p>
</dd>
<dt><strong><a href="towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong></dt>
<dd>
<p>提出 EASQ，用 LoRA 独立参数通路 + MoE 解耦多任务 + 在线 DPO，将极稀疏问卷满意度信号端到端对齐进短视频精排在线学习。</p>
</dd>
<dt><strong><a href="uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong></dt>
<dd>
<p>提出后验价值对齐+独立线性提升的统一流量分配框架，解耦多业务加权方案，线上 A/B 提升 VV +1.69%、Valued VV +3.07%。</p>
</dd>
</dl>

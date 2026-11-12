# ⚖️ 多任务学习与列表级重排 (Multi-Task Learning & Listwise Re-ranking)

> **研究范畴**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

当前方向共收录 **7** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>提出 filter-then-fuse 显式稀疏范式，用静态随机过滤与迭代竞争稀疏(ICS)替代稠密全连接，在十亿级工业数据上实现可扩展的CTR提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong></dt>
<dd>
<p>提出 CAS/CTS/CMS 三模块协同增强双塔召回，兼容在线学习，离线三数据集 AUC 提升，线上广告收入最高 +8.36% 且保持毫秒级延迟。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong></dt>
<dd>
<p>用残差量化把 Target Attention 近似搬进粗排，以码本查表替代逐候选交互，淘宝线上 CTR/CVR/GMV 全面显著提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/follow_the_trace_exploiting_post_click_trajectories_for_online_delayed_conv/">Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将点击后行为演化建模为反馈轨迹，用贝叶斯分解融合静态意图与动态轨迹，并以可靠性门控的回溯补全模块作为模型无关增强器，缓解延迟反馈假阴性偏差。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层面消除 LLM 列表重排的候选顺序敏感性，单次前向即达近完美排列稳健性</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong></dt>
<dd>
<p>提出后验价值对齐+独立线性提升的统一流量分配框架，解耦多业务加权方案，线上 A/B 提升 VV +1.69%、Valued VV +3.07%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/unifying_search_and_recommendation_in_llms_via_gradient_multi_subspace_tuni/">Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning</a></strong></dt>
<dd>
<p>GEMS 用多子空间梯度分解缓解 S&amp;R 梯度冲突，并用零空间投影保留通用知识，在 Qilin/Amazon 上大幅超越 PEFT 与统一 S&amp;R 基线。</p>
</dd>
</dl>

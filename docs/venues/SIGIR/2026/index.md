---
hide:
  - toc
---

# 🏛️ SIGIR 2026 推荐系统论文解读

5 篇 SIGIR 2026 论文解读，覆盖 5 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong></dt>
<dd>
<p>针对本地社区平台地理曝光约束下的“不可能负例”问题，提出区域约束批采样 RCBS，仅改批构造不改模型与损失，离线与线上 A/B 均显著提升召回与排序。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl>
<dt><strong><a href="generative_rec/gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong></dt>
<dd>
<p>提出线性时间双向注意力 GBLA，以 Conv1D+Key Gating+Gated RMSNorm 三件套替换编码器自注意力，混合 1:2 堆叠在 Yandex Music 上持平双向 SA 质量，L=32768 单层提速 8.2×。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl>
<dt><strong><a href="llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在 Full-Ranking 留一法协议下 Recall@20 最高提升 12.4%。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 混合专家架构，将对话推荐建模为结构化动作空间上的序贯决策，实现阶段感知的偏好诱导策略动态切换。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/an_epistemic_position_based_click_model_from_interactions_to_epistemic_dist/">An Epistemic Position-Based Click Model: From Interactions to Epistemic Distributions of Relevance and Bias</a></strong></dt>
<dd>
<p>首个上下文认知型位置点击模型，用 Beta 分布刻画相关性与位置偏置的认知不确定性，提出自归一化与位置条件化梯度估计实现稳定优化</p>
</dd>
</dl>
</div>

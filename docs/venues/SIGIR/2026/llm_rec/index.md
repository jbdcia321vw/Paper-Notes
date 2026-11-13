# 🧠 大语言模型推荐 (LLM4Rec)

**SIGIR 2026** · **6** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [CIKM 2026 (5)](../../../CIKM/2026/llm_rec/) · [KDD 2026 (3)](../../../KDD/2026/llm_rec/) · [WSDM 2026 (2)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×4 · 生成式端到端 ×2

<dl>
<dt><strong><a href="factorized_latent_reasoning_for_llm_based_recommendation/">Factorized Latent Reasoning for LLM-based Recommendation</a></strong></dt>
<dd>
<p>将 LLM 隐空间推理分解为 K 个正交偏好因子，配合 GRPO 隐空间强化学习，在 Amazon 四域上以全库排序取得稳定提升。</p>
</dd>
<dt><strong><a href="lwgr_lagrangian_constrained_personalized_world_knowledge_for_generative_rec/">LWGR: Lagrangian-Constrained Personalized World Knowledge for Generative Recommendation</a></strong></dt>
<dd>
<p>用 OPQ 并行码本生成用户个性化软指令提取 LLM 世界知识，并以拉格朗日原始-对偶约束可控融合进生成式推荐，离线最高 +11.23%，线上广告收入 +1.35%</p>
</dd>
<dt><strong><a href="recgpt_mobile_on_device_large_language_models_for_user_intent_understanding/">RecGPT-Mobile: On-Device Large Language Models for User Intent Understanding in Taobao Feed Recommendation</a></strong></dt>
<dd>
<p>将 Qwen3-0.6B 量化后部署于手机端作为意图理解 Agent，把隐式行为翻译为显式搜索意图 query，配合自适应 Prompt 与意图漂移触发机制，线上平均 +1.8% CLICK / +2.7% PAY / +2.5% GMV。</p>
</dd>
<dt><strong><a href="spectran_spectral_aware_transformer_based_adapter_for_llm_enhanced_sequenti/">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用谱域 Transformer 适配器替代静态 SVD 截断，全谱注意力聚合 LLM 语义嵌入，四数据集三骨干平均提升 9.17%</p>
</dd>
<dt><strong><a href="think_when_needed_model_aware_reasoning_routing_for_llm_based_ranking/">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></strong></dt>
<dd>
<p>提出轻量即插即用路由器，基于排序感知特征与模型感知难度信号，逐实例决定是否触发推理，实现精度-效率帕累托最优。</p>
</dd>
<dt><strong><a href="verifiable_reasoning_for_llm_based_generative_recommendation/">Verifiable Reasoning for LLM-based Generative Recommendation</a></strong></dt>
<dd>
<p>提出 reason-verify-recommend 范式，用混合验证器+代理偏好预测+单调性正则，在四个数据集上显著提升生成式推荐效果。</p>
</dd>
</dl>

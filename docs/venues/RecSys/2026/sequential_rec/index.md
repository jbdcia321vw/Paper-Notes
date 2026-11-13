# ⏱️ 序列与会话推荐

**RecSys 2026** · **3** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [KDD 2026 (4)](../../../KDD/2026/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2026 (2)](../../../WWW/2026/sequential_rec/)

<dl>
<dt><strong><a href="do_sequential_recommendation_benchmarks_really_require_higher_order_sequenc/">Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?</a></strong></dt>
<dd>
<p>用两个不学高阶序列表征的近期加权成对探针（SeqRules/PCTM）在 eSASRec 协议下反超 Transformer，质疑主流基准的容量测量效度。</p>
</dd>
<dt><strong><a href="recrec_latent_interests_recursive_reasoning_for_sequential_recommendation/">RecRec: Latent Interests Recursive Reasoning for Sequential Recommendation</a></strong></dt>
<dd>
<p>解耦推理态与预测态、以 R 个潜在兴趣做递归推理的免 RL 序列推荐框架，Full-Ranking 下 NDCG@10 最高提升 14.8%。</p>
</dd>
<dt><strong><a href="zero_observation_user_reactivation_with_gap_driven_dimensional_gating/">Zero-Observation User Reactivation with Gap-Driven Dimensional Gating</a></strong></dt>
<dd>
<p>定义零观测用户唤醒问题，提出冻结骨干的逐维门控插件 DeltaGate，以 Δt 与历史表征联合路由个性化历史与全局先验，&gt;365d 桶 Hit@10 提升 33%–84%。</p>
</dd>
</dl>

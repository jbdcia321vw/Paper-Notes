---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2025 推荐系统论文解读

4 篇 IJCAI (RecSys Track) 2025 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl>
<dt><strong><a href="sequential_rec/device_cloud_collaborative_correction_for_on_device_recommendation/">Device-Cloud Collaborative Correction for On-Device Recommendation</a></strong></dt>
<dd>
<p>端侧 TTT 自校正网络 + 云端全局校正网络，以 RNN 级开销逼近 Transformer 精度，UAUC 最高提升 4.56%。</p>
</dd>
<dt><strong><a href="sequential_rec/towards_automatic_sampling_of_user_behaviors_for_sequential_recommender_sys/">Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems</a></strong></dt>
<dd>
<p>提出 AutoSAM，用轻量采样器+强化学习多目标奖励自适应丢弃低信息历史行为，在四个真实数据集全库排序下 NDCG@10 平均提升约 8%。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl>
<dt><strong><a href="generative_rec/flow_matching_based_sequential_recommender_model/">Flow Matching based Sequential Recommender Model</a></strong></dt>
<dd>
<p>用直线流轨迹+无噪预测目标+ODE确定性采样替代扩散式序列推荐，四数据集全库排序平均提升6.53%</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/single_node_trigger_backdoor_attacks_in_graph_based_recommendation_systems/">Single-Node Trigger Backdoor Attacks in Graph-Based Recommendation Systems</a></strong></dt>
<dd>
<p>提出单节点触发器后门攻击 SNT-BA，仅注入 1 个虚假用户即可让目标物品在 99% 目标用户 Top-10 中曝光，同时 HR@20 仅降约 5%。</p>
</dd>
</dl>
</div>

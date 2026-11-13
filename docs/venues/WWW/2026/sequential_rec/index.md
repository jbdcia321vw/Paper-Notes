# ⏱️ 序列与会话推荐

**WWW (TheWebConf) 2026** · **2** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [KDD 2026 (4)](../../../KDD/2026/sequential_rec/) · [SIGIR 2026 (9)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/sequential_rec/)

🔥 **高频主题：** WWW 2026 ×2

<dl>
<dt><strong><a href="generative_archetype_grounded_item_representations_for_sequential_recommend/">Generative Archetype-Grounded Item Representations for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 生成物品“目标人群原型”文本并单次前向抽取 prefill/decoding 双嵌入，再以行为关联分数调制排斥力做校准，Full-Ranking 下 HR@10 最高 +6.95%。</p>
</dd>
<dt><strong><a href="make_it_long_keep_it_fast_end_to_end_10k_long_user_behavior_sequence_modeli/">Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation</a></strong></dt>
<dd>
<p>以单查询目标-历史交叉注意力(STCA)将长序列建模降至 O(L)，配合请求级批处理(RLB)与稀疏训练稠密推理，在抖音十亿级流量实现 10K 序列端到端排序。</p>
</dd>
</dl>

# ⏱️ 序列与会话推荐 (Sequential & Session-Based Recommendation)

> **研究范畴**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

当前方向共收录 **2** 篇顶会/顶刊论文解读：

| 会议/期刊 | 年份 | 论文标题 (点击查看 5 分钟精读) | 原文与开源代码 | 核心速览 |
| :---: | :---: | :--- | :---: | :--- |
| `SIGIR` | 2026 | [**ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation**](../venues/SIGIR/2026/llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend.md) | [📄 原文](https://arxiv.org/abs/2605.29322v1) · [💻 代码](https://github.com/DCheol/ACE) | [SIGIR 2026][LLM4Rec] 用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在 Full-Ranking 留一法协议下 Recall@20 最高提升 12.4%。 |
| `SIGIR` | 2026 | [**Gated Bidirectional Linear Attention for Generative Retrieval**](../venues/SIGIR/2026/generative_rec/gated_bidirectional_linear_attention_for_generative_retrieval.md) | [📄 原文](https://arxiv.org/abs/2606.07317v2) · [💻 代码](https://github.com/matfu-pixel/GRID) | [SIGIR 2026][生成式推荐与离散语义 ID] 提出线性时间双向注意力 GBLA，以 Conv1D+Key Gating+Gated RMSNorm 三件套替换编码器自注意力，混合 1:2 堆叠在 Yandex Music 上持平双向 SA 质量，L=32768 单层提速 8.2×。 |

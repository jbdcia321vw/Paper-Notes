# 🕸️ 图神经网络与知识图谱推荐

**KDD 2025** · **3** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [KDD 2026 (1)](../../2026/graph_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

🔥 **高频主题：** 召回 ×3

<dl>
<dt><strong><a href="an_automatic_graph_construction_framework_based_on_large_language_models_fo/">An Automatic Graph Construction Framework based on Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 逐点生成语义向量 + 残差量化抽取隐因子作额外节点，O(N) 构图并设计元路径聚合，Full-Ranking 下 NDCG@10 最高提升 43.57%，华为广告 A/B 提升 RPM 2.69%。</p>
</dd>
<dt><strong><a href="collaborative_filtering_meets_spectrum_shift_connecting_user_item_interacti/">Collaborative Filtering Meets Spectrum Shift: Connecting User-Item Interaction with Graph-Structured Side Information</a></strong></dt>
<dd>
<p>揭示增广图频谱右移现象，提出仅含平移/缩放两因子的 SSC 校正，即插即用提升 LightGCN/JGCF 最高 23%，零额外开销。</p>
</dd>
<dt><strong><a href="lightkg_efficient_knowledge_aware_recommendations_with_simplified_gnn_archi/">LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture</a></strong></dt>
<dd>
<p>用标量对编码关系+线性聚合+免子图对比层，在稀疏场景超越12个KGRS基线5.8%并省84.3%训练时间</p>
</dd>
</dl>

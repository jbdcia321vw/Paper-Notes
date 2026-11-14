# 🤝 协同过滤与双塔召回

**SIGIR 2025** · **3** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [SIGIR 2026 (6)](../../2026/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** 召回 ×2

<dl>
<dt><strong><a href="a_learnable_fully_interacted_two_tower_model_for_pre_ranking_system/">A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System</a></strong></dt>
<dd>
<p>用可学习 Item Meta Matrix 实现显式早期交互、行/列 FC 相似度矩阵实现晚期交互，在保持双塔解耦推理效率下 AUC 相对提升 4.6%–14.3%。</p>
</dd>
<dt><strong><a href="graph_spectral_filtering_with_chebyshev_interpolation_for_recommendation/">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a></strong></dt>
<dd>
<p>用切比雪夫插值逼近非线性图滤波器，直接以原始交互为信号，突破 LightGCN 的截断线性低通瓶颈，Full-Ranking 下全面 SOTA 且推理快 8.1 倍。</p>
</dd>
<dt><strong><a href="ira_adaptive_interest_aware_representation_and_alignment_for_personalized_m/">IRA: Adaptive Interest-aware Representation and Alignment for Personalized Multi-interest Retrieval</a></strong></dt>
<dd>
<p>用可累积/剪枝的文本化 Interest Unit 表征多兴趣，语义对齐替代点击信号做 ANN 召回，离线 H@20 提升 20%，线上点击 +5.4%。</p>
</dd>
</dl>

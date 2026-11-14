# ✨ 生成式推荐与离散语义 ID

**SIGIR 2025** · **4** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [SIGIR 2026 (19)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式端到端 ×3 · 生成式推荐 ×2 · 生成式推荐与离散语义ID ×2

<dl>
<dt><strong><a href="addressing_missing_data_issue_for_diffusion_based_recommendation/">Addressing Missing Data Issue for Diffusion-based Recommendation</a></strong></dt>
<dd>
<p>用双侧 Thompson 采样在引导序列中模拟缺失数据，以一致性正则化让扩散推荐模型外推抵抗真实缺失，YooChoose NDCG@20 提升 26.95%。</p>
</dd>
<dt><strong><a href="generative_recommender_with_end_to_end_learnable_item_tokenization/">Generative Recommender with End-to-End Learnable Item Tokenization</a></strong></dt>
<dd>
<p>提出 ETEGRec 双编码器-解码器框架，用序列-物品对齐与偏好-语义对齐将 RQ-VAE 分词器与 T5 推荐器端到端联合优化，全库排序下 Recall@10 相对 LETTER 提升约 5%。</p>
</dd>
<dt><strong><a href="killing_two_birds_with_one_stone_unifying_retrieval_and_ranking_with_a_sing/">Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model</a></strong></dt>
<dd>
<p>将召回与排序统一为单生成模型的位置级序列生成任务，配合排序驱动增强器与梯度自适应加权实现协同与同步优化，全库排序下召回排序双提升。</p>
</dd>
<dt><strong><a href="order_agnostic_identifier_for_large_language_model_based_generative_recomme/">Order-agnostic Identifier for Large Language Model-based Generative Recommendation</a></strong></dt>
<dd>
<p>提出无序集合标识符范式，用CF+语义多token集合表示物品并同时生成，T5上Recall@10平均提升约50%，推理提速8-18倍。</p>
</dd>
</dl>

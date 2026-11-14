# ⏱️ 序列与会话推荐

**ICLR / NeurIPS / ICML (RecSys) 2025** · **4** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** NeurIPS ×3 · 召回 ×2

<dl>
<dt><strong><a href="long_sequence_recommendation_models_need_decoupled_embeddings/">Long-Sequence Recommendation Models Need Decoupled Embeddings</a></strong></dt>
<dd>
<p>揭示长序列推荐中注意力与表征共享嵌入导致的梯度支配与冲突，提出双嵌入表 DARE，AUC 最高提升 9‰ 并加速检索 50%。</p>
</dd>
<dt><strong><a href="panther_generative_pretraining_beyond_language_for_sequential_user_behavior/">PANTHER: Generative Pretraining Beyond Language for Sequential User Behavior Modeling</a></strong></dt>
<dd>
<p>将支付交易视为行为 token 做生成式预训练，SPRM 卷积+原型交叉注意力捕捉周期模式，用户画像对比学习，微信支付线上欺诈召回 +38.6%。</p>
</dd>
<dt><strong><a href="semantic_retrieval_augmented_contrastive_learning_for_sequential_recommenda/">Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义检索构建用户/物品候选池并配可学习样本合成器，替代随机增广与 K-Means 硬规则，全库排序下 HR@20 最高提升 11.82%。</p>
</dd>
<dt><strong><a href="tv_rec_time_variant_convolutional_filter_for_sequential_recommendation/">TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation</a></strong></dt>
<dd>
<p>用图信号处理中的时变图滤波器替代固定卷积核与自注意力，在6个基准上全库排序平均提升7.49%且推理更快。</p>
</dd>
</dl>

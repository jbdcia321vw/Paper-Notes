# 🔄 自监督对比学习与交互去噪

**CIKM 2025** · **3** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../../WWW/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/self_supervised_denoising/)

🔥 **高频主题：** 召回 ×3

<dl>
<dt><strong><a href="a_self_supervised_mixture_of_experts_framework_for_multi_behavior_recommend/">A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation</a></strong></dt>
<dd>
<p>揭示多行为推荐在已访问/未访问物品上的性能鸿沟，提出双专家 MoE + 专家自适应 SSL（访问过滤/新颖性推断对比 + 行为生成）框架，全库排序下 HR@20 最高提升 65.46%。</p>
</dd>
<dt><strong><a href="empowering_denoising_sequential_recommendation_with_large_language_model_em/">Empowering Denoising Sequential Recommendation with Large Language Model Embeddings</a></strong></dt>
<dd>
<p>用 LLM 语义嵌入与协同嵌入做跨模态兴趣对齐，缓解冷门物品过度去噪，四数据集全排序 HR/NDCG 显著提升。</p>
</dd>
<dt><strong><a href="generative_data_augmentation_in_graph_contrastive_learning_for_recommendati/">Generative Data Augmentation in Graph Contrastive Learning for Recommendation</a></strong></dt>
<dd>
<p>用 VAE 式生成器自适应生成噪声替代固定随机扰动构造对比视图，并引入物品互补矩阵，在三个稀疏数据集上 Full-Ranking 提升约 3–6%。</p>
</dd>
</dl>

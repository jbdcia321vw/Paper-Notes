# 🔄 自监督对比学习与交互去噪 (Self-Supervised, Contrastive & Denoising Rec)

> **研究范畴**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

当前方向共收录 **5** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 自适应学习用户级增强，在噪声序列推荐上 Full-Ranking 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong></dt>
<dd>
<p>针对本地社区平台地理曝光约束下的“不可能负例”问题，提出区域约束批采样 RCBS，仅改批构造不改模型与损失，离线与线上 A/B 均显著提升召回与排序。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结+可学习双网络的模块化集成模拟指数级多样性，推理仅用单网络实现零开销的集成级序列推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致SR模型评估失真，系统分析SSS生效条件与机理并给出公平评估建议</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cold_start_rec/sparse_contrastive_learning_for_content_based_cold_item_recommendation/">Sparse Contrastive Learning for Content-Based Cold Item Recommendation</a></strong></dt>
<dd>
<p>用 α-entmax 稀疏化 sampled softmax 训练纯内容编码器，以物品-物品相似度建模冷启动，并支持内容空间知识蒸馏，冷物品排序与曝光公平性双赢。</p>
</dd>
</dl>

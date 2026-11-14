# ⏱️ 序列与会话推荐

**SIGIR 2025** · **12** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [SIGIR 2026 (15)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** 召回 ×7 · 精排 ×3 · 重排 ×2

<dl>
<dt><strong><a href="ai_guided_accelerator_for_search_experience/">AI Guided Accelerator For Search Experience</a></strong></dt>
<dd>
<p>挖掘电商查询转移链并做意图一致性过滤，再用 Solar-10B 指令微调生成多样化替代收敛查询，线上 CTR +32.2%、转化 +38.3%。</p>
</dd>
<dt><strong><a href="alphafuse_learn_id_embeddings_for_sequential_recommendation_in_null_space_o/">AlphaFuse: Learn ID Embeddings for Sequential Recommendation in Null Space of Language Embeddings</a></strong></dt>
<dd>
<p>将语言嵌入 SVD 分解为语义丰富行空间与语义稀疏零空间，在裁剪后的零空间内学习 ID 嵌入，无需适配器即插即用地融合语义与协同信号。</p>
</dd>
<dt><strong><a href="csrec_rethinking_sequential_recommendation_from_a_causal_perspective/">CSRec: Rethinking Sequential Recommendation from A Causal Perspective</a></strong></dt>
<dd>
<p>将序列推荐重构为因果干预下的决策接受概率建模，用递归约束隔离系统影响，AHR@0.2 提升约 1.5–3 倍。</p>
</dd>
<dt><strong><a href="data_augmentation_as_free_lunch_exploring_the_test_time_augmentation_for_se/">Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation</a></strong></dt>
<dd>
<p>将数据增强从训练期迁移到推理期，提出 TNoise 与 TMask 两个免训练、免参数的测试时增强算子，在 Full-Ranking 协议下平均提升 14%–30%。</p>
</dd>
<dt><strong><a href="diff_dual_side_information_filtering_and_fusion_for_sequential_recommendati/">DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation</a></strong></dt>
<dd>
<p>频域滤波去噪 + 早期/中间双重融合的侧信息序列推荐，Full-Ranking 下 R@20 最高提升 14.1%</p>
</dd>
<dt><strong><a href="fim_frequency_aware_multi_view_interest_modeling_for_local_life_service_rec/">FIM: Frequency-Aware Multi-View Interest Modeling for Local-Life Service Recommendation</a></strong></dt>
<dd>
<p>多视图搜索分离本地生活多周期需求，FFT 频域感知演化建模周期兴趣，Kuaishou 线上 GMV +8.46%。</p>
</dd>
<dt><strong><a href="heterrec_heterogeneous_information_transformer_for_scalable_sequential_reco/">HeterRec: Heterogeneous Information Transformer for Scalable Sequential Recommendation</a></strong></dt>
<dd>
<p>将物品拆解为异构特征 token 序列并设计层级因果 Transformer 与列表级多步预测损失，在工业与 Amazon 数据上 Recall@100 提升 3.8%–4.2%。</p>
</dd>
<dt><strong><a href="improving_sequential_recommenders_through_counterfactual_augmentation_of_sy/">Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure</a></strong></dt>
<dd>
<p>用 Decision Transformer 建模全量曝光序列，并以随机/自改进反事实增广+用户模拟器挖掘未曝光兴趣，KuaiRand 上 Recall@20 相对提升约 87%。</p>
</dd>
<dt><strong><a href="intent_aware_diffusion_with_contrastive_learning_for_sequential_recommendat/">Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 K-means 意图原型引导条件扩散生成意图对齐正视图替代随机增强，在 5 个数据集 Full-Ranking 下 NDCG 平均提升 20.68%。</p>
</dd>
<dt><strong><a href="linear_item_item_model_with_neural_knowledge_for_session_based_recommendati/">Linear Item-Item Model with Neural Knowledge for Session-based Recommendation</a></strong></dt>
<dd>
<p>将神经会话模型的转移知识蒸馏为物品-物品矩阵正则项，与线性自蒸馏相似度统一闭式求解，R@20 最高提升 14.78% 且推理 FLOPs 降低 813 倍。</p>
</dd>
<dt><strong><a href="pyramid_mixer_multi_dimensional_multi_period_interest_modeling_for_sequenti/">Pyramid Mixer: Multi-dimensional Multi-period Interest Modeling for Sequential Recommendation</a></strong></dt>
<dd>
<p>用纯 MLP 金字塔 Mixer 做跨行为/跨特征/跨周期三维兴趣建模，离线全库排序超越 SASRec/MLP4Rec，抖音线上停留时长 +0.106%。</p>
</dd>
<dt><strong><a href="star_rec_making_peace_with_length_variance_and_pattern_diversity_in_sequent/">STAR-Rec: Making Peace with Length Variance and Pattern Diversity in Sequential Recommendation</a></strong></dt>
<dd>
<p>用偏好感知注意力与 SSM 双路径自适应融合 + 序列级 MoE，统一建模长度方差与行为模式多样性，Leave-One-Out 全库排序下 Recall@10 最高提升 2.15%。</p>
</dd>
</dl>

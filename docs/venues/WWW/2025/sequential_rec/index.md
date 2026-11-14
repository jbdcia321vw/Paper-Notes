# ⏱️ 序列与会话推荐

**WWW (TheWebConf) 2025** · **7** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (7)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/)

🔥 **高频主题：** WWW 2025 ×7 · 精排 ×4 · 召回 ×2

<dl>
<dt><strong><a href="a_contextual_aware_position_encoding_for_sequential_recommendation/">A Contextual-Aware Position Encoding for Sequential Recommendation</a></strong></dt>
<dd>
<p>首个为序列推荐定制的上下文感知位置编码 CAPE：用目标-上下文不相似度累加动态分配位置，配合门控融合异构特征，在 5 个骨干与工业 A/B 上稳定涨点。</p>
</dd>
<dt><strong><a href="efficient_user_history_modeling_with_amortized_inference_for_deep_learning_/">Efficient user history modeling with amortized inference for deep learning recommendation models</a></strong></dt>
<dd>
<p>用 cross-attention 追加候选替代逐项拼接，使历史编码可摊销推理，LinkedIn Feed/Ads 延迟降 30%</p>
</dd>
<dt><strong><a href="flare_fusing_language_models_and_collaborative_architectures_for_recommende/">FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancement</a></strong></dt>
<dd>
<p>用冻结 mT5+Perceiver 将文本注入 Bert4Rec，并以重调 Bert4Rec 基线、大词表评测与 Critiquing 任务重估 ID+文本增益。</p>
</dd>
<dt><strong><a href="large_memory_network_for_recommendation/">Large Memory Network for Recommendation</a></strong></dt>
<dd>
<p>用百万级全局共享记忆块压缩用户历史行为，以乘积量化双轴激活实现 O(√n d) 检索，抖音电商搜索 AUC +4.68%、订单/人 +0.87%。</p>
</dd>
<dt><strong><a href="long_term_interest_clock_fine_grained_time_perception_in_streaming_recommen/">Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System</a></strong></dt>
<dd>
<p>用候选物品+当前时刻双查询从长期行为中检索子序列，并以时距感知注意力聚合，实现流式推荐细粒度时间感知，线上活跃天数+0.122%。</p>
</dd>
<dt><strong><a href="retrieval_with_learned_similarities/">Retrieval with Learned Similarities</a></strong></dt>
<dd>
<p>证明 MoL 为通用相似度逼近器，配互信息负载均衡损失与带误差界的近似 top-K 检索，序列推荐 HR@1 平均提升 29.1%，检索延迟最高降 66×。</p>
</dd>
<dt><strong><a href="td3_tucker_decomposition_based_dataset_distillation_method_for_sequential_r/">TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 Tucker 分解把合成序列摘要解耦为用户/时间/共享物品因子与关系核，配合特征空间对齐双层优化，在 0.2%–7.4% 数据量下逼近甚至超越全量训练。</p>
</dd>
</dl>

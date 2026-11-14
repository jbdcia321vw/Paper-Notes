# ⏱️ 序列与会话推荐

**AAAI (RecSys Track) 2026** · **5** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (6)](../../2025/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** AAAI 2026 ×5 · 召回 ×3 · 精排 ×2

<dl>
<dt><strong><a href="bid_farewell_to_seesaw_towards_accurate_long_tail_session_based_recommendat/">Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents</a></strong></dt>
<dd>
<p>提出混合意图学习与意图约束损失的双约束即插即用框架 HID，在三个数据集上同时提升会话推荐准确率与长尾覆盖，打破 see-saw 困境。</p>
</dd>
<dt><strong><a href="driving_engagement_in_daily_fantasy_sports_with_a_scalable_and_urgency_awar/">Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine</a></strong></dt>
<dd>
<p>在 DIN 上注入实时紧迫度特征与时间位置编码，配合 neuralNDCG 列表损失，在 6.5 亿用户级工业数据上 nDCG@1 相对 LightGBM 提升 9%。</p>
</dd>
<dt><strong><a href="exploiting_inter_session_information_with_frequency_enhanced_dual_path_netw/">Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation</a></strong></dt>
<dd>
<p>FreqRec 用可学习复数频域 MLP 双路建模跨会话群体节奏与用户局部高频，并加频域一致性损失，全库排序下 NDCG@10 最高提升 7.38%。</p>
</dd>
<dt><strong><a href="hymoerec_hybrid_mixture_of_experts_for_sequential_recommendation/">HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation</a></strong></dt>
<dd>
<p>用共享+稀疏双分支混合专家替换均匀 PFFN，配合自适应融合与负载均衡，在 ML-1M/Beauty 全库排序上稳定超越 Mamba4Rec。</p>
</dd>
<dt><strong><a href="wavelet_enhanced_adaptive_frequency_filter_for_sequential_recommendation/">Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation</a></strong></dt>
<dd>
<p>用上下文动态调制频域滤波器并叠加 Haar 小波高频增强，在 4 个基准 Full-Ranking 下 HR@20 最高提升 11.4%，长序列下训练开销低于 BSARec/SLIME4Rec。</p>
</dd>
</dl>

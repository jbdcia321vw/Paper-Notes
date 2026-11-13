# ✨ 生成式推荐与离散语义 ID

**WWW (TheWebConf) 2026** · **6** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/)

🔥 **高频主题：** WWW ×3 · WWW 2026 ×3 · 生成式推荐与离散语义ID ×3 · 生成式端到端 ×3 · 生成式推荐 ×2

<dl>
<dt><strong><a href="beyond_the_flat_sequence_hierarchical_and_preference_aware_generative_recom/">Beyond the Flat Sequence: Hierarchical and Preference-Aware Generative Recommendations</a></strong></dt>
<dd>
<p>用会话层次预训练+偏好引导稀疏注意力打破 HSTU 扁平序列假设，APPGallery 全库 AUC 0.8929、线上 eCPM +1.99%</p>
</dd>
<dt><strong><a href="diffgrm_diffusion_based_generative_recommendation_model/">DiffGRM: Diffusion-based Generative Recommendation Model</a></strong></dt>
<dd>
<p>用掩码离散扩散替代自回归解码 SID，配合 OPQ 解耦分词、难度感知相干加噪与置信度并行去噪，NDCG@10 提升 6.9%–15.5%。</p>
</dd>
<dt><strong><a href="diffusion_generative_recommendation_with_continuous_tokens/">Diffusion Generative Recommendation with Continuous Tokens</a></strong></dt>
<dd>
<p>用 σ-VAE 连续分词器替代量化码本，并以条件扩散+离散散度损失生成用户偏好，四数据集 Full-Ranking 上 HR@10 平均超 TIGER 11.76%。</p>
</dd>
<dt><strong><a href="dualgr_generative_retrieval_with_long_and_short_term_interests_modeling/">DualGR: Generative Retrieval with Long and Short-Term Interests Modeling</a></strong></dt>
<dd>
<p>双分支长短期路由+桶内搜索式SID解码+曝光感知NTP损失，快手双列场景在线HR@1000达19.53%，A/B视频播放+0.527%。</p>
</dd>
<dt><strong><a href="rastp_representation_aware_semantic_token_pruning_for_generative_recommenda/">RASTP: Representation-Aware Semantic Token Pruning for Generative Recommendation with Semantic Identifiers</a></strong></dt>
<dd>
<p>以表征L1范数与注意力中心性乘积打分，在Transformer中间层动态剪枝SID冗余token，训练提速26.7%且性能持平略升。</p>
</dd>
<dt><strong><a href="towards_context_aware_reasoning_enhanced_generative_searching_in_e_commerce/">Towards Context-aware Reasoning-enhanced Generative Searching in E-commerce</a></strong></dt>
<dd>
<p>将电商搜索异构上下文统一为JSON文本与残差K-Means语义ID，用SFT-RL自演化后训练与去偏R-GRPO提升生成式搜索排序，All-100K上HR@10相对提升6.24%。</p>
</dd>
</dl>

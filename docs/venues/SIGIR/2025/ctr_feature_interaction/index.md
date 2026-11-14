# 🎯 CTR/CVR 预估与特征交叉

**SIGIR 2025** · **8** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [SIGIR 2026 (9)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×8 · 精排 ×7

<dl>
<dt><strong><a href="deep_multiple_quantization_network_on_long_behavior_sequence_for_click_thro/">Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>用多码本量化将超长行为序列端到端压缩为约百个兴趣簇码字，配合 HSTU 簇间交互与候选感知注意力，消除两阶段检索的相关性分布偏差，线上 CTR +3.5%、RPM +2.0%。</p>
</dd>
<dt><strong><a href="dlf_enhancing_explicit_implicit_interaction_via_dynamic_low_order_aware_fus/">DLF: Enhancing Explicit-Implicit Interaction via Dynamic Low-Order-Aware Fusion for CTR Prediction</a></strong></dt>
<dd>
<p>以残差感知低阶交互网络与逐层注意力融合，动态平衡显隐式特征交叉，四数据集AUC全面SOTA。</p>
</dd>
<dt><strong><a href="elec_efficient_large_language_model_empowered_click_through_rate_prediction/">ELEC: Efficient Large Language Model-Empowered Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>伪孪生增益/朴素双网络+分数级CLID与表示级MSE双蒸馏，在线仅跑表格DCNv2，AUC较FLIP提升0.0020/0.0034。</p>
</dd>
<dt><strong><a href="hybrid_advertising_in_the_sponsored_search/">Hybrid Advertising in the Sponsored Search</a></strong></dt>
<dd>
<p>提出混合广告拍卖模型与 HRegNet 神经网络机制，在近 DSIC/IR 约束下联合优化独立店铺与店铺-品牌捆绑包的广告位分配与定价，真实数据上收益显著超越 VCG 与 JRegNet。</p>
</dd>
<dt><strong><a href="interest_changes_considering_user_interest_life_cycle_in_recommendation_sys/">Interest Changes: Considering User Interest Life Cycle in Recommendation System</a></strong></dt>
<dd>
<p>用曝光/点击/交互直方图+VQ聚类刻画兴趣生命周期，经特征重校准与专家网络融合注入MMOE精排，Lofter线上CTR+0.38%、CVR+1.04%。</p>
</dd>
<dt><strong><a href="lrea_low_rank_efficient_attention_on_modeling_long_term_user_behaviors_for_/">LREA: Low-Rank Efficient Attention on Modeling Long-Term User Behaviors for CTR Prediction</a></strong></dt>
<dd>
<p>用低秩分解+矩阵吸收把万级长行为序列压到秩 r 维做目标注意力，配非负正则保非线性，线上仅增 3ms 延迟、CTR +5.88%。</p>
</dd>
<dt><strong><a href="multi_objective_aligned_bidword_generation_model_for_e_commerce_search_adve/">Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising</a></strong></dt>
<dd>
<p>用判别器提供相关性/真实性/CPM三维奖励，扩展MODPO做多目标偏好对齐，驱动LLM生成高商业价值竞价词，线上广告收入+2.13%。</p>
</dd>
<dt><strong><a href="nam_a_normalization_attention_model_for_personalized_product_search_in_flig/">NAM: A Normalization Attention Model for Personalized Product Search In Fliggy</a></strong></dt>
<dd>
<p>用逆物品频率(IIF)对注意力做全局归一化并门控个性化强度，解决热门物品高估/长尾低估，离线CTCVR AUC+0.001、线上CTCVR+0.8%。</p>
</dd>
</dl>

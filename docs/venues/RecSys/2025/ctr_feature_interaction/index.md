# 🎯 CTR/CVR 预估与特征交叉

**RecSys 2025** · **7** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [RecSys 2026 (3)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×6 · 精排 ×5 · 召回 ×2

<dl>
<dt><strong><a href="blending_sequential_embeddings_graphs_and_engineered_features_4th_place_sol/">Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025</a></strong></dt>
<dd>
<p>异构因果Transformer序列编码+TwHIN图+300维手工特征经DCN-v2融合，构建跨6任务通用用户画像，获挑战赛第4名。</p>
</dd>
<dt><strong><a href="counterfactual_inference_under_thompson_sampling/">Counterfactual Inference under Thompson Sampling</a></strong></dt>
<dd>
<p>推导 Thompson Sampling 下动作倾向性的精确可计算闭式解，使 IPS 类离线评估在 TS 策略下首次可行，OBP 仿真验证无偏收敛。</p>
</dd>
<dt><strong><a href="decoupled_entity_representation_learning_for_pinterest_ads_ranking/">Decoupled Entity Representation Learning for Pinterest Ads Ranking</a></strong></dt>
<dd>
<p>提出解耦式上游-下游实体表征框架 DERM，三塔+对比自监督+滑动平均聚合，离线 AUC 提升 0.19%、线上 CTR +1.38%、CPA -1.61%</p>
</dd>
<dt><strong><a href="exploring_scaling_laws_of_ctr_model_for_online_performance_improvement/">Exploring Scaling Laws of CTR Model for Online Performance Improvement</a></strong></dt>
<dd>
<p>提出 SUAN 统一注意力块实现 CTR 模型三级量级缩放律，再经稀疏注意力+并行推理+在线蒸馏落地线上，CTR +2.81%。</p>
</dd>
<dt><strong><a href="identifying_offline_metrics_that_predict_online_impact_a_pragmatic_strategy/">Identifying Offline Metrics that Predict Online Impact: A Pragmatic Strategy for Real-World Recommender Systems</a></strong></dt>
<dd>
<p>用帕累托前沿近似单模型多偏好分流，在 OTTO 线上实验验证 Recall@20→CTR、OD@20→CVR、乘积→销量 的离线-在线对齐关系。</p>
</dd>
<dt><strong><a href="modeling_long_term_user_behaviors_with_diffusion_driven_multi_interest_netw/">Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction</a></strong></dt>
<dd>
<p>用目标正交分解构造兴趣通道做多兴趣抽取，再用条件扩散生成增强兴趣，工业 A/B 上 CTR +1.52%。</p>
</dd>
<dt><strong><a href="r4ec_a_reasoning_reflection_and_refinement_framework_for_recommendation_sys/">R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems</a></strong></dt>
<dd>
<p>用 actor/reflection 双小模型迭代反思精炼 LLM 知识，注入 CTR 主干，AUC 相对提升 2.5%+，线上收入 +2.2%。</p>
</dd>
</dl>

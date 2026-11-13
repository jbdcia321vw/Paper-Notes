# 🎯 CTR/CVR 预估与特征交叉

**SIGIR 2026** · **10** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [CIKM 2026 (4)](../../../CIKM/2026/ctr_feature_interaction/) · [KDD 2026 (3)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2026 (1)](../../../RecSys/2026/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (2)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×7 · 精排 ×7 · 特征交叉 ×2

<dl>
<dt><strong><a href="beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>揭示稠密MLP隐式权重稀疏现象，提出 filter-then-fuse 多视图显式稀疏框架 SSR，在十亿级工业数据与三个公开集上以更低算力超越 RankMixer。</p>
</dd>
<dt><strong><a href="cheaper_is_better_a_discount_aware_network_for_conversion_rate_prediction_i/">Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System</a></strong></dt>
<dd>
<p>用 FFT 时频分解 + 用户/上下文双重去偏 + 折扣率回归辅助任务建模商品折扣率时序，离线 AUC +1.61%、线上 pCVR +3.63%、GMV +2.23%。</p>
</dd>
<dt><strong><a href="counterfactual_multi_task_learning_for_delayed_conversion_modeling_in_e_com/">Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion</a></strong></dt>
<dd>
<p>面向大促预热期延迟转化，提出多任务+个性化门控迁移+反事实因果正则的 CM-DCM，离线 AUC 与线上 GMV 双收益。</p>
</dd>
<dt><strong><a href="exploring_test_time_scaling_via_prediction_merging_on_large_scale_recommend/">Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出测试时缩放新维度：通过异构架构组合与同构随机初始化生成多样预测并在 logit 空间融合，在 Criteo/Avazu/KDD12 上稳定超越参数缩放。</p>
</dd>
<dt><strong><a href="follow_the_trace_exploiting_post_click_trajectories_for_online_delayed_conv/">Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将点击后行为演化建模为反馈轨迹，用贝叶斯分解+可靠性门控回溯补全动态修正未转化样本后验，Criteo AUC 0.8382 超越 DEFUSE。</p>
</dd>
<dt><strong><a href="generative_bid_shading_in_real_time_bidding_advertising/">Generative Bid Shading in Real-Time Bidding Advertising</a></strong></dt>
<dd>
<p>用自回归残差生成+CHNet奖励模型+GRPO偏好对齐替代两阶段出价调价，离线SR达41.74%、线上ROI+3.4%且推理耗时降37.6%</p>
</dd>
<dt><strong><a href="hyformer_revisiting_the_roles_of_sequence_modeling_and_feature_interaction_/">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></strong></dt>
<dd>
<p>提出统一混合Transformer架构HyFormer，以Query Decoding与Query Boosting交替迭代融合长序列建模与异构特征交叉，在十亿级工业数据与线上A/B上显著超越LONGER与RankMixer。</p>
</dd>
<dt><strong><a href="r_f_inventory_a_large_scale_dataset_for_monotonic_inventory_estimation_in_r/">R&amp;F-Inventory: A Large-Scale Dataset for Monotonic Inventory Estimation in Reach and Frequency Advertising</a></strong></dt>
<dd>
<p>首个公开的 R&amp;F 合约库存预估数据集，以「定向-排期-频控」为上下文提供完整预算-效果曲线，定义单点预测与多维单调曲线重建两大基准任务。</p>
</dd>
<dt><strong><a href="rq_gmm_residual_quantized_gaussian_mixture_model_for_multimodal_semantic_di/">RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction</a></strong></dt>
<dd>
<p>用高斯混合模型替代硬聚类做残差量化，将多模态嵌入离散为语义ID，提升码本利用率与重建精度，线上广告主价值+1.502%</p>
</dd>
<dt><strong><a href="understanding_dnns_in_feature_interaction_models_a_dimensional_collapse_per/">Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective</a></strong></dt>
<dd>
<p>从嵌入维度坍缩视角重新解释DNN在特征交叉模型中的作用，证明并行/堆叠DNN通过展平奇异值谱缓解坍缩，并给出基于梯度的理论分析。</p>
</dd>
</dl>

# 🎯 CTR/CVR 预估与特征交叉

**SIGIR 2026** · **11** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

🔥 **高频主题：** CTR预估与特征交叉 ×9 · 精排 ×7 · 延迟反馈建模 ×2 · 特征交叉 ×2

<dl>
<dt><strong><a href="beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>提出 filter-then-fuse 显式稀疏范式，用静态随机过滤与迭代竞争稀疏(ICS)替代稠密全连接，在十亿级工业数据上实现可扩展的CTR提升。</p>
</dd>
<dt><strong><a href="cheaper_is_better_a_discount_aware_network_for_conversion_rate_prediction_i/">Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System</a></strong></dt>
<dd>
<p>用 FFT 时频分解 + 用户/上下文双重去偏 + 折扣率回归辅助任务建模商品折扣率时序，离线 AUC +1.61%、线上 pCVR +3.63%、GMV +2.23%。</p>
</dd>
<dt><strong><a href="clax_fast_and_flexible_neural_click_models_in_jax/">CLAX: Fast and Flexible Neural Click Models in JAX</a></strong></dt>
<dd>
<p>用梯度优化替代 EM 训练经典点击模型，JAX 实现单卡 2 小时跑完 10 亿级 Baidu-ULTR 会话</p>
</dd>
<dt><strong><a href="counterfactual_multi_task_learning_for_delayed_conversion_modeling_in_e_com/">Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion</a></strong></dt>
<dd>
<p>面向大促预热期延迟转化，提出多任务+个性化门控迁移+反事实因果正则的 CM-DCM，离线 AUC 与线上 GMV 双收益。</p>
</dd>
<dt><strong><a href="fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>提出目标条件化频域滤波双分支网络，用注意力谱熵观察引导频域去噪，三数据集GAUC/AUC全面超越SOTA</p>
</dd>
<dt><strong><a href="follow_the_trace_exploiting_post_click_trajectories_for_online_delayed_conv/">Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将点击后行为演化建模为反馈轨迹，用贝叶斯分解融合静态意图与动态轨迹，并以可靠性门控的回溯补全模块作为模型无关增强器，缓解延迟反馈假阴性偏差。</p>
</dd>
<dt><strong><a href="generative_bid_shading_in_real_time_bidding_advertising/">Generative Bid Shading in Real-Time Bidding Advertising</a></strong></dt>
<dd>
<p>用自回归生成式模型端到端输出调价比例，并以CHNet奖励模型+GRPO偏好对齐优化短期与长期剩余，离线与美团线上A/B均显著提升。</p>
</dd>
<dt><strong><a href="hyformer_revisiting_the_roles_of_sequence_modeling_and_feature_interaction_/">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></strong></dt>
<dd>
<p>提出统一混合Transformer架构HyFormer，以Query Decoding与Query Boosting交替迭代融合长序列建模与异构特征交叉，在十亿级工业数据与线上A/B上显著超越LONGER与RankMixer。</p>
</dd>
<dt><strong><a href="rq_gmm_residual_quantized_gaussian_mixture_model_for_multimodal_semantic_di/">RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction</a></strong></dt>
<dd>
<p>用高斯混合模型替代硬聚类做残差量化，将多模态嵌入离散为语义ID，提升码本利用率与重建精度，线上广告主价值+1.502%</p>
</dd>
<dt><strong><a href="understanding_dnns_in_feature_interaction_models_a_dimensional_collapse_per/">Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective</a></strong></dt>
<dd>
<p>从嵌入维度坍缩视角重新解释DNN在特征交叉模型中的作用，证明并行/堆叠DNN通过展平奇异值谱缓解坍缩，并给出基于梯度的理论分析。</p>
</dd>
<dt><strong><a href="unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动信号仅作相关物品内偏好精修，融合交叉编码器分级标签、多渠道检索先验与去偏互动构建统一监督目标训练双编码器，离线 P@25 +10.5%、NDCG@25 +5.7%，线上曝光 +0.60%。</p>
</dd>
</dl>

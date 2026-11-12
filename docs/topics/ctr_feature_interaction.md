# 🎯 CTR/CVR 预估与特征交叉 (CTR Prediction & Feature Interaction)

> **研究范畴**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

当前方向共收录 **21** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/an_epistemic_position_based_click_model_from_interactions_to_epistemic_dist/">An Epistemic Position-Based Click Model: From Interactions to Epistemic Distributions of Relevance and Bias</a></strong></dt>
<dd>
<p>首个上下文认知 PBM：用 Beta 分布刻画相关性与位置偏置的认知不确定性，以自归一化+位置条件化梯度估计稳定训练，测试集对数似然显著优于点式 PBM。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>提出 filter-then-fuse 显式稀疏范式，用静态随机过滤与迭代竞争稀疏(ICS)替代稠密全连接，在十亿级工业数据上实现可扩展的CTR提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/cheaper_is_better_a_discount_aware_network_for_conversion_rate_prediction_i/">Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System</a></strong></dt>
<dd>
<p>用 FFT 时频分解 + 用户/上下文双重去偏 + 折扣率回归辅助任务建模商品折扣率时序，离线 AUC +1.61%、线上 pCVR +3.63%、GMV +2.23%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/clax_fast_and_flexible_neural_click_models_in_jax/">CLAX: Fast and Flexible Neural Click Models in JAX</a></strong></dt>
<dd>
<p>用梯度优化替代 EM 训练经典点击模型，JAX 实现单卡 2 小时跑完 10 亿级 Baidu-ULTR 会话</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong></dt>
<dd>
<p>提出可学习多序列构造+线性注意力，将用户历史解耦为K条主题潜序列以缓解上下文污染，Meta四场景离线NE与线上A/B均显著提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/counterfactual_multi_task_learning_for_delayed_conversion_modeling_in_e_com/">Counterfactual Multi-task Learning for Delayed Conversion Modeling in E-commerce Sales Pre-Promotion</a></strong></dt>
<dd>
<p>面向大促预热期延迟转化，提出多任务+个性化门控迁移+反事实因果正则的 CM-DCM，离线 AUC 与线上 GMV 双收益。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/exploring_test_time_scaling_via_prediction_merging_on_large_scale_recommend/">Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出测试时缩放新维度：通过异构架构组合与同构随机初始化生成多样预测并在 logit 空间融合，在 Criteo/Avazu/KDD12 上稳定超越参数缩放。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong></dt>
<dd>
<p>提出目标条件化频域滤波双分支网络，用注意力谱熵观察引导频域去噪，三数据集GAUC/AUC全面超越SOTA</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/follow_the_trace_exploiting_post_click_trajectories_for_online_delayed_conv/">Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction</a></strong></dt>
<dd>
<p>将点击后行为演化建模为反馈轨迹，用贝叶斯分解融合静态意图与动态轨迹，并以可靠性门控的回溯补全模块作为模型无关增强器，缓解延迟反馈假阴性偏差。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/generative_bid_shading_in_real_time_bidding_advertising/">Generative Bid Shading in Real-Time Bidding Advertising</a></strong></dt>
<dd>
<p>用自回归生成式模型端到端输出调价比例，并以CHNet奖励模型+GRPO偏好对齐优化短期与长期剩余，离线与美团线上A/B均显著提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/hyformer_revisiting_the_roles_of_sequence_modeling_and_feature_interaction_/">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></strong></dt>
<dd>
<p>提出统一混合Transformer架构HyFormer，以Query Decoding与Query Boosting交替迭代融合长序列建模与异构特征交叉，在十亿级工业数据与线上A/B上显著超越LONGER与RankMixer。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱足门控+TTNP自监督学习兴趣生命周期，PPRR降超60%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/modular_representation_compression_adapting_llms_for_efficient_and_effectiv/">Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 中间层表示优势 (MRA) 现象，提出模块化解耦压缩框架 MARC，将 4096 维 LLM 表示压缩至 128 维，线上 A/B 测试 eCPM 提升 2.82%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/next_interest_flow_a_generative_pre_training_paradigm_for_recommender_syste/">Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines</a></strong></dt>
<dd>
<p>将用户意图建模为隐流形上的连续演化轨迹，用切空间正交与测地平滑约束生成式预训练，并以双向对齐+TSP桥接判别式精排，淘宝线上CTCVR+11.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong></dt>
<dd>
<p>在 Stiefel 流形上约束 Item-ID 投影矩阵，保持奇异值谱等距以缓解稀疏扩展下的表征坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/r_f_inventory_a_large_scale_dataset_for_monotonic_inventory_estimation_in_r/">R&amp;F-Inventory: A Large-Scale Dataset for Monotonic Inventory Estimation in Reach and Frequency Advertising</a></strong></dt>
<dd>
<p>发布首个含预算-触达曲线结构的 R&amp;F 合约库存数据集，定义单点预测与多维单调曲线重建两任务及可复现基准。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/rq_gmm_residual_quantized_gaussian_mixture_model_for_multimodal_semantic_di/">RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction</a></strong></dt>
<dd>
<p>用高斯混合模型替代硬聚类做残差量化，将多模态嵌入离散为语义ID，提升码本利用率与重建精度，线上广告主价值+1.502%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/semantic_ids_for_recommender_systems_at_snapchat_use_cases_technical_challe/">Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices</a></strong></dt>
<dd>
<p>Snapchat 工业级 SID 落地实践：STE 全码本反传与多模态融合缓解码本坍缩，启发式桶内消歧解决 SID-Item 碰撞，离线与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/solaris_speculative_offloading_of_latent_based_representation_for_inference/">SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling</a></strong></dt>
<dd>
<p>借鉴投机解码思想，异步预计算并缓存基础模型用户-物品交互表征，实现推理时知识蒸馏，Meta 广告线上收入 +0.67%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/understanding_dnns_in_feature_interaction_models_a_dimensional_collapse_per/">Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective</a></strong></dt>
<dd>
<p>从嵌入维度坍缩视角重新解释DNN在特征交叉模型中的作用，证明并行/堆叠DNN通过展平奇异值谱缓解坍缩，并给出基于梯度的理论分析。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动信号仅作相关物品内偏好精修，融合交叉编码器分级标签、多渠道检索先验与去偏互动构建统一监督目标训练双编码器，离线 P@25 +10.5%、NDCG@25 +5.7%，线上曝光 +0.60%。</p>
</dd>
</dl>

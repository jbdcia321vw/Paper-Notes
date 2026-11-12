# 🎯 CTR/CVR 预估与特征交叉 (CTR Prediction & Feature Interaction)

> **研究范畴**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

当前方向共收录 **10** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/an_epistemic_position_based_click_model_from_interactions_to_epistemic_dist/">An Epistemic Position-Based Click Model: From Interactions to Epistemic Distributions of Relevance and Bias</a></strong></dt>
<dd>
<p>首个上下文认知型位置点击模型，用 Beta 分布刻画相关性与位置偏置的认知不确定性，提出自归一化与位置条件化梯度估计实现稳定优化</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>提出 filter-then-fuse 显式稀疏范式，用静态随机过滤与迭代竞争稀疏(ICS)替代稠密全连接，在十亿级工业数据上实现可扩展的CTR提升。</p>
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
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/modular_representation_compression_adapting_llms_for_efficient_and_effectiv/">Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 中间层表示优势 (MRA) 现象，提出模块化解耦压缩框架 MARC，将 4096 维 LLM 表示压缩至 128 维，线上 A/B 测试 eCPM 提升 2.82%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/r_f_inventory_a_large_scale_dataset_for_monotonic_inventory_estimation_in_r/">R&amp;F-Inventory: A Large-Scale Dataset for Monotonic Inventory Estimation in Reach and Frequency Advertising</a></strong></dt>
<dd>
<p>发布首个含预算-触达曲线结构的 R&amp;F 合约库存数据集，定义单点预测与多维单调曲线重建两任务及可复现基准。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/semantic_ids_for_recommender_systems_at_snapchat_use_cases_technical_challe/">Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices</a></strong></dt>
<dd>
<p>Snapchat 工业级 SID 落地实践：STE 全码本反传与多模态融合缓解码本坍缩，启发式桶内消歧解决 SID-Item 碰撞，离线与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/solaris_speculative_offloading_of_latent_based_representation_for_inference/">SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling</a></strong></dt>
<dd>
<p>借鉴投机解码思想，异步预计算并缓存基础模型用户-物品交互表征，实现推理时知识蒸馏，Meta 广告线上收入 +0.67%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动信号仅作相关物品内偏好精修，融合交叉编码器分级标签、多渠道检索先验与去偏互动构建统一监督目标训练双编码器，离线 P@25 +10.5%、NDCG@25 +5.7%，线上曝光 +0.60%。</p>
</dd>
</dl>

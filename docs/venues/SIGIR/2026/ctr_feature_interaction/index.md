# 🎯 CTR/CVR 预估与特征交叉

**SIGIR 2026** · **5** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

🔥 **高频主题：** CTR预估与特征交叉 ×5 · 精排 ×3

<dl>
<dt><strong><a href="beyond_dense_connectivity_explicit_sparsity_for_scalable_recommendation/">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></strong></dt>
<dd>
<p>提出 filter-then-fuse 显式稀疏范式，用静态随机过滤与迭代竞争稀疏(ICS)替代稠密全连接，在十亿级工业数据上实现可扩展的CTR提升。</p>
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
<dt><strong><a href="unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动信号仅作相关物品内偏好精修，融合交叉编码器分级标签、多渠道检索先验与去偏互动构建统一监督目标训练双编码器，离线 P@25 +10.5%、NDCG@25 +5.7%，线上曝光 +0.60%。</p>
</dd>
</dl>

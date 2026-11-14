# 🎯 CTR/CVR 预估与特征交叉

**KDD 2025** · **8** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [KDD 2026 (5)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×7 · 精排 ×7 · 自动出价 ×2

<dl>
<dt><strong><a href="an_audio_centric_multi_task_learning_framework_for_streaming_ads_targeting_/">An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify</a></strong></dt>
<dd>
<p>面向音频主导多模态广告场景的 CAMoE 框架，以模态感知任务分组、自适应损失掩码与 DCN 专家实现近帕累托最优 CTR 预估，线上音频 CTR +14.5%。</p>
</dd>
<dt><strong><a href="beyond_advertising_mechanism_design_for_platform_wide_marketing_service_qua/">Beyond Advertising: Mechanism Design for Platform-Wide Marketing Service &#x27;QuanZhanTui&#x27;</a></strong></dt>
<dd>
<p>提出全站推场景下的库存约束卖家模型与流式支付拍卖 LPA，在合成与真实数据上同时提升流式福利、用户体验与平台收入。</p>
</dd>
<dt><strong><a href="bid2x_revealing_dynamics_of_bidding_environment_in_online_advertising_from_/">Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens</a></strong></dt>
<dd>
<p>将自动出价环境建模重构为跨场景基础模型的下一token预测，用变量/时间双注意力+零膨胀投影，8数据集MAE全面领先，线上GMV+4.65%。</p>
</dd>
<dt><strong><a href="contextual_generative_auction_with_permutation_level_externalities_for_onli/">Contextual Generative Auction with Permutation-level Externalities for Online Advertising</a></strong></dt>
<dd>
<p>用自回归生成器直接生成广告位序列并建模排列级外部性，配合可微事后遗憾学习支付规则，淘宝线上 RPM +3.2%。</p>
</dd>
<dt><strong><a href="lightweight_auto_bidding_based_on_traffic_prediction_in_live_advertising/">Lightweight Auto-bidding based on Traffic Prediction in Live Advertising</a></strong></dt>
<dd>
<p>直播广告场景下，用LightGBM预测全时段累计成本/点击，结合LP对偶最优出价公式与投影梯度下降求解对偶变量，实现秒级轻量自动出价，线上GMV+4.28%。</p>
</dd>
<dt><strong><a href="scaling_transformers_for_discriminative_recommendation_via_generative_pretr/">Scaling Transformers for Discriminative Recommendation via Generative Pretraining</a></strong></dt>
<dd>
<p>用生成式预训练初始化判别式排序模型并冻结稀疏Embedding，破解CTR/CVR过拟合，使稠密参数从13K扩到0.3B仍遵循幂律提升。</p>
</dd>
<dt><strong><a href="swat_statistical_modeling_of_video_watch_time_through_user_behavior_analysi/">SWaT: Statistical Modeling of Video Watch Time through User Behavior Analysis</a></strong></dt>
<dd>
<p>将用户观看行为假设翻译为分桶二项/几何分布白盒统计模型，以软标签 BCE 训练，离线与线上 A/B 均稳定超越 WLR/TPM/CREAD。</p>
</dd>
<dt><strong><a href="tayfcs_towards_light_feature_combination_selection_for_deep_recommender_sys/">TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems</a></strong></dt>
<dd>
<p>用泰勒展开+IME近似一次性梯度回传估计高阶组合重要性，再用LR滑窗剔除冗余，三数据集AUC显著提升且线上CVR+13.9%</p>
</dd>
</dl>

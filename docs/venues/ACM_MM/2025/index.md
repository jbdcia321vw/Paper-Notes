---
hide:
  - toc
---

# 🏛️ ACM MM 2025 推荐系统论文解读

9 篇 ACM MM 2025 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl>
<dt><strong><a href="sequential_rec/chord_customizing_hybrid_precision_on_device_model_for_sequential_recommend/">CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration</a></strong></dt>
<dd>
<p>云端超网络按用户画像生成逐通道混合精度量化策略（2-bit 编码），端侧单次前向完成个性化适配与推理，精度反超全精度基线且传输降 173.8×。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (4)</h2>
<dl>
<dt><strong><a href="multimodal_rec/dual_phase_playtime_guided_recommendation_interest_intensity_exploration_an/">Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks</a></strong></dt>
<dd>
<p>用双 Beta 分布解耦游玩时长强弱兴趣，并以多模态随机游走做跨类目探索，在 Steam 数据上同时提升准确率与覆盖度。</p>
</dd>
<dt><strong><a href="multimodal_rec/generating_negative_samples_for_multi_modal_recommendation/">Generating Negative Samples for Multi-Modal Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 三步提示生成兼具凝聚力与难度的属性级负样本，并以因果学习解耦关键特征，全库排序下 Recall@10 最高提升 9.02%。</p>
</dd>
<dt><strong><a href="multimodal_rec/i_3_mrec_invariant_learning_with_information_bottleneck_for_incomplete_moda/">I$^3$-MRec: Invariant Learning with Information Bottleneck for Incomplete Modality Recommendation</a></strong></dt>
<dd>
<p>将各模态视为独立语义环境做 IRM 不变学习，并用 IB 压缩模态特有信息，在模态缺失场景下 Full-Ranking Recall@20 稳定提升 2%–4%。</p>
</dd>
<dt><strong><a href="multimodal_rec/refining_contrastive_learning_and_homography_relations_for_multi_modal_reco/">Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation</a></strong></dt>
<dd>
<p>用元网络+正交约束精炼多模态对比学习，并补建用户兴趣图与物品共现图，全库排序下 R@20 最高提升约 9.8%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/tackling_device_data_distribution_real_time_shift_via_prototype_based_param/">Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing</a></strong></dt>
<dd>
<p>云端神经适配器免反传生成参数编辑矩阵，多原型聚类+跨层知识迁移实现端侧实时分布偏移适配，1s 内完成个性化且 NDCG@5 显著超越微调。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/leveraging_multimodal_data_and_side_users_for_diffusion_cross_domain_recomm/">Leveraging Multimodal Data and Side Users for Diffusion Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 抽取多模态特征做初步对齐，再以边侧用户+重叠用户两阶段扩散生成目标域冷启动用户向量，Amazon 三任务 MAE 最高降 13.6%。</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/generative_flow_networks_for_personalized_multimedia_systems_a_case_study_o/">Generative Flow Networks for Personalized Multimedia Systems: A Case Study on Short Video Feeds</a></strong></dt>
<dd>
<p>用 GFlowNet 多候选生成替代单动作 RL 策略，在短视频预取场景实现个性化 QoE 与带宽双优化。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/lego_a_lightweight_and_efficient_multiple_attribute_unlearning_framework_fo/">LEGO: A Lightweight and Efficient Multiple-Attribute Unlearning Framework for Recommender Systems</a></strong></dt>
<dd>
<p>将多属性遗忘拆为并行嵌入校准与权重组合两步，以互信息最小化保证同时遗忘，动态需求下近零额外开销。</p>
</dd>
</dl>
</div>

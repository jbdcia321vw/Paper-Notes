# 👥 社交与群组推荐 (Social & Group Recommendation)

> **研究范畴**：社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模

当前方向共收录 **23** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/social_group_rec/disentangled_modeling_of_preferences_and_social_influence_for_group_recomme/">Disentangled Modeling of Preferences and Social Influence for Group Recommendation</a></strong></dt>
<dd>
<p>解耦用户偏好与社会影响的双通道(超)图传播，配合社交重要性引导的对比学习，缓解群组推荐的偏好偏差与数据稀疏。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/the_role_of_follow_networks_and_twitter_s_content_recommender_on_partisan_s/">The Role of Follow Networks and Twitter&#x27;s Content Recommender on Partisan Skew and Rumor Exposure during the 2022 U.S. Midterm Election</a></strong></dt>
<dd>
<p>用 18 个自动化审计账号在 2022 美国中期选举期间对比算法流与时间流，量化推荐算法与关注网络对党派倾斜、谣言与低质信息曝光的交互效应。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/social_group_rec/community_aware_social_community_recommendation/">Community-Aware Social Community Recommendation</a></strong></dt>
<dd>
<p>提出 CASO，用模块度最大化、社交紧密度聚合与协同编码三路解耦用户表征，并以 HSIC 互斥与 KL 社区检测损失优化，在 6 个真实社交网络上 Full-Ranking 全面超越 9 个基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/social_group_rec/heterogeneous_influence_maximization_in_user_recommendation/">Heterogeneous Influence Maximization in User Recommendation</a></strong></dt>
<dd>
<p>将异质传播概率与 RR 集影响力最大化引入好友邀请推荐，HeteroIR/HeteroIM 在 NSpread/Recall/NDCG 全面领先并落地腾讯游戏 A/B 提升 8.5%/10%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/social_group_rec/scaling_graph_neural_networks_for_friend_recommendation_multi_hash_user_emb/">Scaling Graph Neural Networks for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling</a></strong></dt>
<dd>
<p>用多哈希ID嵌入(表压缩&gt;98%)与时间戳排序CSR二分采样(O(log d+K))支撑194M用户/28B边GNN好友排序，线上好友添加+16%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/rl_bandit_rec/learning_peer_influence_probabilities_with_linear_contextual_bandits/">Learning Peer Influence Probabilities with Linear Contextual Bandits</a></strong></dt>
<dd>
<p>将同伴影响力概率学习建模为双目标线性上下文 Bandit，刻画 Regret–RMSE 帕累托前沿并提出 InfluenceCB 以单参数 β 导航权衡。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/social_group_rec/a_multistakeholder_approach_to_value_driven_co_design_of_recommender_system/">A Multistakeholder Approach to Value-Driven Co-Design of Recommender System Evaluation Metrics in Digital Archives</a></strong></dt>
<dd>
<p>通过 5×5 焦点小组将档案领域五类利益相关者价值映射为四阶段研究漏斗评估指标框架，提出研究路径质量、元数据加权相关性等指标方向。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/social_group_rec/consistent_explainers_or_unreliable_narrators_understanding_llm_generated_g/">Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations</a></strong></dt>
<dd>
<p>用社会选择聚合策略作多基线对4个LLM的群组推荐与解释做一致性审计，发现推荐近似ADD、解释常称平均但夹带未定义流行度/相似度/多样性。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cold_start_rec/socripple_a_two_stage_framework_for_cold_start_video_recommendations/">SocRipple: A Two-Stage Framework for Cold-Start Video Recommendations</a></strong></dt>
<dd>
<p>社交种子+嵌入KNN涟漪扩散的两阶段冷启视频召回框架，Recall@200 提升至 12.8–13.2%，线上冷启曝光 +36% 且互动率持平。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/social_group_rec/you_don_t_bring_me_flowers_mitigating_unwanted_recommendations_through_conf/">You Don&#x27;t Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control</a></strong></dt>
<dd>
<p>用保形风险控制对任意排序器做后处理，以“重复安全内容替换”替代纯删除，在 KuaiRand 上可证明地把不想要内容比例压到用户指定 α 以下且保住 nDCG。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/social_group_rec/are_we_really_making_progress_in_group_recommendation_unmasking_the_tie_bre/">Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion</a></strong></dt>
<dd>
<p>揭示群组推荐中训练期 sigmoid 分数压缩与评估期确定性破平交互导致的系统性评估偏差，提出并列感知 HR/NDCG 精确期望协议，使多个 SOTA 方法报告增益大幅缩水。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/social_group_rec/consensus_vs_dissent_dynamic_llm_modeling_of_subjective_preferences_in_grou/">Consensus vs. Dissent: Dynamic LLM Modeling of Subjective Preferences in Group Recommenders</a></strong></dt>
<dd>
<p>用人类问卷+DeepSeek-V3.1 蒸馏 CoT 微调 Llama/OLMo 作实时判断模型，动态挑选社会选择聚合策略，用户研究(n=284)中公平与共识得分最高。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/whose_posts_get_ranked_identical_text_exposure_gaps_in_bluesky_custom_feeds/">Whose Posts Get Ranked: Identical-Text Exposure Gaps in Bluesky Custom Feeds</a></strong></dt>
<dd>
<p>基于自然重复帖的曝光审计：同文本跨作者在250个Bluesky自定义Feed中曝光差异达33%，作者历史主导排名</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/social_group_rec/frog_effective_friend_recommendation_in_online_games_via_modality_aware_use/">FROG: Effective Friend Recommendation in Online Games via Modality-aware User Preferences</a></strong></dt>
<dd>
<p>面向在线游戏好友推荐，用模态感知双向匹配+局部/全局双路注意力建模用户偏好，离线 HR/NDCG 最高提升 15.82%/14.59%，线上 accRate 提升 33.3%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/social_group_rec/invariance_matters_empowering_social_recommendation_via_graph_invariant_lea/">Invariance Matters: Empowering Social Recommendation via Graph Invariant Learning</a></strong></dt>
<dd>
<p>用图不变学习替代社交图硬过滤：多环境生成器+不变风险最小化+对抗环境探索，Full-Ranking 下 NDCG@20 较 GBSR 提升 5.4%–8.7%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/social_group_rec/score_based_generative_diffusion_models_for_social_recommendations/">Score-based Generative Diffusion Models for Social Recommendations</a></strong></dt>
<dd>
<p>用 SDE 扩散模型在隐空间直接生成协同一致的社交表征，配合课程训练与跨域对比学习，Full-Ranking 下 Recall@10 相对提升 1.85%–3.07%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/social_group_rec/quantifying_and_minimizing_perception_gap_in_social_networks/">Quantifying and Minimizing Perception Gap in Social Networks</a></strong></dt>
<dd>
<p>提出连续意见下的感知差距指数 PGI，用谱图理论刻画网络连通性与社群结构对局部-全局认知偏差的影响，并证明连边推荐最小化 PGI 的不可近似性，给出近最优贪心启发式。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/debias_fairness_industrial/multi_agent_attacks_for_black_box_social_recommendations/">Multi-agent Attacks for Black-box Social Recommendations</a></strong></dt>
<dd>
<p>用多智能体强化学习协同生成冷启动物品画像与跨社区社交画像，对黑盒社交推荐发起无目标投毒/规避攻击，全库排序下 NDCG@10 相对下降约 19%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/balancing_fairness_and_high_match_rates_in_reciprocal_recommender_systems_a/">Balancing Fairness and High Match Rates in Reciprocal Recommender Systems: A Nash Social Welfare Approach</a></strong></dt>
<dd>
<p>将双边匹配推荐建模为推荐机会分配，用 Nash 社会福利交替优化实现近无嫉妒的公平推荐，并以 Sinkhorn 加速至大规模场景。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/social_group_rec/leveraging_member_group_relations_via_multi_view_graph_filtering_for_effect/">Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation</a></strong></dt>
<dd>
<p>提出免训练多视图图滤波 Group-GF，用三张物品相似图与多项式滤波器聚合成员-群组关系，HR@10 最高提升 16.4% 且运行时间仅 1.55s。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/social_group_rec/virtual_stars_real_fans_understanding_the_vtuber_ecosystem/">Virtual Stars, Real Fans: Understanding the VTuber Ecosystem</a></strong></dt>
<dd>
<p>基于Bilibili 270万场直播的VTuber核心粉丝（会员）行为测量研究，逆向构建FanRanker将潜在付费会员推荐给主播，Top-50命中85%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/social_group_rec/from_aggregation_to_selection_user_validated_distributed_social_recommendat/">From Aggregation to Selection: User-Validated Distributed Social Recommendation</a></strong></dt>
<dd>
<p>将分布式社交推荐从模型聚合转为用户驱动的算法选择+验证者多数共识投票，提出 Acc@K 共识评测，四数据集 Acc@2 平均提升 4.51%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/social_group_rec/pulse_socially_aware_user_representation_modeling_toward_parameter_efficien/">PULSE: Socially-Aware User Representation Modeling Toward Parameter-Efficient Graph Collaborative Filtering</a></strong></dt>
<dd>
<p>用社区结构与社交邻居物品信号替代显式用户嵌入，参数降 50% 且 NDCG@20 提升 8-9%</p>
</dd>
</dl>

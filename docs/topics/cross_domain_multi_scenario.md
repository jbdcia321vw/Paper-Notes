# 🌐 跨域与多场景推荐 (Cross-Domain & Multi-Scenario Recommendation)

> **研究范畴**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

当前方向共收录 **72** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/cross_domain_multi_scenario/behavior_importance_aware_graph_neural_architecture_search_for_cross_domain/">Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用一次性免重训超网为各域定制 GNN 架构，并以双层隐式梯度学习源域行为重要性权重，Amazon 与微信广告数据 AUC 最高提升 10.57%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/cross_domain_multi_scenario/one_for_dozens_adaptive_recommendation_for_all_domains_with_counterfactual_/">One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation</a></strong></dt>
<dd>
<p>用分层专家+彩票假设掩码剪枝为数十域学专属子网，并以流行度反事实增强少数域，DomainAUC 提升 2‰~8.4‰。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/cross_domain_multi_scenario/from_ids_to_semantics_a_generative_framework_for_cross_domain_recommendatio/">From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization</a></strong></dt>
<dd>
<p>用领域自适应语义ID分词+对称动态路由的生成式跨域推荐框架，Leave-Last-Out全库排序下全面超越SOTA。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/generative_rec/tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi_domain/">Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation</a></strong></dt>
<dd>
<p>提出 TokenMoE+码本+互信息校准的统一物品分词框架 UniTok，单模型跨 10 域训练，NDCG@10 最高提升 51.89%，参数量降 9.63×。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/cross_domain_multi_scenario/horizon_a_benchmark_for_in_the_wild_user_behaviour_modeling/">HORIZON: A Benchmark for In-the-wild User Behaviour Modeling</a></strong></dt>
<dd>
<p>重构 Amazon Reviews 为 54M 用户跨域长程基准，提出时间-用户双轴四象限评估与 LLM 查询改写/长程建模任务，揭示 ID 模型在时间外推下崩塌。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/cross_domain_multi_scenario/leveraging_multimodal_data_and_side_users_for_diffusion_cross_domain_recomm/">Leveraging Multimodal Data and Side Users for Diffusion Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用 MLLM 抽取多模态特征做初步对齐，再以边侧用户+重叠用户两阶段扩散生成目标域冷启动用户向量，Amazon 三任务 MAE 最高降 13.6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/a_soft_partitioned_semi_supervised_collaborative_transfer_learning_approach/">A Soft-partitioned Semi-supervised Collaborative Transfer Learning Approach for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用软分区动态参数缓解主导域压制、用主导域伪标签加权迁移缓解稀疏域过拟合，离线全库AUC与美团线上A/B双验证。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/benefit_from_rich_tackling_search_interaction_sparsity_in_search_enhanced_r/">Benefit from Rich: Tackling Search Interaction Sparsity in Search Enhanced Recommendation</a></strong></dt>
<dd>
<p>用 LLM+RQ-VAE 将用户搜索/推荐偏好离散为码本，构建 User-Code 图做消息传递，把搜索富用户信号传给稀疏用户，Qilin 上 HR@1 相对 UniSAR 提升 12.1%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/global_distribution_aware_scenario_specific_variational_representation_lear/">Global-Distribution Aware Scenario-Specific Variational Representation Learning Framework</a></strong></dt>
<dd>
<p>用变分推断为每个场景下的用户/物品学习分布表征，并以全局行为模式与物品属性构造先验，缓解稀疏场景表征欠学习，即插即用提升 PEPNet/MARIA。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/scenario_wise_rec_a_multi_scenario_recommendation_benchmark/">Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark</a></strong></dt>
<dd>
<p>首个多场景推荐统一基准，整合 6 个公开数据集与 12 个 MSR 模型，统一数据处理/训练/评测管线并附工业广告数据验证。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/a_dual_expert_strategy_integrating_llms_to_mitigate_negative_transfer_in_cr/">A Dual-Expert Strategy Integrating LLMs to Mitigate Negative Transfer in Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用域门控双专家+物品级注意力掩码+双池负采样对比学习，抑制 LLM 跨域序列推荐中的负迁移，10 域上超越 26 个基线，线上 CTR 相对提升 47.6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/cross_country_code_mixing_for_generative_recommendation/">Cross-Country Code-Mixing for Generative Recommendation</a></strong></dt>
<dd>
<p>借鉴多语言 code-switching，用双约束 token 级跨国混码合成混合序列并做上下文自适应重加权，在无重叠 ID 的跨国场景显著提升小国召回。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/cross_domain_sequential_recommendation_with_representation_level_user_trans/">Cross-Domain Sequential Recommendation with Representation-Level User Transfer</a></strong></dt>
<dd>
<p>仅通过用户表征级迁移实现跨域序列推荐，融合层注入源域用户向量，兼容纵向联邦与部分重叠用户，全库排序下 NDCG@10 提升 7.4–24.1%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/fedcgr_federated_cross_domain_generative_recommendation/">FedCGR: Federated Cross-Domain Generative Recommendation</a></strong></dt>
<dd>
<p>以固定 SID 词表为跨域通信接口，用可靠性感知 CF 残差与原型个性化聚合解决联邦生成式 CDR 的语义瓶颈与负迁移，全库排序下显著超越联邦生成基线。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/lfm4ads_large_foundation_model_for_ads_recommendation/">LFM4Ads: Large Foundation Model for Ads Recommendation</a></strong></dt>
<dd>
<p>腾讯广告提出预训练-迁移范式，联合迁移用户/物品/交叉表征，以时间感知EMA聚合样本级CR解耦上游推理，三级粒度下游复用，离线AUC+0.0189、线上GMV+2.45%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/collaborative_filtering/transretrieval_scaling_up_transformer_based_retrieval_for_industrial_recomm/">TransRetrieval: Scaling Up Transformer-Based Retrieval for Industrial Recommendation</a></strong></dt>
<dd>
<p>用加权平均聚合+目标Token压缩+位置式域嵌入，让Transformer在召回阶段实现对数线性Scaling，线上收入+2.53%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/ctr_feature_interaction/uniscale_synergistic_entire_space_data_and_model_scaling_for_search_ranking/">UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking</a></strong></dt>
<dd>
<p>提出数据-架构协同缩放框架：ES³全空间样本系统扩展训练信号，HHSFT异构分层Transformer吸收跨域分布，离线AUC+1.14%、线上GMV+2.04%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/cross_domain_multi_scenario/a_unified_framework_for_adaptive_representation_enhancement_and_inversed_le/">A Unified Framework for Adaptive Representation Enhancement and Inversed Learning in Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>解耦共享/专属表征，用 LightGCN 域内增强 + 自注意力跨域自适应融合 + GRL 逆向约束，全库排序下 Recall@20 最高提升 8.31%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/cross_domain_multi_scenario/cross_domain_causal_preference_learning_for_out_of_distribution_recommendat/">Cross-Domain Causal Preference Learning for Out-of-Distribution Recommendation</a></strong></dt>
<dd>
<p>用域对抗抽取域共享偏好、DAG 学习潜属性因果结构，缓解目标域稀疏下的 OOD 泛化，OOD 测试 HR@10 提升约 2-6%。</p>
</dd>
<dt><strong><a href="../../venues/ICDE/2025/cross_domain_multi_scenario/personalized_multi_interest_modeling_for_cross_domain_recommendation_to_col/">Personalized Multi-Interest Modeling for Cross-Domain Recommendation to Cold-Start Users</a></strong></dt>
<dd>
<p>用归一化流增强神经过程，将单峰高斯先验转为多峰分布以建模用户多兴趣，并引入偏好池捕获跨用户共同偏好，五场景 MAE 平均降约 45%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2025/cross_domain_multi_scenario/adversarial_alignment_and_disentanglement_for_cross_domain_ctr_prediction_w/">Adversarial Alignment and Disentanglement for Cross-Domain CTR Prediction with Domain-Encompassing Features</a></strong></dt>
<dd>
<p>提出 A²DCDR，用 DC-MMD 对抗对齐 + CLUB 互信息解耦 + 特征重构，保留域不变与有价值非对齐特征，四组 Amazon 场景 HR@10 最高 +6.96，线上 CTR +7.18%。</p>
</dd>
<dt><strong><a href="../../venues/ICDM/2026/sequential_rec/user_representation_via_cross_multi_source_behavior_pre_training_for_mobile/">User Representation via Cross Multi-source Behavior Pre-training for Mobile Games</a></strong></dt>
<dd>
<p>面向设备级多源多粒度行为日志，提出跨源级联掩码预训练 CM-PTM，以源→类目→App→操作逐级预测统一建模跨源依赖，离线 AUC 提升 1.05%–2.85%，线上 A/B 提升 1.28%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/effective_two_stage_knowledge_transfer_for_multi_entity_cross_domain_recomm/">Effective Two-Stage Knowledge Transfer for Multi-Entity Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>提出 MKT 两阶段多实体知识迁移框架，用异构特征对齐与极化分布损失解决异质特征 schema 下的负迁移，线上 CTR +4.13%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/exploring_preference_guided_diffusion_model_for_cross_domain_recommendation/">Exploring Preference-Guided Diffusion Model for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用源域交互历史构造偏好引导信号，逐步注入扩散逆向过程生成目标域冷启动用户表征，三场景 MAE 最高降 32.60%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/measure_domain_s_gap_a_similar_domain_selection_principle_for_multi_domain_/">Measure Domain&#x27;s Gap: A Similar Domain Selection Principle for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用原型距离+bandit动态为每个域挑选相似域子集，解耦共享专家以缓解多域推荐负迁移，三数据集AUC稳定提升。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/perscen_learning_personalized_interaction_pattern_and_scenario_preference_f/">PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching</a></strong></dt>
<dd>
<p>用用户专属特征图 + VQ 码本蒸馏场景感知偏好 + 渐进式 GLU 融合，在 KuaiRand/Alimama 四场景召回上全面超越 ADIN/SASS/M5。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/revisiting_self_attention_for_cross_domain_sequential_recommendation/">Revisiting Self-attention for Cross-domain Sequential Recommendation</a></strong></dt>
<dd>
<p>将跨域序列推荐重构为推荐损失与跨域注意力抑制的双目标帕累托优化，以即插即用方式改造自注意力，平均提升 Recall@10 达 9.8%/16.0%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/atomic_intent_reasoning_bringing_llm_semantics_to_industrial_cross_domain_r/">Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations</a></strong></dt>
<dd>
<p>将 LLM 意图推理离线化为原子行为-意图对缓存，在线经意图树构建与目标感知检索实现约 400× 加速，快手 A/B 测试 GMV +3.446%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/meta_lattice_model_space_redesign_for_cost_effective_industry_scale_ads_rec/">Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations</a></strong></dt>
<dd>
<p>以模型空间重构统一碎片化广告组合，融合归因窗口混合、帕累托特征选择与推理期知识迁移，线上Topline +10%、容量省20%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/sharpness_aware_model_merging_with_salience_recovery_for_llm_based_cross_do/">Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用锐度感知平坦极小值对齐 + 重尾重参数化恢复显著偏好信号，破解 LLM 跨域序列推荐模型合并中的参数冲突与多域饱和瓶颈。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/affect_aware_cross_domain_recommendation_for_art_therapy_via_music_preferen/">Affect-aware Cross-Domain Recommendation for Art Therapy via Music Preference Elicitation</a></strong></dt>
<dd>
<p>以音乐偏好诱导替代纯视觉诱导，提出 Mozart/Haydn/Salieri 三种情感感知跨域艺术治疗推荐，200 人用户研究验证情绪改善。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/llm_rec/heterogeneous_user_modeling_for_llm_based_recommendation/">Heterogeneous User Modeling for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用压缩提示+[USER]锚点token将跨域异构行为压成单一用户表征，并以域重要性平滑加权缓解域跷跷板，六域全库排序R@10最高提升约30%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/impacts_of_mainstream_driven_algorithms_on_recommendations_for_children_acr/">Impacts of Mainstream-Driven Algorithms on Recommendations for Children Across Domains: A Reproducibility Study</a></strong></dt>
<dd>
<p>跨电影/音乐/图书三域复现儿童与主流用户偏好偏差，并引入流行度偏差指标，揭示推荐算法对儿童群体的域特异性失效。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/leave_no_one_behind_fairness_aware_cross_domain_recommender_systems_for_non/">Leave No One Behind: Fairness-Aware Cross-Domain Recommender Systems for Non-Overlapping Users</a></strong></dt>
<dd>
<p>为无源域数据的非重叠用户生成虚拟源域画像，双注意力+限制器实现模型无关的跨域公平性纠偏，UGF最高降94.13%且整体精度不降。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/rankgraph_unified_heterogeneous_graph_learning_for_cross_domain_recommendat/">RankGraph: Unified Heterogeneous Graph Learning for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>构建跨产品异构异质图，用 GPU 加速 RGCN + 对比学习产出图 token 注入推荐基础模型，线上 CTR +0.92%、CVR +2.82%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/zero_shot_cross_domain_knowledge_distillation_a_case_study_on_youtube_music/">Zero-shot Cross-domain Knowledge Distillation: A Case study on YouTube Music</a></strong></dt>
<dd>
<p>用 YouTube 视频大教师零样本蒸馏到低流量 YouTube Music 双排序模型，离线 AUC 微增但线上新发内容互动 +11.39%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/empowering_cross_domain_sequential_recommendation_with_hybrid_tokenization_/">Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding</a></strong></dt>
<dd>
<p>跨域共享-专属多塔 RQ-VAE 混合分词 + 两段式串并解码，Leave-One-Out 全库排序下精度 +1.5%、时延 -85.1%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/adaptive_domain_scaling_for_personalized_sequential_modeling_in_recommender/">Adaptive Domain Scaling for Personalized Sequential Modeling in Recommenders</a></strong></dt>
<dd>
<p>用域特征动态生成序列物品的个性化MLP权重与候选物品的多查询表示，改造目标注意力实现多域序列建模，抖音双场景线上GMV+0.79%、ADVV+1.00%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/adaptive_graph_integration_for_cross_domain_recommendation_via_heterogeneou/">Adaptive Graph Integration for Cross-Domain Recommendation via Heterogeneous Graph Coordinators</a></strong></dt>
<dd>
<p>用异质自适应图协调器把多域图拼成统一结构做多域预训练，再以图提示迁移到目标域，Full-Ranking 下全面超越 9 个基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/agent_conversational_rec/agentcf_memory_enhanced_llm_based_agents_for_popularity_aware_cross_domain_/">AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations</a></strong></dt>
<dd>
<p>以双层记忆+两步融合+兴趣组共享记忆，令 LLM 用户智能体在跨域场景显式建模流行度影响，采样排序 MRR 平均超 AgentCF 14.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/bridge_the_domains_large_language_models_enhanced_cross_domain_sequential_r/">Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义嵌入+分层画像桥接跨域物品与用户偏好，破解重叠困境与转移复杂性，三数据集 H@10 最高提升 31.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/cdc_causal_domain_clustering_for_multi_domain_recommendation/">CDC: Causal Domain Clustering for Multi-Domain Recommendation</a></strong></dt>
<dd>
<p>用孤立/混合双亲和矩阵+因果发现系数自适应融合，联合优化目标域簇与训练源域集，50+域上DomainAUC提升4.14‰/6.76‰，线上eCPM+4.9%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/learning_universal_user_representations_leveraging_cross_domain_user_intent/">Learning Universal User Representations Leveraging Cross-domain User Intent at Snapchat</a></strong></dt>
<dd>
<p>将 Snapchat 多业务面事件拼接为统一序列训练 Transformer 用户塔，产出跨域通用用户表征，线上多场景 A/B 全面正向。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/x_cross_dynamic_integration_of_language_models_for_cross_domain_sequential_/">X-Cross: Dynamic Integration of Language Models for Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>逐层动态集成多个 LoRA 源域语言模型，仅用 25% 参数与 50-75% 更少微调数据即达/超 LoRA 跨域序列推荐效果。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/bridging_behavior_and_semantics_for_time_aware_cross_domain_sequential_reco/">Bridging Behavior and Semantics for Time-aware Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用神经 ODE 解耦长短期行为、LLM 离散时间 token 反事实增强语义、门控自适应域迁移，1:999 采样排序下 MRR 最高提升 30.5%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/fedmm_federated_collaborative_signal_quantization_for_multi_market_ctr_pred/">FedMM: Federated Collaborative Signal Quantization for Multi-Market CTR Prediction</a></strong></dt>
<dd>
<p>用双层 RQ-VAE 码本作联邦传输介质，对齐跨市场不相交 ID 空间，AUC 全面超越本地/联邦基线且通信量降 20×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong></dt>
<dd>
<p>用可迁移物品增广+双阶段训练+域感知画像，缓解CDSR域不平衡与域迁移难题，全库排序下H@10最高提升26.96%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/reinforcing_user_interest_evolution_in_multi_scenario_learning_for_recommen/">Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems</a></strong></dt>
<dd>
<p>用 Double Q-learning 估计场景级行为置信度作为门控，加权三元组对比损失以抑制多场景负迁移，KuaiSAR 全库排序 N@20 提升约 3.3 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/universal_item_tokenization_for_transferable_generative_recommendation/">Universal Item Tokenization for Transferable Generative Recommendation</a></strong></dt>
<dd>
<p>用 MLLM+树状码本构建跨域通用物品分词器，结合内容重建与共现协同对齐，实现可迁移生成式推荐，全库排序下全面超越基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR-AP/2025/cross_domain_multi_scenario/marco_a_cooperative_knowledge_transfer_framework_for_personalized_cross_dom/">MARCO: A Cooperative Knowledge Transfer Framework for Personalized Cross-domain Recommendations</a></strong></dt>
<dd>
<p>将多源跨域推荐建模为协作式多智能体强化学习，每源域一个 Agent 估计贡献并加熵正则，冷启动 MAE 最高降 32.6%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/cross_domain_multi_scenario/automated_prompting_for_non_overlapping_cross_domain_sequential_recommendat/">Automated Prompting for Non-overlapping Cross-domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用可学习连续 Prompt（域无关+域特定+冻结标签）替代域对齐约束，实现无重叠实体、仅隐式反馈的双目标跨域序列推荐，Movie-Book 上 HR@10 相对最优基线提升约 17.8%。</p>
</dd>
<dt><strong><a href="../../venues/TMM/2025/cross_domain_multi_scenario/federated_user_preference_modeling_for_privacy_preserving_cross_domain_reco/">Federated User Preference Modeling for Privacy-Preserving Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用评论+潜在正物品挖掘综合偏好，再以 LDP 原型经联邦聚合跨域迁移，四组 CDR 任务 HR@10 平均提升约 13%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/ctr_feature_interaction/a_unified_framework_for_multi_domain_ctr_prediction_via_large_language_mode/">A Unified Framework for Multi-Domain CTR Prediction via Large Language Models</a></strong></dt>
<dd>
<p>用 LLM 分层语义表征作跨域共享底座、可插拔 DSN 建模域特性，掩码损失解耦梯度，AUC 相对提升 24%–63% 并支持零样本新域。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/cross_domain_multi_scenario/causal_deconfounding_via_confounder_disentanglement_for_dual_target_cross_d/">Causal Deconfounding via Confounder Disentanglement for Dual-Target Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>显式解耦单域/跨域观测混杂因子，经后门调整保留正向效应并消除负向偏差，七数据集 HR@10 平均提升 6.17%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/cross_domain_multi_scenario/coherence_guided_preference_disentanglement_for_cross_domain_recommendation/">Coherence-guided Preference Disentanglement for Cross-domain Recommendations</a></strong></dt>
<dd>
<p>用全局-局部图 + 一致性约束抽取跨域共享物品属性，并以流行度加权 Softmax 将共享用户偏好解耦为兴趣/从众，六组双目标 CDR 任务 HR@10 最高提升约 37%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/cross_domain_multi_scenario/federated_semantic_learning_for_privacy_preserving_cross_domain_recommendat/">Federated Semantic Learning for Privacy-preserving Cross-domain Recommendation</a></strong></dt>
<dd>
<p>提出 FFMSR，用多层 PLM+MoE 提取语义、服务器端聚类融合跨域语义、FFT 滤波去噪，在完全非重叠联邦场景下 Full-Ranking 提升 1.4%–5.6%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/llm_rec/one_model_for_all_large_language_models_are_domain_agnostic_recommendation_/">One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems</a></strong></dt>
<dd>
<p>用 LLM 编码跨域混合标题序列实现域无关推荐，采样 1000 负例评测，SPIAO 上较最优基线相对提升约 19%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/cross_domain_multi_scenario/causal_invariant_cross_domain_out_of_distribution_recommendation/">Causal-Invariant Cross-Domain Out-of-Distribution Recommendation</a></strong></dt>
<dd>
<p>统一建模跨域与单域双重分布偏移，用双层因果结构与 LLM 引导混杂因子发现实现跨域 OOD 推荐，HR@10 平均提升 6.34%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/cross_domain_multi_scenario/review_based_hyperbolic_cross_domain_recommendation/">Review-Based Hyperbolic Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>将评论文本嵌入双曲空间做跨域推荐，用度数归一化与尺度对齐避免域对齐引发的层级坍塌，NDCG@10 较 SER 提升约 10.4%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/cross_domain_multi_scenario/towards_personalized_federated_multi_scenario_multi_task_recommendation/">Towards Personalized Federated Multi-Scenario Multi-Task Recommendation</a></strong></dt>
<dd>
<p>首个联邦多场景多任务推荐框架 PF-MSMTrec，通过专家参数解耦、联邦批归一化与冲突协调缓解数据/模型/任务三级冲突，AliExpress 与 Tenrec 上 CTR/CTCVR AUC 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/cross_domain_multi_scenario/automated_information_flow_selection_for_multi_scenario_multi_task_recommen/">Automated Information Flow Selection for Multi-scenario Multi-task Recommendation</a></strong></dt>
<dd>
<p>用 LoRA 解耦四类场景-任务信息单元并以可学习掩码剪枝无效信息流，AUC 全面超越 HiNet/M3oE，线上 CVR 均涨 2.46%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/abxi_invariant_interest_adaptation_for_task_guided_cross_domain_sequential_/">ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用任务引导对齐消除预测错配，并以 dLoRA/iLoRA 双 LoRA 抽取域不变兴趣，三数据集 MRR 大幅超越 CDSR 基线。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/image_fusion_for_cross_domain_sequential_recommendation/">Image Fusion for Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结 CLIP 图像嵌入补足 ID 表征，并以多注意力分别建模域内/跨域序列，在 Amazon 双域场景 MRR 提升约 5%–18%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/pre_train_and_fine_tune_recommenders_as_large_models/">Pre-train and Fine-tune: Recommenders as Large Models</a></strong></dt>
<dd>
<p>以信息瓶颈理论解释推荐模型微调，提出即插即用的信息感知自适应核 IAK，冻结十亿参数预训练推荐大模型，仅用一周域内数据微调即获 CTR-AUC 平均 +0.26%~+1.28% 与线上 GMV +0.74%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/separated_contrastive_learning_for_matching_in_cross_domain_recommendation_/">Separated Contrastive Learning for Matching in Cross-domain Recommendation with Curriculum Scheduling</a></strong></dt>
<dd>
<p>将 intra-CL 与 inter-CL 解耦为两阶段课程训练，配合 stop-gradient 与基于 Katz 中心性的负样本难度调度，稳定跨域召回训练并显著提升 HIT@100。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/uncovering_cross_domain_recommendation_ability_of_large_language_models/">Uncovering Cross-Domain Recommendation Ability of Large Language Models</a></strong></dt>
<dd>
<p>用源域购买序列+跨域引导提示驱动 LLM 零样本重排目标域候选，揭示域间隙与模型规模对跨域迁移的决定性影响。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/fedecider_an_llm_based_framework_for_federated_cross_domain_recommendation/">FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>联邦跨域 LLM 推荐中，服务器只广播 LoRA 更新的方向分量以消除尺度噪声，客户端用可学习个性化权重做数据感知融合，Goodreads 平均 Hit@5 提升 3.8%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/generative_data_transformation_from_mixed_to_unified_data/">Generative Data Transformation: From Mixed to Unified Data</a></strong></dt>
<dd>
<p>提出数据侧范式 Taesar，用三模型对比解码把跨域混合序列重生成纯目标域序列，免改架构即插即用，Books HR@10 提升 29.4%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/s2cdr_smoothing_sharpening_process_model_for_cross_domain_recommendation/">S2CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用无噪声热方程平滑+锐化ODE替代扩散模型做跨域冷启动，免训练即在三场景六跨域对上 HR/NDCG 平均提升 7.07%/7.42%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/the_double_edged_sword_of_knowledge_transfer_diagnosing_and_curing_fairness/">The Double-Edged Sword of Knowledge Transfer: Diagnosing and Curing Fairness Pathologies in Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>理论揭示跨域推荐中“差异迁移”与“信息增益不均”两类公平性病理，提出模型无关的 CDFA 框架，在 Tenrec QB/QK 上 UGF 最高降 98.97% 且精度不降。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cold_start_rec/trinity_a_scenario_aware_recommendation_framework_for_large_scale_cold_star/">Trinity: A Scenario-Aware Recommendation Framework for Large-Scale Cold-Start Users</a></strong></dt>
<dd>
<p>面向十亿级产品迁移的“特征-架构-更新”三位一体冷启动框架，Copilot 新场景 AUC 0.564→0.726、COPC 0.13→0.95，线上时长 +5.61%。</p>
</dd>
</dl>

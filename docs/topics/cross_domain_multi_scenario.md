# 🌐 跨域与多场景推荐 (Cross-Domain & Multi-Scenario Recommendation)

> **研究范畴**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

当前方向共收录 **28** 篇顶会/顶刊论文解读：

<dl>
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
<dt><strong><a href="../../venues/ICDM/2026/sequential_rec/user_representation_via_cross_multi_source_behavior_pre_training_for_mobile/">User Representation via Cross Multi-source Behavior Pre-training for Mobile Games</a></strong></dt>
<dd>
<p>面向设备级多源多粒度行为日志，提出跨源级联掩码预训练 CM-PTM，以源→类目→App→操作逐级预测统一建模跨源依赖，离线 AUC 提升 1.05%–2.85%，线上 A/B 提升 1.28%。</p>
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
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/empowering_cross_domain_sequential_recommendation_with_hybrid_tokenization_/">Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding</a></strong></dt>
<dd>
<p>跨域共享-专属多塔 RQ-VAE 混合分词 + 两段式串并解码，Leave-One-Out 全库排序下精度 +1.5%、时延 -85.1%。</p>
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
<dt><strong><a href="../../venues/TOIS/2026/cross_domain_multi_scenario/causal_invariant_cross_domain_out_of_distribution_recommendation/">Causal-Invariant Cross-Domain Out-of-Distribution Recommendation</a></strong></dt>
<dd>
<p>统一建模跨域与单域双重分布偏移，用双层因果结构与 LLM 引导混杂因子发现实现跨域 OOD 推荐，HR@10 平均提升 6.34%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/cross_domain_multi_scenario/automated_information_flow_selection_for_multi_scenario_multi_task_recommen/">Automated Information Flow Selection for Multi-scenario Multi-task Recommendation</a></strong></dt>
<dd>
<p>用 LoRA 解耦四类场景-任务信息单元并以可学习掩码剪枝无效信息流，AUC 全面超越 HiNet/M3oE，线上 CVR 均涨 2.46%。</p>
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

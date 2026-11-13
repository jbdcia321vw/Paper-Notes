# 🌐 跨域与多场景推荐 (Cross-Domain & Multi-Scenario Recommendation)

> **研究范畴**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

当前方向共收录 **13** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/a_dual_expert_strategy_integrating_llms_to_mitigate_negative_transfer_in_cr/">A Dual-Expert Strategy Integrating LLMs to Mitigate Negative Transfer in Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用域门控双专家+物品级注意力掩码+双池负采样对比学习，抑制 LLM 跨域序列推荐中的负迁移，10 域上超越 26 个基线，线上 CTR 相对提升 47.6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/generative_rec/cross_country_code_mixing_for_generative_recommendation/">Cross-Country Code-Mixing for Generative Recommendation</a></strong></dt>
<dd>
<p>借鉴多语言 code-switching，用双约束 token 级跨国混码合成混合序列并做上下文自适应重加权，在无重叠 ID 的跨国场景显著提升小国召回。</p>
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
<dt><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/bridging_behavior_and_semantics_for_time_aware_cross_domain_sequential_reco/">Bridging Behavior and Semantics for Time-aware Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用神经ODE建模连续时间行为偏好演化，LLM+反事实扰动提取时间敏感语义，门控自适应控制跨域迁移权重以缓解负迁移。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Models Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong></dt>
<dd>
<p>用 LLM 生成可迁移跨域行为+噪声过滤+两阶段训练+域感知画像，缓解域不平衡与域迁移问题，三数据集全排序 Recall/NDCG 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/reinforcing_user_interest_evolution_in_multi_scenario_learning_for_recommen/">Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems</a></strong></dt>
<dd>
<p>用 Double Q-learning 估计场景置信度作为门控，加权三元组对比损失建模跨场景兴趣演化，KuaiSAR 全库排序 NDCG@20 大幅领先。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/universal_item_tokenization_for_transferable_generative_recommendation/">Universal Item Tokenization for Transferable Generative Recommendation</a></strong></dt>
<dd>
<p>基于MLLM与树状码本的通用物品分词器，融合多模态重建与协同对齐，实现跨域可迁移生成式推荐。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/cross_domain_multi_scenario/automated_information_flow_selection_for_multi_scenario_multi_task_recommen/">Automated Information Flow Selection for Multi-scenario Multi-task Recommendation</a></strong></dt>
<dd>
<p>用 LoRA 解耦四类场景-任务信息单元并以可学习掩码剪枝无效信息流，AUC 全面超越 HiNet/M3oE，线上 CVR 均涨 2.46%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/s2cdr_smoothing_sharpening_process_model_for_cross_domain_recommendation/">S2CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation</a></strong></dt>
<dd>
<p>用无噪声热方程平滑+锐化ODE替代扩散模型做跨域冷启动，免训练即在三场景六跨域对上 HR/NDCG 平均提升 7.07%/7.42%</p>
</dd>
</dl>

# 🤝 协同过滤与双塔召回 (Collaborative Filtering & Two-Tower Retrieval)

> **研究范畴**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

当前方向共收录 **144** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/AAAI/2025/collaborative_filtering/from_pairwise_to_ranking_climbing_the_ladder_to_ideal_collaborative_filteri/">From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking</a></strong></dt>
<dd>
<p>用噪声注入构造伪排序并以多序分类排序损失替代成对损失，在四个数据集 Full-Ranking 下 NDCG@10 最高提升 38.9%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/debias_fairness_industrial/personalized_federated_collaborative_filtering_a_variational_autoencoder_ap/">Personalized Federated Collaborative Filtering: A Variational AutoEncoder Approach</a></strong></dt>
<dd>
<p>将 FedCF 重构为 VAE 任务，用门控双编码器分离共享/个性化知识，HR@20 较 FedRAP 提升约 97%。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2025/multimodal_rec/stair_manipulating_collaborative_and_multimodal_information_for_e_commerce_/">STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation</a></strong></dt>
<dd>
<p>揭示电商场景模态信息弱于协同信号，提出逐步图卷积沿维度分配层权重实现协同-模态共存，并以反向逐步卷积约束更新缓解模态遗忘，三数据集全排序提升2-6%且开销近LightGCN。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/collaborative_filtering/crops_improving_dense_retrieval_with_cross_perspective_positive_samples_in_/">CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search</a></strong></dt>
<dd>
<p>用查询改写/推荐流/LLM 三源扩充正样本，配分层标签与 H-InfoNCE，快手搜索线上 CTR +0.87%、改写率 -0.65%</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/extracting_interaction_aware_monosemantic_concepts_in_recommender_systems/">Extracting Interaction-Aware Monosemantic Concepts in Recommender Systems</a></strong></dt>
<dd>
<p>用预测感知 SAE 从双塔嵌入中提取单义神经元，在 ML1M/Last.FM 上实现高纯度概念发现与事后可控干预。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/fidelity_aware_recommendation_explanations_via_stochastic_path_integration/">Fidelity-Aware Recommendation Explanations via Stochastic Path Integration</a></strong></dt>
<dd>
<p>将路径积分归因引入推荐解释，用随机基线采样适配稀疏二值交互，在三种模型三数据集上刷新反事实保真度基准。</p>
</dd>
<dt><strong><a href="../../venues/AAAI/2026/collaborative_filtering/generalization_bounds_for_semi_supervised_matrix_completion_with_distributi/">Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information</a></strong></dt>
<dd>
<p>提出隐式反馈采样分布与评分矩阵共享低秩子空间的半监督矩阵完成范式，证明误差可解耦为两项独立界，并在 ML-100K/Douban/Yelp 上验证。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/copl_collaborative_preference_learning_for_personalizing_llms/">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></strong></dt>
<dd>
<p>用用户-回复二部图协同过滤学习用户表征，结合 MoLE 路由个性化奖励模型，并支持无优化未见用户适配，在 UltraFeedback-P 等基准上显著超越个性化奖励模型基线。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/rethinking_llm_based_recommendations_a_personalized_query_driven_parallel_i/">Rethinking LLM-Based Recommendations: A Personalized Query-Driven Parallel Integration</a></strong></dt>
<dd>
<p>提出 QUEREC 并行框架，用 LLM 零样本生成个性化用户查询与特征增强物品描述，与 CF 模型并行全库检索后自适应重排，Full-Ranking 下最高提升 57%。</p>
</dd>
<dt><strong><a href="../../venues/ACL_EMNLP/2026/collaborative_filtering/clusterrag_cluster_based_collaborative_filtering_for_personalized_retrieval/">ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation</a></strong></dt>
<dd>
<p>用 HDBSCAN 将用户按画像嵌入聚类，簇内 ColBERTv2 排序选近邻，簇-文档两级检索融合协同画像做个性化 RAG，LaMP 全任务最优。</p>
</dd>
<dt><strong><a href="../../venues/ACM_MM/2025/debias_fairness_industrial/lego_a_lightweight_and_efficient_multiple_attribute_unlearning_framework_fo/">LEGO: A Lightweight and Efficient Multiple-Attribute Unlearning Framework for Recommender Systems</a></strong></dt>
<dd>
<p>将多属性遗忘拆为并行嵌入校准与权重组合两步，以互信息最小化保证同时遗忘，动态需求下近零额外开销。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/collaborative_filtering/a_node_aware_dynamic_quantization_approach_for_graph_collaborative_filterin/">A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering</a></strong></dt>
<dd>
<p>提出节点感知动态量化 GNAQ，用参数共享学习每节点量化步长并以图消息传递更新，2-bit 下 Recall@10 平均提升 27.8%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/a_self_supervised_mixture_of_experts_framework_for_multi_behavior_recommend/">A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation</a></strong></dt>
<dd>
<p>揭示多行为推荐在已访问/未访问物品上的性能鸿沟，提出双专家 MoE + 专家自适应 SSL（访问过滤/新颖性推断对比 + 行为生成）框架，全库排序下 HR@20 最高提升 65.46%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/graph_rec/enhancing_graph_collaborative_filtering_with_fourierkan_feature_transformat/">Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation</a></strong></dt>
<dd>
<p>用 Fourier KAN 替代 NGCF 中被误删的交互特征变换 W2，在 7:1:2 时间切分全库排序下 R@20 最高提升 7.8%，且可作即插即用骨干。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/evaluating_and_addressing_fairness_across_user_groups_in_negative_sampling_/">Evaluating and Addressing Fairness Across User Groups in Negative Sampling for Recommender Systems</a></strong></dt>
<dd>
<p>实证揭示负采样比 K 增大使活跃用户增益、非活跃用户受损，提出分组特定负采样比并用带先验 Hyperband 优化，全库排序下 NDCG 提升 2–10%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/llm_rec/llm_enhanced_linear_autoencoders_for_recommendation/">LLM-Enhanced Linear Autoencoders for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 语义矩阵构造 item-item 相关阵并以知识蒸馏正则注入 EASE 闭式解，全库排序下 R@20 平均提升 27.6%。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/multi_task_reranking/mpformer_adaptive_framework_for_industrial_multi_task_personalized_sequenti/">MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever</a></strong></dt>
<dd>
<p>用目标条件化 Transformer 统一多目标召回，并以排序分数监督动态配额分配，在快手 4 亿 DAU 上实现 +0.43% 总时长与 66.7% 推理资源节省。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/collaborative_filtering/patient_clustering_via_integrated_profiling_of_clinical_and_digital_data/">Patient Clustering via Integrated Profiling of Clinical and Digital Data</a></strong></dt>
<dd>
<p>用掩码约束非负低秩分解统一融合诊断码与门户浏览/搜索行为，同时产出可解释软聚类与患者嵌入，聚类与推荐双任务全面领先。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2025/cold_start_rec/shallow_autoencoding_recommender_with_cold_start_handling_via_side_features/">Shallow AutoEncoding Recommender with Cold Start Handling via Side Features</a></strong></dt>
<dd>
<p>将用户/物品侧特征拼入 EASE 输入矩阵，并用内容相似度矩阵作冷物品权重先验，闭式解统一处理双冷启动。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/collaborative_filtering/dualspectralcf_training_free_sign_aware_spectral_collaborative_filtering/">DualSpectralCF: Training-Free Sign-Aware Spectral Collaborative Filtering</a></strong></dt>
<dd>
<p>在任意谱滤波骨干上以带符号输入信号与带符号物品算子注入显式负反馈，免训练即插即用，全库排序下 R@20 最高提升 32.6%，比 SIGformer 快 7.7–155.3 倍。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/think_to_personalize_unifying_reasoning_and_retrieval_for_user_centric_pers/">Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval</a></strong></dt>
<dd>
<p>用 SFT+GRPO 两阶段让 LLM 显式推理用户历史生成意图增强查询并统一编码，Full-Ranking 下 Broad/LongTail 召回显著提升，线上订单量 +0.46%</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/llm_rec/training_free_llm_based_recommendation_with_post_llm_item_refinement_using_/">Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals</a></strong></dt>
<dd>
<p>提出 post-LLM 范式 CoRRe，用共购图方向细化与流行度幅度校准改造 LLM 物品表征，免训练即在三域 Amazon 上超越全部免训练基线并多数超越训练式方法。</p>
</dd>
<dt><strong><a href="../../venues/CIKM/2026/collaborative_filtering/transretrieval_scaling_up_transformer_based_retrieval_for_industrial_recomm/">TransRetrieval: Scaling Up Transformer-Based Retrieval for Industrial Recommendation</a></strong></dt>
<dd>
<p>用加权平均聚合+目标Token压缩+位置式域嵌入，让Transformer在召回阶段实现对数线性Scaling，线上收入+2.53%。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/collaborative_filtering/a_situation_aware_enhancer_for_personalized_recommendation/">A Situation-aware Enhancer for Personalized Recommendation</a></strong></dt>
<dd>
<p>将情境视为交互前置条件，用可插拔双塔增强器 SARE 建模个性化情境感知与偏好影响，在 MIND/KuaiRand 上显著超越骨干与 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/collaborative_filtering/learning_recommender_systems_with_soft_target_a_decoupled_perspective/">Learning Recommender Systems with Soft Target: A Decoupled Perspective</a></strong></dt>
<dd>
<p>将 Softmax 软标签目标解耦为目标置信度与未观测项兴趣分布两项，配合 K-means 邻居标签传播生成软标签，在四个骨干模型三个数据集上稳定提升。</p>
</dd>
<dt><strong><a href="../../venues/DASFAA/2025/debias_fairness_industrial/separating_and_learning_latent_confounders_to_enhancing_user_preferences_mo/">Separating and Learning Latent Confounders to Enhancing User Preferences Modeling</a></strong></dt>
<dd>
<p>将前序推荐系统作为未观测混杂代理，用变分推断在隐空间解耦用户偏好与混杂，实现通用去偏，五数据集 Recall@10 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/are_representation_disentanglement_and_interpretability_linked_in_recommend/">Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study</a></strong></dt>
<dd>
<p>复现5个推荐模型×4数据集，量化解耦度与可解释性，发现解耦与效果无关但与表示可解释性强正相关。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/embedding_cultural_diversity_in_prototype_based_recommender_systems/">Embedding Cultural Diversity in Prototype-based Recommender Systems</a></strong></dt>
<dd>
<p>在 ProtoMF 原型矩阵分解上引入 K-近邻原型过滤与原型正交分布正则，以物品国别为文化代理缓解流行度偏差，全库排序下长尾平均排名降 27%、HR@10 反升 2%。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2025/sequential_rec/saferec_self_attention_and_frequency_enriched_model_for_next_basket_recomme/">SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation</a></strong></dt>
<dd>
<p>在 SASRec 上并联频率感知打分模块，用物品出现频次与历史指示向量补足重复购买建模，Leave-One-Basket 全库排序下 Recall@10 最高提升 8%。</p>
</dd>
<dt><strong><a href="../../venues/ECIR/2026/collaborative_filtering/a_reproducible_and_fair_evaluation_of_partition_aware_collaborative_filteri/">A Reproducible and Fair Evaluation of Partition-aware Collaborative Filtering</a></strong></dt>
<dd>
<p>在统一用户留出划分与 TPE 调参下复现 FPSR/FPSR+，发现 BISM 在 Amazon-CDs 反超，FPSR 家族在长尾显著占优。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/graph_rec/contextgnn_beyond_two_tower_recommendation_systems/">ContextGNN: Beyond Two-Tower Recommendation Systems</a></strong></dt>
<dd>
<p>提出局部成对表征+全局双塔回退的混合GNN架构，在RelBench十个任务上平均MAP较最强成对基线提升20%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/collaborative_filtering/covariate_moderated_empirical_bayes_matrix_factorization/">Covariate-moderated Empirical Bayes Matrix Factorization</a></strong></dt>
<dd>
<p>用协变量调制的经验贝叶斯先验统一矩阵分解，MovieLens 100K 上 RMSE 全面优于 EBMF/MFAI/VAE 系基线。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_int/">FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens</a></strong></dt>
<dd>
<p>用解耦投影+RQ-VAE 将 CF 嵌入映射为冻结 LLM 词表中的离散 token，配合对比对齐实现免微调的语义可解释与性能增益。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/language_representations_can_be_what_recommenders_need_findings_and_potenti/">Language Representations Can be What Recommenders Need: Findings and Potentials</a></strong></dt>
<dd>
<p>揭示语言空间与推荐行为空间同构，提出纯语言表征的 AlphaRec，在 Amazon 三数据集 Full-Ranking 上超越 ID-based CF 6.8%–9.8%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/recflow_an_industrial_full_flow_recommendation_dataset/">RecFlow: An Industrial Full Flow Recommendation Dataset</a></strong></dt>
<dd>
<p>首个覆盖工业六阶段漏斗全流程的推荐数据集，含1.9B阶段样本，揭示训练-服务分布偏移并验证阶段样本作为难负例与级联蒸馏的显著收益。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/collaborative_filtering/recommendations_with_sparse_comparison_data_provably_fast_convergence_for_n/">Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization</a></strong></dt>
<dd>
<p>将成对比较反馈建模为低秩矩阵分解下的离散选择似然，证明稀疏数据下投影梯度下降在热启动邻域内指数收敛，样本复杂度近最优。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential/">CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation</a></strong></dt>
<dd>
<p>用可学习全局 K/V 池跨用户共享协同信息，将 KV cache 压缩至 0.8% 且精度不降，解码延迟随 batch 增长近乎平坦。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/collaborative_filtering/generalizing_linear_autoencoder_recommenders_with_decoupled_expected_quadra/">Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss</a></strong></dt>
<dd>
<p>将 EDLAE 目标推广为 DEQL，首次给出 b&gt;0 闭式解并用 Miller 矩阵逆定理把复杂度从 O(n^4) 降到 O(n^3)，在 9 个基准上稳定超越 b=0 基线。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/obliviate_efficient_unlearning_in_recommender_systems/">Obliviate: Efficient Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>两阶段低秩适配器+局部校准的推荐系统机器遗忘框架，20%删除率下兼顾遗忘完整性与效用，比全量重训快约 3 倍。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/rejuvenating_cross_entropy_loss_in_knowledge_distillation_for_recommender_s/">Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems</a></strong></dt>
<dd>
<p>揭示CE蒸馏损失与NDCG的闭包假设矛盾，提出师生协同子集拆分与自适应加权的RCE-KD，全库排序下NDCG@20最高提升11.76%。</p>
</dd>
<dt><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/collaborative_filtering/two_level_softmax_sampling_done_right_correcting_bias_from_size_imbalance_a/">Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion</a></strong></dt>
<dd>
<p>揭示两级 softmax 采样因簇大小失衡与簇内相似度离散导致的系统性偏差，提出 S-2LS/SD-2LS 校正，KL 保真度提升 1–4 个数量级且开销近乎为零。</p>
</dd>
<dt><strong><a href="../../venues/IJCAI/2026/generative_rec/dav_gen_end_to_end_generative_retrieval_via_draft_and_verify/">DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify</a></strong></dt>
<dd>
<p>将生成式检索重构为“起草-验证”双阶段单模型范式，以混合稀疏-稠密表征与复合损失统一召回与排序，工业搜索延迟降至70ms。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/multi_task_reranking/breaking_the_top_k_barrier_advancing_top_k_ranking_metrics_optimization_in_/">Breaking the Top-$K$Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems</a></strong></dt>
<dd>
<p>用量化分位数把 Top-K 截断转成阈值比较，推导 NDCG@K 的紧致平滑上界 SL@K，四数据集三骨干平均提升 6.03%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/embed_progressive_implicit_preference_in_unified_space_for_deep_collaborati/">Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering</a></strong></dt>
<dd>
<p>将多类隐式反馈按稀疏度映射为有序类别，用嵌套编码+嵌套OLR把渐进偏好压入统一空间，实现单次NNS一阶段排序。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/llm_rec/exploring_feature_based_knowledge_distillation_for_recommender_system_a_fre/">Exploring Feature-based Knowledge Distillation for Recommender System: A Frequency Perspective</a></strong></dt>
<dd>
<p>从频域视角揭示特征蒸馏等权缺陷，提出轻量图滤波重加权 FreqD，全库排序下 Recall@20 最高提升 18.2%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/gradients_as_an_action_towards_communication_efficient_federated_recommende/">Gradients as an Action: Towards Communication-Efficient Federated Recommender Systems via Adaptive Action Sharing</a></strong></dt>
<dd>
<p>将物品梯度聚类为少量“动作”共享传输，配合自适应聚类与带宽感知压缩，在 ML-100K/1M/Lastfm-2K 上以最高 96.88% 压缩率不损性能。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/high_precision_audience_expansion_via_extreme_classification_in_a_two_sided/">High Precision Audience Expansion via Extreme Classification in a Two-Sided Marketplace</a></strong></dt>
<dd>
<p>将 Airbnb 位置召回从矩形边界重构为 25M 级 S2 单元极端多分类，离线精度 +11.01%、线上未取消预订者 +0.17%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/iu4rec_interest_unit_based_product_organization_and_recommendation_for_e_co/">IU4Rec: Interest Unit-Based Product Organization and Recommendation for E-Commerce Platform</a></strong></dt>
<dd>
<p>闲鱼C2C限库存场景下以兴趣单元(IU)重构商品组织与两阶段推荐，IU-Boosted网络离线AUC+1.91%、线上CTR+2.90%</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/on_the_role_of_weight_decay_in_collaborative_filtering_a_popularity_perspec/">On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective</a></strong></dt>
<dd>
<p>揭示权重衰减本质是将流行度编码进嵌入模长，提出 PRISM 初始化策略替代权重衰减，NDCG 最高提升 4.77% 且训练加速 38.48%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/optimizing_recall_or_relevance_a_multi_task_multi_head_approach_for_item_to/">Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for Item-to-Item Retrieval in Recommendation</a></strong></dt>
<dd>
<p>用多任务蒸馏损失+多头I2I架构联合优化共互动召回与语义相关性，离线召回@500提升14.4%、L2主题相关性提升56.5%，并完成线上A/B验证。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/real_time_indexing_for_large_scale_recommendation_by_streaming_vector_quant/">Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever</a></strong></dt>
<dd>
<p>用流式 VQ 码本替代 HNSW/DR 静态索引，实现实时挂载、索引均衡与可修复，抖音全量上线显著提升用户参与。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2025/collaborative_filtering/synergizing_implicit_and_explicit_user_interests_a_multi_embedding_retrieva/">Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest</a></strong></dt>
<dd>
<p>Pinterest 工业级多嵌入召回框架，用可微聚类建模隐式兴趣、条件召回建模显式兴趣，线上 HF Repins +1.09%、A-Pincepts +0.81%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/collaborative_filtering/breaking_information_cocoons_a_hyperbolic_framework_for_balancing_explorati/">Breaking Information Cocoons: A Hyperbolic Framework for Balancing Exploration and Exploitation in Recommender Systems</a></strong></dt>
<dd>
<p>洛伦兹双曲空间对齐 LLM 语义与协同信号，并用 Dasgupta 代价驱动的无超参层次聚类实现用户可调探索-利用权衡，效用最高 +5.49%、多样性 +11.39%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/generative_rec/collaborative_memory_augmentation_for_generative_recommendation/">Collaborative Memory Augmentation for Generative Recommendation</a></strong></dt>
<dd>
<p>为生成式推荐引入跨用户协同记忆库，用可学习查询token压缩序列、目标感知检索与门控交叉注意力融合，Full-Ranking下NDCG@5最高提升23.7%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/graph_rec/how_do_graph_signals_affect_recommendation_unveiling_the_mystery_of_low_and/">How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals</a></strong></dt>
<dd>
<p>理论证明同绝对值低频/高频图信号在推荐中等价，提出频率信号缩放器与空间翻转插件，在四个数据集全库排序下 Recall@20 最高提升约 3.0%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/lower_bias_higher_welfare_how_creator_competition_reshapes_bias_variance_tr/">Lower Bias, Higher Welfare: How Creator Competition Reshapes Bias-Variance Tradeoff in Recommendation Platforms?</a></strong></dt>
<dd>
<p>博弈论建模创作者竞争如何重塑平台正则化选择，理论与实验一致表明策略环境下应弱正则（低偏差）以最大化用户福利。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/rethinking_popularity_bias_in_collaborative_filtering_via_analytical_vector/">Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition</a></strong></dt>
<dd>
<p>揭示 BPR 优化在嵌入空间刻出“流行度方向”的几何偏差，提出非对称方向分解校正 DDC，在 5 个骨干上 MRR@10 提升 6%–29% 且 AvgPop@10 降 21%–47%。</p>
</dd>
<dt><strong><a href="../../venues/KDD/2026/collaborative_filtering/scaling_multilingual_semantic_search_in_uber_eats_delivery/">Scaling Multilingual Semantic Search in Uber Eats Delivery</a></strong></dt>
<dd>
<p>基于 gte-Qwen2 双塔统一多语言多垂类语义召回，InfoNCE+Triplet NCE 两阶段训练与 MRL 多维度服务，Store R@20 相对 Legacy 提升 88%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/a_non_parametric_choice_model_that_learns_how_users_choose_between_recommen/">A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options</a></strong></dt>
<dd>
<p>用 sigmoid 核密度估计学习误差分布，无需假设 MNL/ENL 即可同时推断用户偏好与选择机制，抗竞争型曝光偏差。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/agentic_personalisation_of_cross_channel_marketing_experiences/">Agentic Personalisation of Cross-Channel Marketing Experiences</a></strong></dt>
<dd>
<p>用 DiD 因果估计 + Thompson 采样构建模块化营销智能体，在 640 万用户线上 RCT 中显著提升漏斗事件与 GMV。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/balancing_accuracy_and_novelty_with_sub_item_popularity/">Balancing Accuracy and Novelty with Sub-Item Popularity</a></strong></dt>
<dd>
<p>复用 RecJPQ 子 ID 分解在子码本粒度计算个性化流行度 sPPS，与 item 级 PPS 加权融合，在 NDCG 不降前提下显著提升个性化新颖度。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/correcting_the_logq_correction_revisiting_sampled_softmax_for_large_scale_r/">Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval</a></strong></dt>
<dd>
<p>指出标准 logQ 修正误把正样本当作采样项，提出带置信度权重的改进修正，在公开与工业数据上稳定超越标准 logQ。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/estimating_quantum_execution_requirements_for_feature_selection_in_recommen/">Estimating Quantum Execution Requirements for Feature Selection in Recommender Systems Using Extreme Value Theory</a></strong></dt>
<dd>
<p>用极值理论对 QAOA 求解 QUBO 特征选择的输出建模，估计达到经典 SA 水平所需的量子执行次数，并在模拟器与真实 QPU 上验证。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/hierarchical_graph_information_bottleneck_for_multi_behavior_recommendation/">Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用信息瓶颈原理重构多行为层次推荐，以 InfoNCE 保真、HSIC 压缩、可学习边丢弃去噪，全库排序下 HR@10 最高提升 15.08%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multimodal_rec/improving_visual_recommendation_on_e_commerce_platforms_using_vision_langua/">Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models</a></strong></dt>
<dd>
<p>在 Mercari C2C 平台微调 SigLIP 图像编码器做视觉相似召回，离线 nDCG@5 +9.1%，线上 CTR +50%、CVR +14%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/industry_insights_from_comparing_deep_learning_and_gbdt_models_for_e_commer/">Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank</a></strong></dt>
<dd>
<p>在 OTTO 电商 LTR 场景系统对比 DNN 与生产级 LambdaMART，Two-Tower+Softmax CE 离线 NDCG_c +4.32%，8 周线上点击 +1.86%、营收 +0.56%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/ip2_entity_guided_interest_probing_for_personalized_news_recommendation/">IP2: Entity-Guided Interest Probing for Personalized News Recommendation</a></strong></dt>
<dd>
<p>以实体-标题对比预训练探测新闻内实体焦点，并用跨注意力双塔用户编码器建模新闻间实体引导兴趣，MIND-large AUC 达 72.06%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/kamae_bridging_spark_and_keras_for_seamless_ml_preprocessing/">Kamae: Bridging Spark and Keras for Seamless ML Preprocessing</a></strong></dt>
<dd>
<p>将 PySpark 预处理管线一对一翻译为等价 Keras 层，消除离线/在线特征工程重复实现，Expedia 线上延迟降 61%、成本降 58%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/debias_fairness_industrial/leveraging_geometric_insights_in_hyperbolic_triplet_loss_for_improved_recom/">Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations</a></strong></dt>
<dd>
<p>洛伦兹双曲三元组损失 TriplH，以物品-物品项自适应间隔替代固定 margin，兼顾精度、去流行度偏差与 3 倍推理加速。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/location_matters_leveraging_multi_resolution_geo_embeddings_for_housing_sea/">Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search</a></strong></dt>
<dd>
<p>将 H3 多分辨率六边形网格与城市嵌入注入双塔对比学习，缓解房源地理稀疏，离线重排模拟租金转化提升 85%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/meta_off_policy_estimation/">Meta Off-Policy Estimation</a></strong></dt>
<dd>
<p>将多个 OPE 估计器视为相关研究做固定效应元分析，用 BLUE 闭式解融合并给出保守置信区间，OBD 上方差降超 50%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/on_the_reliability_of_sampling_strategies_in_offline_recommender_evaluation/">On the Reliability of Sampling Strategies in Offline Recommender Evaluation</a></strong></dt>
<dd>
<p>以全观测 KuaiRec 为真值，四维度（分辨率/保真度/鲁棒性/预测力）系统解耦曝光偏差与采样偏差，给出采样策略选择指南。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/orthogonal_low_rank_embedding_stabilization/">Orthogonal Low Rank Embedding Stabilization</a></strong></dt>
<dd>
<p>用低秩SVD+正交Procrustes把重训后的用户/物品嵌入对齐到固定参考空间，保内积无损，跨期余弦相似度从0.01升至0.82</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/multi_task_reranking/radar_recall_augmentation_through_deferred_asynchronous_retrieval/">RADAR: Recall Augmentation through Deferred Asynchronous Retrieval</a></strong></dt>
<dd>
<p>用离线异步全复杂度精排模型对 50 倍候选池预打分并缓存 Top-200，线上绕过召回与粗排直接注入精排，Recall@200 翻倍、线上 +0.8%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/recommendation_and_temptation/">Recommendation and Temptation</a></strong></dt>
<dd>
<p>将用户选择解耦为 enrichment 与 temptation 双效用并引入站外选项，证明局部贪心最优并给出可估计算子，MovieLens 与仿真全面超越基线。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/suggest_complement_inspire_story_of_two_tower_recommendations_at_allegro_co/">Suggest, Complement, Inspire: Story of Two Tower Recommendations at Allegro.com</a></strong></dt>
<dd>
<p>以内容特征双塔为统一底座，仅改查询塔与索引逻辑即复用至相似/互补/灵感三类召回，两年线上 A/B 验证 CTR 与 GMV 双增。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/the_future_is_sparse_embedding_compression_for_scalable_retrieval_in_recomm/">The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems</a></strong></dt>
<dd>
<p>用余弦重建的稀疏自编码器把稠密向量压成高维 k-稀疏向量，免重训骨干，12× 压缩仅损 1.35% CTR，线上显著优于同尺寸 Matryoshka。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/collaborative_filtering/the_hidden_cost_of_defaults_in_recommender_system_evaluation/">The Hidden Cost of Defaults in Recommender System Evaluation</a></strong></dt>
<dd>
<p>审计 RecBole 未文档化的 early_stop=10 默认策略如何静默截断随机搜索与贝叶斯优化，量化搜索路径不稳定性与可复现性风险。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2025/sequential_rec/user_long_term_multi_interest_retrieval_model_for_recommendation/">User Long-Term Multi-Interest Retrieval Model for Recommendation</a></strong></dt>
<dd>
<p>以类别感知分层双兴趣学习与指针生成式类别→物品级联检索，将召回阶段行为序列扩展至千级，淘宝离线HR@2000达16.55%，线上点击+5.54%、订单+11.01%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/generative_rec/egr_embedding_native_generative_retrieval_with_a_shared_llm/">EGR: Embedding-Native Generative Retrieval with a Shared LLM</a></strong></dt>
<dd>
<p>用单一共享 LLM 同时编码物品与用户历史，以双对比损失联合训练，免量化免 grounding，线上 CVR +2.91%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/multi_task_reranking/embedding_subspace_partitioning_for_dynamic_multi_objective_retrieval/">Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval</a></strong></dt>
<dd>
<p>将双塔嵌入切分为任务感知正交子空间，用服务时可调权重的加权内积替代单一点积，单模型即可在线遍历多目标 Pareto 前沿。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/mosaic_a_fleet_of_user_embedding_specialists_for_recommendation_at_meta/">Mosaic: A Fleet of User Embedding Specialists for Recommendation at Meta</a></strong></dt>
<dd>
<p>以四类异构专家模型族构建用户表征舰队，配合 MRM 与余弦冗余损失最大化增量信息，并提出免日志 CoEval 评估，离线 NE 稳定下降、线上 A/B 显著提升。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/multi_probe_zero_collision_hash_mpzch_mitigating_embedding_collisions_and_e/">Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders</a></strong></dt>
<dd>
<p>用线性探测+身份/元数据辅助张量与主动淘汰，在工业级嵌入表上实现零碰撞并提升新内容新鲜度，训练/推理延迟几乎不变。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/multilingual_semantic_retrieval_for_apple_music_search/">Multilingual Semantic Retrieval for Apple Music Search</a></strong></dt>
<dd>
<p>305M 多语双塔 + 课程式多目标微调，配合分位数校准混合召回，离线 Hit@10 相对提升 69%，线上尾部查询 CR 提升 7.93% 且无结果率降 86%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/parameter_efficient_single_collaborative_branch_for_recommendation/">Parameter-Efficient Single Collaborative Branch for Recommendation</a></strong></dt>
<dd>
<p>用单支路权重共享替代用户/物品双塔，参数量减半，精度持平且覆盖率与流行度偏差显著改善。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/personalizing_incremental_video_search_with_hybrid_text_and_id_embeddings/">Personalizing Incremental Video Search with Hybrid Text and ID Embeddings</a></strong></dt>
<dd>
<p>面向 Apple TV 逐键增量搜索，用文本语义与 ID 协同双嵌入注入 XGBoost 排序器，短前缀 NDCG@10 提升 8.63%，线上 CTR +1.14%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/recevolve_a_knowledge_driven_autonomous_agent_system_for_recommender_system/">RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems</a></strong></dt>
<dd>
<p>将双塔召回模型的架构搜索全生命周期交给多智能体闭环，41 次生产级训练换来 NDCG@50 相对 +19.9%、线上满意度 +3.77%，并意外暴露奖励黑客漏洞。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/smart_llm_augmented_hybrid_retrieval_for_dynamic_product_ads/">SMART: LLM-Augmented Hybrid Retrieval for Dynamic Product Ads</a></strong></dt>
<dd>
<p>揭示规则查询擅长 BM25 重定向、LLM 查询擅长 ANN 拓客的检索分解，用输出审计门控仅路由 10% 用户，线上转化率 +27.6%</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/llm_rec/spear_selection_aware_personalized_end_to_end_adaptive_rewriting_and_retrie/">SPEAR: Selection-aware Personalized End-to-end Adaptive Rewriting and Retrieval for Community Search</a></strong></dt>
<dd>
<p>以梯度隔离双嵌入+乘性门控+动态改写选择器统一查询改写与检索，离线点击召回@10提升99.5%，线上QVCTR+0.259%。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/training_seeds_and_model_selection_stability_in_recommender_system_evaluati/">Training seeds and model-selection stability in recommender-system evaluation</a></strong></dt>
<dd>
<p>固定数据划分与超参、仅变训练种子，从用户级指标敏感性、验证选模稳定性、验证-测试一致性与推荐列表一致性四层量化种子噪声，主张种子应纳入评估协议。</p>
</dd>
<dt><strong><a href="../../venues/RecSys/2026/collaborative_filtering/unipinrec_unifying_generative_retrieval_and_ranking_at_pinterest_scale/">UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale</a></strong></dt>
<dd>
<p>用 MAM 掩码动作建模 + 混合样本 + 跨阶段 KV 缓存复用，将生成式召回与排序全栈统一，线上延迟降 11.1%、QPS 升 63.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/a_learnable_fully_interacted_two_tower_model_for_pre_ranking_system/">A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System</a></strong></dt>
<dd>
<p>用可学习 Item Meta Matrix 实现显式早期交互、行/列 FC 相似度矩阵实现晚期交互，在保持双塔解耦推理效率下 AUC 相对提升 4.6%–14.3%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/a_worrying_reproducibility_study_of_intent_aware_recommendation_models/">A Worrying Reproducibility Study of Intent-Aware Recommendation Models</a></strong></dt>
<dd>
<p>复现 5 个意图感知推荐模型，发现 2 个不可复现、全部被调优后的传统非神经基线超越，揭示方法学顽疾。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/csmf_cascaded_selective_mask_fine_tuning_for_multi_objective_embedding_base/">CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval</a></strong></dt>
<dd>
<p>借鉴 PEFT 选择性掩码思想，按曝光→点击→转化级联顺序分阶段微调双塔召回，零参数增量实现多目标融合检索。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/graph_spectral_filtering_with_chebyshev_interpolation_for_recommendation/">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a></strong></dt>
<dd>
<p>用切比雪夫插值逼近非线性图滤波器，直接以原始交互为信号，突破 LightGCN 的截断线性低通瓶颈，Full-Ranking 下全面 SOTA 且推理快 8.1 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/ira_adaptive_interest_aware_representation_and_alignment_for_personalized_m/">IRA: Adaptive Interest-aware Representation and Alignment for Personalized Multi-interest Retrieval</a></strong></dt>
<dd>
<p>用可累积/剪枝的文本化 Interest Unit 表征多兴趣，语义对齐替代点击信号做 ANN 召回，离线 H@20 提升 20%，线上点击 +5.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/unbiased_collaborative_filtering_with_fair_sampling/">Unbiased Collaborative Filtering with Fair Sampling</a></strong></dt>
<dd>
<p>通过对称补充采样使每个用户/物品等概率成为正负样本，无需估计倾向分即可消除流行度偏置，全库排序下 Recall/NDCG 与 ARP 双优。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/why_is_normalization_necessary_for_linear_recommenders/">Why is Normalization Necessary for Linear Recommenders?</a></strong></dt>
<dd>
<p>理论剖析线性自编码器归一化对流行度/邻域偏差的作用，提出数据自适应归一化 DAN，六数据集全库排序下长尾最高提升 128.57%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong></dt>
<dd>
<p>以塔内自校正、跨塔EMA同步与级联模型共享三机制增强双塔召回，离线三数据集AUC一致提升，线上广告收入+8.36%且QPS仅降0.59%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong></dt>
<dd>
<p>以原型意图与分布意图重构表征，用对齐-均匀性+粗细粒度匹配联合优化，Amazon-book R@20 相对提升 10.6%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong></dt>
<dd>
<p>在 Item-ID 投影层上施加 Stiefel 流形正交约束，抑制稀疏缩放下的嵌入坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/promax_exploring_the_potential_of_llm_derived_profiles_with_distribution_sh/">ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems</a></strong></dt>
<dd>
<p>将 LLM 画像视为检索信号而非融合特征，用双路分布重塑（监督+自监督）引导基座模型，LightGCN 上 NDCG@10 最高提升 8.44%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/rethinking_semantic_collaborative_integration_why_alignment_is_not_enough/">Rethinking Semantic–Collaborative Integration: Why Alignment Is Not Enough</a></strong></dt>
<dd>
<p>论证语义-协同全局对齐假设过强，提出共享+私有潜结构与互补性诊断，最小融合在 Amazon 三域 Full-Ranking 上超越对齐范式最高 41.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom Index 与 Int8 融合 ANN 协同设计，吞吐提升 23.7×、成本效率提升 13.35×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong></dt>
<dd>
<p>以语义相关性为主监督、互动为相关项内偏好信号，融合交叉编码器分级标签、多通道检索先验与去偏互动构建统一双塔训练目标，离线 P@25 +10.5%、NDCG@25 +5.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong></dt>
<dd>
<p>零权重免训练新闻推荐框架，以时间衰减×表征相似度打分，离线全库排序超越 NRMS，线上 CTR 逼近 SOTA 且推理快 600 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR-AP/2025/debias_fairness_industrial/accurate_and_diverse_recommendations_via_propensity_weighted_linear_autoenc/">Accurate and Diverse Recommendations via Propensity-Weighted Linear Autoencoders</a></strong></dt>
<dd>
<p>用对数 Sigmoid 重定义 IPS 倾向分，避免幂律对热门物品过度惩罚，在 EASE 类线性自编码器上以闭式解实现覆盖率大幅提升且精度不降。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR-AP/2026/collaborative_filtering/lsnr_learned_sparse_news_recommendations_with_attentive_user_modeling/">LSNR: Learned Sparse News Recommendations with Attentive User Modeling</a></strong></dt>
<dd>
<p>用独立编码+注意力聚合替代 STAR 的历史拼接，构建双通路稀疏用户向量，MIND NDCG@10 较 STAR 提升 11%，短历史场景反超稠密模型。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/llm_rec/collm_integrating_collaborative_embeddings_into_large_language_models_for_r/">CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation</a></strong></dt>
<dd>
<p>将传统协同过滤模型学到的用户/物品表征经 MLP 映射进 LLM 词嵌入空间，以两阶段调优兼顾暖启动与冷启动，AUC 相对 TALLRec 提升约 8%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/collaborative_filtering/intent_propagation_contrastive_collaborative_filtering/">Intent Propagation Contrastive Collaborative Filtering</a></strong></dt>
<dd>
<p>双螺旋消息传播融合高阶关系与意图解耦，对比学习对齐结构/意图视图，Full-Ranking 下 NDCG 平均提升 9.36%</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2025/collaborative_filtering/latent_factor_modeling_with_expert_network_for_multi_behavior_recommendatio/">Latent Factor Modeling with Expert Network for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用门控专家网络显式建模共享潜在因子，配合对比学习保证因子独立与一致，在三个多行为数据集 Full-Ranking 下 HR@10 最高提升 3.95%。</p>
</dd>
<dt><strong><a href="../../venues/TKDE/2026/collaborative_filtering/memory_is_no_longer_a_bottleneck_memory_efficient_graph_filtering_for_scala/">Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering</a></strong></dt>
<dd>
<p>用逐用户 Krylov 子空间替代全量物品相似度图，实现免训练多项式图滤波，内存降 5.74×、推理加速 26× 且精度反超 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/debias_fairness_industrial/a_thorough_performance_benchmarking_on_lightweight_embedding_based_recommen/">A Thorough Performance Benchmarking on Lightweight Embedding-based Recommender Systems</a></strong></dt>
<dd>
<p>统一协议横评 7 类轻量嵌入压缩方法，提出 MagPrune 强基线，并首次量化跨任务迁移性与树莓派端侧效率瓶颈。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/llm_rec/enhancing_id_based_recommendation_with_large_language_models/">Enhancing ID-based Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>用 LoRA 微调 LLM 仅凭纯 ID 序列生成伪交互数据，经双重过滤后注入任意 ID 基推荐模型，全库排序下平均提升 3%–9%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/collaborative_filtering/lightweight_embeddings_with_graph_rewiring_for_collaborative_filtering/">Lightweight Embeddings with Graph Rewiring for Collaborative Filtering</a></strong></dt>
<dd>
<p>量化组合码本+图重连剪枝，在严格存储预算下兼顾 GNN 推荐精度与边缘端推理效率，iFashion 存储降 97.7%。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/graph_rec/position_aware_graph_transformer_for_recommendation/">Position-aware Graph Transformer for Recommendation</a></strong></dt>
<dd>
<p>用四种为交互图定制的节点位置编码把 Transformer 全局建模能力注入 GCN 推荐骨干，在四个数据集上以 Full-Ranking 协议取得最高 27.69% 的 Recall@20 相对提升。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2025/debias_fairness_industrial/post_training_attribute_unlearning_in_recommender_systems/">Post-Training Attribute Unlearning in Recommender Systems</a></strong></dt>
<dd>
<p>提出锚点分布MMD与函数空间排序正则的PoT-AU双损失，多类属性遗忘复杂度降至O(T)且推荐性能近无损。</p>
</dd>
<dt><strong><a href="../../venues/TOIS/2026/debias_fairness_industrial/mitigating_popularity_bias_in_recommendation_with_global_listwise_learning_/">Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting</a></strong></dt>
<dd>
<p>用 IPS 加权多项似然实现列表级全局无偏学习，并以 Bi-Weighting 平滑倾向估计与渐进式去偏，六数据集 Recall@20 平均提升 9.43%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/collaborative_filtering/balancing_embedding_spectrum_for_recommendation/">Balancing Embedding Spectrum for Recommendation</a></strong></dt>
<dd>
<p>从谱分析视角揭示成对学习中的嵌入坍缩，提出 DirectSpec 全通滤波直接均衡奇异值谱，MF/LightGCN 上 nDCG@10 最高提升 52.6%/42.4%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/collaborative_filtering/enhancing_item_level_bundle_representation_for_bundle_recommendation/">Enhancing Item-level Bundle Representation for Bundle Recommendation</a></strong></dt>
<dd>
<p>用 B-U-I 高阶关联与预训练伪交互增强捆绑包物品级表征，全库排序下平均提升 7.5%。</p>
</dd>
<dt><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/evaluating_impact_of_user_cluster_targeted_attacks_in_matrix_factorisation_/">Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders</a></strong></dt>
<dd>
<p>系统分析 MF 推荐中面向用户簇的投毒攻击，揭示 V 矩阵是攻击传播主通道，簇内真实评分量决定脆弱性。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2025/collaborative_filtering/s_diff_an_anisotropic_diffusion_model_for_collaborative_filtering_in_spectr/">S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain</a></strong></dt>
<dd>
<p>将扩散前向过程搬到物品图拉普拉斯谱域，用特征值参数化各向异性噪声保低频，配合 FiLM 条件去噪，全库排序下 Recall@20 最高提升约 5.5%。</p>
</dd>
<dt><strong><a href="../../venues/WSDM/2026/collaborative_filtering/have_we_really_understood_collaborative_information_an_empirical_investigat/">Have We Really Understood Collaborative Information? An Empirical Investigation</a></strong></dt>
<dd>
<p>以物品共现路径长度量化协作信息为0/1/2-hop协作关系，系统实证其在六个基准上的分布、对12种推荐算法的影响与预测依赖，揭示模型强于简单CR、弱于复杂CR的核心规律。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/agent_conversational_rec/bridging_conversational_and_collaborative_signals_for_conversational_recomm/">Bridging Conversational and Collaborative Signals for Conversational Recommendation</a></strong></dt>
<dd>
<p>构建 Reddit-ML32M 跨源数据集，用 SASRec 协同嵌入对 GPT-3.5 零样本推荐做相似度重排，H@5 提升 12.32%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/generative_rec/collaborative_diffusion_model_for_recommender_system/">Collaborative Diffusion Model for Recommender System</a></strong></dt>
<dd>
<p>将评论词转为伪用户，在扩散去噪过程中聚合真实/伪邻居偏好信息，缓解个性化信息流失，三数据集 R@20 全面领先。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/criteria_aware_graph_filtering_extremely_fast_yet_accurate_multi_criteria_r/">Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation</a></strong></dt>
<dd>
<p>免训练免矩阵分解的多准则图滤波推荐，按准则定制多项式低通滤波器并融合用户准则偏好，BA 数据集 0.2 秒推理、NDCG@5 最高提升 24%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/does_weighting_improve_matrix_factorization_for_recommender_systems/">Does Weighting Improve Matrix Factorization for Recommender Systems?</a></strong></dt>
<dd>
<p>系统实证加权隐式反馈矩阵分解：大秩模型下不加权常持平或更优，并给出加权目标闭式解与预条件共轭梯度高效求解。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/frequency_aware_graph_signal_processing_for_collaborative_filtering/">Frequency-aware Graph Signal Processing for Collaborative Filtering</a></strong></dt>
<dd>
<p>用级联高通-低通滤波捕获独有/共性特征、并行低通滤波捕获高阶邻域，免训练线性组合在六个数据集上 Full-Ranking 提升 2-5%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/graph_neural_controlled_differential_equations_for_collaborative_filtering/">Graph Neural Controlled Differential Equations For Collaborative Filtering</a></strong></dt>
<dd>
<p>用 MLP 控制器生成连续时变权重矩阵替代 GODE-CF 的离散权重，在四个 Amazon 5-core 数据集 Full-Ranking 下 Recall@20 全面超越 LightGCN/GODE-CF。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/graphhash_graph_clustering_enables_parameter_efficiency_in_recommender_syst/">GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems</a></strong></dt>
<dd>
<p>用模块度二部图聚类替代 ID 哈希，将图平滑前置到预处理阶段，嵌入表缩减 75%+ 时 Recall@20 平均提升 101.52%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/interactrank_personalized_web_scale_search_pre_ranking_with_cross_interacti/">InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features</a></strong></dt>
<dd>
<p>在双塔预排序点积之上叠加历史交互统计特征 IQP 与实时用户序列，以 1.1× FLOPs 实现线上 SIFR +6.5%、F1S +9.2%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/multimodal_rec/multimodal_semantic_retrieval_for_product_search/">Multimodal semantic retrieval for product search</a></strong></dt>
<dd>
<p>在电商产品检索中系统对比 4-tower/3-tower 图文融合双塔，用 FAISS 全库 Cosine 检索验证多模态表征在购买召回与相关性精度上的增益。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/self_supervised_denoising/personalized_denoising_implicit_feedback_for_robust_recommender_system/">Personalized Denoising Implicit Feedback for Robust Recommender System</a></strong></dt>
<dd>
<p>揭示全局损失分布中正常/噪声交互严重重叠，提出基于用户个人损失分布的重采样去噪 PLD，BPR 下 Recall@20 最高提升 7.36%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/collaborative_filtering/rankformer_a_graph_transformer_for_recommendation_based_on_ranking_objectiv/">Rankformer: A Graph Transformer for Recommendation based on Ranking Objective</a></strong></dt>
<dd>
<p>从 BPR 梯度下降推导出图 Transformer 层，正负样本分离聚合+基准项+偏移项，线性加速后 4 数据集平均提升 4.48%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/sequential_rec/retrieval_with_learned_similarities/">Retrieval with Learned Similarities</a></strong></dt>
<dd>
<p>证明 MoL 为通用相似度逼近器，配互信息负载均衡损失与带误差界的近似 top-K 检索，序列推荐 HR@1 平均提升 29.1%，检索延迟最高降 66×。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2025/llm_rec/training_large_recommendation_models_via_graph_language_token_alignment/">Training Large Recommendation Models via Graph-Language Token Alignment</a></strong></dt>
<dd>
<p>将 LightGCN 用户/物品节点经投影器对齐进 LLaMA-2 词表空间，用 GLLM logits 匹配层做端到端物品预测，Full-Ranking 下 NDCG@10 最高提升 185%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/diversity_augmented_negative_sampling_for_implicit_collaborative_filtering/">Diversity-Augmented Negative Sampling for Implicit Collaborative Filtering</a></strong></dt>
<dd>
<p>用用户级缓存 + 多样性增强 k-DPP + mixup 合成负样本，解决两阶段负采样同质化问题，Full-Ranking 下 NDCG@20 最高提升 8.76%。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/efficient_learning_of_sparse_representations_from_interactions/">Efficient Learning of Sparse Representations from Interactions</a></strong></dt>
<dd>
<p>在 ELSA 线性自编码器训练中渐进剪枝学习高维稀疏物品嵌入，实现最高 100× 压缩仅损 2.5% nDCG，并派生可解释语义分段。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/grank_towards_target_aware_and_streamlined_industrial_retrieval_with_a_gene/">GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework</a></strong></dt>
<dd>
<p>用训练期目标注入把 Target-Aware 能力蒸馏进 MIPS 兼容生成器，再以轻量交叉注意力重排，免结构化索引实现 Recall@500 +30% 与 1.7× QPS。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/identifying_and_upweighting_power_niche_users_to_mitigate_popularity_bias_i/">Identifying and Upweighting Power-Niche Users to Mitigate Popularity Bias in Recommendations</a></strong></dt>
<dd>
<p>识别高活跃-小众偏好用户并对其在BPR损失中双重加权，在三个基准上实现性能与流行度去偏的帕累托双赢。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/pi2i_a_personalized_item_based_collaborative_filtering_retrieval_framework/">PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework</a></strong></dt>
<dd>
<p>放宽 I2I 截断保召回、Trigger-Target 负采样对齐离线在线、目标注意力精排打分，淘宝线上成交率 +1.05%</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/spattack_subgroup_poisoning_attacks_on_federated_recommender_systems/">Spattack: Subgroup Poisoning Attacks on Federated Recommender Systems</a></strong></dt>
<dd>
<p>首个联邦推荐子群投毒攻击：近似-提升范式+对比排斥与自适应加权，0.1%恶意客户端即达100% ER@5且几乎不伤非目标群。</p>
</dd>
<dt><strong><a href="../../venues/WWW/2026/collaborative_filtering/talos_optimizing_top_k_accuracy_in_recommender_systems/">Talos: Optimizing Top-$K$ Accuracy in Recommender Systems</a></strong></dt>
<dd>
<p>用量化阈值把 Top-K 截断改写为分数-阈值比较，配合采样分位回归与约束项，直接优化 Precision@K/Recall@K 并具备 DRO 鲁棒性。</p>
</dd>
</dl>

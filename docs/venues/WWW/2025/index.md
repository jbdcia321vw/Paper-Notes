---
hide:
  - toc
---

# 🏛️ WWW (TheWebConf) 2025 推荐系统论文解读

88 篇 WWW (TheWebConf) 2025 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (7)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/criteria_aware_graph_filtering_extremely_fast_yet_accurate_multi_criteria_r/">Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation</a></strong></dt>
<dd>
<p>免训练免矩阵分解的多准则图滤波推荐，按准则定制多项式低通滤波器并融合用户准则偏好，BA 数据集 0.2 秒推理、NDCG@5 最高提升 24%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/does_weighting_improve_matrix_factorization_for_recommender_systems/">Does Weighting Improve Matrix Factorization for Recommender Systems?</a></strong></dt>
<dd>
<p>系统实证加权隐式反馈矩阵分解：大秩模型下不加权常持平或更优，并给出加权目标闭式解与预条件共轭梯度高效求解。</p>
</dd>
<dt><strong><a href="collaborative_filtering/frequency_aware_graph_signal_processing_for_collaborative_filtering/">Frequency-aware Graph Signal Processing for Collaborative Filtering</a></strong></dt>
<dd>
<p>用级联高通-低通滤波捕获独有/共性特征、并行低通滤波捕获高阶邻域，免训练线性组合在六个数据集上 Full-Ranking 提升 2-5%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/graph_neural_controlled_differential_equations_for_collaborative_filtering/">Graph Neural Controlled Differential Equations For Collaborative Filtering</a></strong></dt>
<dd>
<p>用 MLP 控制器生成连续时变权重矩阵替代 GODE-CF 的离散权重，在四个 Amazon 5-core 数据集 Full-Ranking 下 Recall@20 全面超越 LightGCN/GODE-CF。</p>
</dd>
<dt><strong><a href="collaborative_filtering/graphhash_graph_clustering_enables_parameter_efficiency_in_recommender_syst/">GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems</a></strong></dt>
<dd>
<p>用模块度二部图聚类替代 ID 哈希，将图平滑前置到预处理阶段，嵌入表缩减 75%+ 时 Recall@20 平均提升 101.52%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/interactrank_personalized_web_scale_search_pre_ranking_with_cross_interacti/">InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features</a></strong></dt>
<dd>
<p>在双塔预排序点积之上叠加历史交互统计特征 IQP 与实时用户序列，以 1.1× FLOPs 实现线上 SIFR +6.5%、F1S +9.2%。</p>
</dd>
<dt><strong><a href="collaborative_filtering/rankformer_a_graph_transformer_for_recommendation_based_on_ranking_objectiv/">Rankformer: A Graph Transformer for Recommendation based on Ranking Objective</a></strong></dt>
<dd>
<p>从 BPR 梯度下降推导出图 Transformer 层，正负样本分离聚合+基准项+偏移项，线性加速后 4 数据集平均提升 4.48%</p>
</dd>
</dl>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl>
<dt><strong><a href="graph_rec/graph_sequential_alignment_and_uniformity_toward_enhanced_recommendation_sy/">Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems</a></strong></dt>
<dd>
<p>以共享嵌入层联合训练 LightGCN 与 SASRec，用对齐-均匀性损失实现跨子模块知识迁移，Leave-One-Out 全库排序下 R@10 最高提升 13.4%。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/mixrec_individual_and_collective_mixing_empowers_data_augmentation_for_reco/">MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems</a></strong></dt>
<dd>
<p>用 Beta/Dirichlet 混合在嵌入空间零成本造正负视图，配合双混合对比损失，全库排序下 Recall@20 最高提升 5.05%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/personalized_denoising_implicit_feedback_for_robust_recommender_system/">Personalized Denoising Implicit Feedback for Robust Recommender System</a></strong></dt>
<dd>
<p>揭示全局损失分布中正常/噪声交互严重重叠，提出基于用户个人损失分布的重采样去噪 PLD，BPR 下 Recall@20 最高提升 7.36%。</p>
</dd>
<dt><strong><a href="self_supervised_denoising/unleashing_the_power_of_large_language_model_for_denoising_recommendation/">Unleashing the Power of Large Language Model for Denoising Recommendation</a></strong></dt>
<dd>
<p>LLaRD 用 LLM 生成偏好知识与图 CoT 关系知识，再以信息瓶颈对齐去噪图，Full-Ranking 下 Recall@20 最高提升约 14%。</p>
</dd>
</dl>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (7)</h2>
<dl>
<dt><strong><a href="sequential_rec/a_contextual_aware_position_encoding_for_sequential_recommendation/">A Contextual-Aware Position Encoding for Sequential Recommendation</a></strong></dt>
<dd>
<p>首个为序列推荐定制的上下文感知位置编码 CAPE：用目标-上下文不相似度累加动态分配位置，配合门控融合异构特征，在 5 个骨干与工业 A/B 上稳定涨点。</p>
</dd>
<dt><strong><a href="sequential_rec/efficient_user_history_modeling_with_amortized_inference_for_deep_learning_/">Efficient user history modeling with amortized inference for deep learning recommendation models</a></strong></dt>
<dd>
<p>用 cross-attention 追加候选替代逐项拼接，使历史编码可摊销推理，LinkedIn Feed/Ads 延迟降 30%</p>
</dd>
<dt><strong><a href="sequential_rec/flare_fusing_language_models_and_collaborative_architectures_for_recommende/">FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancement</a></strong></dt>
<dd>
<p>用冻结 mT5+Perceiver 将文本注入 Bert4Rec，并以重调 Bert4Rec 基线、大词表评测与 Critiquing 任务重估 ID+文本增益。</p>
</dd>
<dt><strong><a href="sequential_rec/large_memory_network_for_recommendation/">Large Memory Network for Recommendation</a></strong></dt>
<dd>
<p>用百万级全局共享记忆块压缩用户历史行为，以乘积量化双轴激活实现 O(√n d) 检索，抖音电商搜索 AUC +4.68%、订单/人 +0.87%。</p>
</dd>
<dt><strong><a href="sequential_rec/long_term_interest_clock_fine_grained_time_perception_in_streaming_recommen/">Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System</a></strong></dt>
<dd>
<p>用候选物品+当前时刻双查询从长期行为中检索子序列，并以时距感知注意力聚合，实现流式推荐细粒度时间感知，线上活跃天数+0.122%。</p>
</dd>
<dt><strong><a href="sequential_rec/retrieval_with_learned_similarities/">Retrieval with Learned Similarities</a></strong></dt>
<dd>
<p>证明 MoL 为通用相似度逼近器，配互信息负载均衡损失与带误差界的近似 top-K 检索，序列推荐 HR@1 平均提升 29.1%，检索延迟最高降 66×。</p>
</dd>
<dt><strong><a href="sequential_rec/td3_tucker_decomposition_based_dataset_distillation_method_for_sequential_r/">TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation</a></strong></dt>
<dd>
<p>用 Tucker 分解把合成序列摘要解耦为用户/时间/共享物品因子与关系核，配合特征空间对齐双层优化，在 0.2%–7.4% 数据量下逼近甚至超越全量训练。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (4)</h2>
<dl>
<dt><strong><a href="generative_rec/collaborative_diffusion_model_for_recommender_system/">Collaborative Diffusion Model for Recommender System</a></strong></dt>
<dd>
<p>将评论词转为伪用户，在扩散去噪过程中聚合真实/伪邻居偏好信息，缓解个性化信息流失，三数据集 R@20 全面领先。</p>
</dd>
<dt><strong><a href="generative_rec/distinguished_quantized_guidance_for_diffusion_based_sequence_recommendatio/">Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation</a></strong></dt>
<dd>
<p>用语义码本量化序列增强扩散引导，并以对比差异最大化抑制流行度偏置生成，四数据集全库排序全面领先。</p>
</dd>
<dt><strong><a href="generative_rec/generative_large_recommendation_models_emerging_trends_in_llms_for_recommen/">Generative Large Recommendation Models: Emerging Trends in LLMs for Recommendation</a></strong></dt>
<dd>
<p>系统梳理生成式大推荐模型范式：数据质量、缩放律、用户行为挖掘与训练推理效率四大议题及未来方向。</p>
</dd>
<dt><strong><a href="generative_rec/hyperbolic_diffusion_recommender_model/">Hyperbolic Diffusion Recommender Model</a></strong></dt>
<dd>
<p>将扩散推荐从欧氏空间迁移至双曲流形，用聚类锚定的方向性噪声替代各向同性高斯噪声，Full-Ranking 下 R@10 最高提升 5.6%。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (19)</h2>
<dl>
<dt><strong><a href="llm_rec/beyond_retrieval_generating_narratives_in_conversational_recommender_system/">Beyond Retrieval: Generating Narratives in Conversational Recommender Systems</a></strong></dt>
<dd>
<p>构建 REGEN 叙事数据集并提出 CF+内容嵌入经 Adapter 注入 LLM 的融合架构，在 Office/Clothing 两域上 BLEU/ROUGE/语义相似度较单模态提升 4–12%。</p>
</dd>
<dt><strong><a href="llm_rec/beyond_utility_evaluating_llm_as_recommender/">Beyond Utility: Evaluating LLM as Recommender</a></strong></dt>
<dd>
<p>提出含历史长度敏感性、候选位置偏差、画像生成与幻觉四维的 LLM 推荐多维评测框架，揭示 LLM 在重排与冷启动占优但位置偏差严重。</p>
</dd>
<dt><strong><a href="llm_rec/d2k_turning_historical_data_into_retrievable_knowledge_for_recommender_syst/">D2K: Turning Historical Data into Retrievable Knowledge for Recommender Systems</a></strong></dt>
<dd>
<p>将历史日志转为用户-物品-上下文三元知识库，Transformer 编码 + 个性化适配注入任意 CTR 模型，AUC 相对提升 1.35%–2.90%。</p>
</dd>
<dt><strong><a href="llm_rec/eager_llm_enhancing_large_language_models_as_recommenders_through_exogenous/">EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration</a></strong></dt>
<dd>
<p>以双源知识富集物品索引+非侵入多尺度对齐重建+退火适配器，将外源行为/语义信号注入 LLM 生成式推荐，Amazon 三数据集 Full-Ranking 下 Recall@10 最高提升 26.16%。</p>
</dd>
<dt><strong><a href="llm_rec/enhancing_news_recommendation_with_hierarchical_llm_prompting/">Enhancing News Recommendation with Hierarchical LLM Prompting</a></strong></dt>
<dd>
<p>用分层提示让 LLM 生成更丰富的新闻标题与实体，作为模型无关的数据增强模块，在 MIND 上以 AUC 68.88 超越 GLoCIM 等 SOTA。</p>
</dd>
<dt><strong><a href="llm_rec/external_large_foundation_model_how_to_efficiently_serve_trillions_of_param/">External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation</a></strong></dt>
<dd>
<p>以外部蒸馏+数据增强服务解耦教师训练，用辅助头与学生适配器弥合流式分布漂移，实现万亿参数基础模型零延迟增益服务多业务精排模型。</p>
</dd>
<dt><strong><a href="llm_rec/filtering_discomforting_recommendations_with_large_language_models/">Filtering Discomforting Recommendations with Large Language Models</a></strong></dt>
<dd>
<p>用可编辑偏好画像+对话式规则生成，以插件方式过滤主观不适推荐；离线代理任务与24人一周用户研究验证有效性。</p>
</dd>
<dt><strong><a href="llm_rec/g_refer_graph_retrieval_augmented_large_language_model_for_explainable_reco/">G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation</a></strong></dt>
<dd>
<p>用路径级+节点级混合图检索把协同过滤信号显式翻译为文本，经知识剪枝与 RAFT 微调驱动 LLM 生成可解释推荐，BERT-F1 最高提升 2.81%。</p>
</dd>
<dt><strong><a href="llm_rec/large_language_models_as_narrative_driven_recommenders/">Large Language Models as Narrative-Driven Recommenders</a></strong></dt>
<dd>
<p>系统评测 38 个 LLM 在 Reddit 叙事式电影推荐上的表现，发现零样本提示即可显著超越 doc2vec 基线 70%+，中型开源模型接近大型闭源模型。</p>
</dd>
<dt><strong><a href="llm_rec/legommenders_a_comprehensive_content_based_recommendation_library_with_llm_/">Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support</a></strong></dt>
<dd>
<p>首个支持内容编码器与行为/交互模块端到端联合训练的推荐库，内置 LLM 编码与数据增强，覆盖 1000+ 模型组合与 15 个数据集，推理缓存带来最高 50x 加速。</p>
</dd>
</dl>
<p><a href="llm_rec/">查看全部19篇「大语言模型推荐 (LLM4Rec)」论文 →</a></p>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (6)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/a_llm_based_controllable_scalable_human_involved_user_simulator_framework_f/">A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>以插件管理器分阶段管控 LLM 用户模拟器，通过记忆隔离与偏好二分抑制数据泄漏，在 ReDial/OpenDialKG/MovieLens 上显著提升 CRS 评估可信度。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/agentsociety_challenge_designing_llm_agents_for_user_modeling_and_recommend/">AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms</a></strong></dt>
<dd>
<p>首个 WWW LLM Agent 双赛道竞赛：Yelp/Amazon/Goodreads 三源数据 + 交互式模拟器，295 队参赛，Track1/2 最终阶段提升 9.1%/15.9%，并验证模拟 groundtruth 与真实分布强相关。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/bridging_conversational_and_collaborative_signals_for_conversational_recomm/">Bridging Conversational and Collaborative Signals for Conversational Recommendation</a></strong></dt>
<dd>
<p>构建 Reddit-ML32M 跨源数据集，用 SASRec 协同嵌入对 GPT-3.5 零样本推荐做相似度重排，H@5 提升 12.32%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/collaborative_retrieval_for_large_language_model_based_conversational_recom/">Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems</a></strong></dt>
<dd>
<p>用 EASE 协同检索 + 两步 LLM 反思重排为黑盒 LLM 注入 CF 知识，Reddit-v2/Redial 全库排序 Recall@20 显著超越 Zero-shot LLM。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/disentangling_likes_and_dislikes_in_personalized_generative_explainable_rec/">Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation</a></strong></dt>
<dd>
<p>用 LLM 重构正负观点分离的可解释推荐数据集，提出情感匹配与正负内容相似度指标，并证明将预测评分作为输入可显著提升情感感知解释生成。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/recusersim_a_realistic_and_diverse_user_simulator_for_evaluating_conversati/">RecUserSim: A Realistic and Diverse User Simulator for Evaluating Conversational Recommender Systems</a></strong></dt>
<dd>
<p>提出四模块 LLM Agent 用户模拟器，以有限理性三层 Rating-Action-Response 机制实现真实角色扮演、多样人群覆盖与显式打分，并落地华为 Celia 餐饮助手。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (7)</h2>
<dl>
<dt><strong><a href="multimodal_rec/comment_staytime_prediction_with_llm_enhanced_comment_understanding/">Comment Staytime Prediction with LLM-enhanced Comment Understanding</a></strong></dt>
<dd>
<p>首个评论区停留时长数据集 KuaiComt 与两阶段 LCU 框架：LLM 领域微调 + 评论排序辅助任务，离线全指标提升且线上停留时长 +1.27%。</p>
</dd>
<dt><strong><a href="multimodal_rec/generating_with_fairness_a_modality_diffused_counterfactual_framework_for_i/">Generating with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete Multimodal Recommendations</a></strong></dt>
<dd>
<p>用模态感知扩散补全缺失模态，并以反事实推断剥离可见性偏差，在三个数据集上同时提升准确率与公平性。</p>
</dd>
<dt><strong><a href="multimodal_rec/hcmrm_a_high_consistency_multimodal_relevance_model_for_search_ads/">HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads</a></strong></dt>
<dd>
<p>用视频关键词切分合成伪查询做三元组预训练，并以对称层次 Softmax 对齐四档相关性标签，快手搜索广告线上无关广告率降 6.1%、收入增 1.4%。</p>
</dd>
<dt><strong><a href="multimodal_rec/hierarchical_time_aware_mixture_of_experts_for_multi_modal_sequential_recom/">Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation</a></strong></dt>
<dd>
<p>用交互MoE过滤模态冗余、时序MoE注入显式时间戳，配合三项辅助任务，在四个Amazon数据集全库排序上NDCG@5最高提升20.9%。</p>
</dd>
<dt><strong><a href="multimodal_rec/multimodal_semantic_retrieval_for_product_search/">Multimodal semantic retrieval for product search</a></strong></dt>
<dd>
<p>在电商产品检索中系统对比 4-tower/3-tower 图文融合双塔，用 FAISS 全库 Cosine 检索验证多模态表征在购买召回与相关性精度上的增益。</p>
</dd>
<dt><strong><a href="multimodal_rec/personalized_image_generation_with_large_multimodal_models/">Personalized Image Generation with Large Multimodal Models</a></strong></dt>
<dd>
<p>Pigeon 用 LMM+掩码生成器从噪声历史图中抽取视觉偏好，两阶段偏好对齐（掩码重建+DPO）实现贴纸/海报个性化生成，个性化指标提升 7%~31%。</p>
</dd>
<dt><strong><a href="multimodal_rec/persrv_personalized_sticker_retrieval_with_vision_language_model/">PerSRV: Personalized Sticker Retrieval with Vision-Language Model</a></strong></dt>
<dd>
<p>用 SFT-LLaVA 生成贴纸语义、三路众包效用分与 CLIP 风格聚类，实现召回+个性化排序的贴纸检索，WeChat 数据 M-MRR@20 提升 8.95%。</p>
</dd>
</dl>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl>
<dt><strong><a href="social_group_rec/leveraging_member_group_relations_via_multi_view_graph_filtering_for_effect/">Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation</a></strong></dt>
<dd>
<p>提出免训练多视图图滤波 Group-GF，用三张物品相似图与多项式滤波器聚合成员-群组关系，HR@10 最高提升 16.4% 且运行时间仅 1.55s。</p>
</dd>
<dt><strong><a href="social_group_rec/virtual_stars_real_fans_understanding_the_vtuber_ecosystem/">Virtual Stars, Real Fans: Understanding the VTuber Ecosystem</a></strong></dt>
<dd>
<p>基于Bilibili 270万场直播的VTuber核心粉丝（会员）行为测量研究，逆向构建FanRanker将潜在付费会员推荐给主播，Top-50命中85%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl>
<dt><strong><a href="cold_start_rec/crown_a_novel_approach_to_comprehending_users_preferences_for_accurate_pers/">CROWN: A Novel Approach to Comprehending Users&#x27; Preferences for Accurate Personalized News Recommendation</a></strong></dt>
<dd>
<p>以类别引导意图解耦、标题-内容一致性加权与 GNN 混合用户表征，统一解决新闻推荐的多意图、读后偏好差异与冷启动三大难题。</p>
</dd>
<dt><strong><a href="cold_start_rec/personalized_diffusion_model_reshapes_cold_start_bundle_recommendation/">Personalized Diffusion Model Reshapes Cold-Start Bundle Recommendation</a></strong></dt>
<dd>
<p>用个性化条件扩散在分布空间生成捆绑包，配合解耦自注意力与KL正则，冷启动捆绑推荐全库排序Recall@20最高提升1155%。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (5)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/abxi_invariant_interest_adaptation_for_task_guided_cross_domain_sequential_/">ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用任务引导对齐消除预测错配，并以 dLoRA/iLoRA 双 LoRA 抽取域不变兴趣，三数据集 MRR 大幅超越 CDSR 基线。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/image_fusion_for_cross_domain_sequential_recommendation/">Image Fusion for Cross-Domain Sequential Recommendation</a></strong></dt>
<dd>
<p>用冻结 CLIP 图像嵌入补足 ID 表征，并以多注意力分别建模域内/跨域序列，在 Amazon 双域场景 MRR 提升约 5%–18%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/pre_train_and_fine_tune_recommenders_as_large_models/">Pre-train and Fine-tune: Recommenders as Large Models</a></strong></dt>
<dd>
<p>以信息瓶颈理论解释推荐模型微调，提出即插即用的信息感知自适应核 IAK，冻结十亿参数预训练推荐大模型，仅用一周域内数据微调即获 CTR-AUC 平均 +0.26%~+1.28% 与线上 GMV +0.74%。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/separated_contrastive_learning_for_matching_in_cross_domain_recommendation_/">Separated Contrastive Learning for Matching in Cross-domain Recommendation with Curriculum Scheduling</a></strong></dt>
<dd>
<p>将 intra-CL 与 inter-CL 解耦为两阶段课程训练，配合 stop-gradient 与基于 Katz 中心性的负样本难度调度，稳定跨域召回训练并显著提升 HIT@100。</p>
</dd>
<dt><strong><a href="cross_domain_multi_scenario/uncovering_cross_domain_recommendation_ability_of_large_language_models/">Uncovering Cross-Domain Recommendation Ability of Large Language Models</a></strong></dt>
<dd>
<p>用源域购买序列+跨域引导提示驱动 LLM 零样本重排目标域候选，揭示域间隙与模型规模对跨域迁移的决定性影响。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (7)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/bat_benchmark_for_auto_bidding_task/">BAT: Benchmark for Auto-bidding Task</a></strong></dt>
<dd>
<p>发布 Avito 双拍卖格式（VCG/FP）真实自动出价数据集，配套 ALM/TA-PID 等 5 个基线，覆盖预算平滑与 CPC 约束两类 RTB 任务。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/ctr_driven_advertising_image_generation_with_multimodal_large_language_mode/">CTR-Driven Advertising Image Generation with Multimodal Large Language Models</a></strong></dt>
<dd>
<p>用 MLLM 双分支奖励模型把 CTR 相对比较转化为偏好信号，结合产品中心偏好优化 PCPO 驱动广告图生成，线上 CTR 提升 2%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/efficient_and_practical_approximation_algorithms_for_advertising_in_content/">Efficient and Practical Approximation Algorithms for Advertising in Content Feeds</a></strong></dt>
<dd>
<p>针对信息流原生广告位分配，提出反向贪心 2-近似算法，突破前人 4-近似瓶颈，并在合成与真实数据上首次系统评测。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/middleman_bias_in_advertising_aligning_relevance_of_keyphrase_recommendatio/">Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search</a></strong></dt>
<dd>
<p>揭示广告关键词相关性过滤中的“中间人偏差”，用Search相关性判定重构训练标签，并以轻量Cross-Encoder在eBay线上取得CVR+11.6%的收益。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/on_the_practice_of_deep_hierarchical_ensemble_network_for_ad_conversion_rat/">On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction</a></strong></dt>
<dd>
<p>Pinterest 工业实践：两层 DHEN 多任务骨干 + 五类行为序列端到端建模 + 未来行为自监督辅助损失，线上 CPA 降 2.15%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/prectr_a_synergistic_framework_for_integrating_personalized_search_relevanc/">PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction</a></strong></dt>
<dd>
<p>用条件概率融合把相关性分级与CTR统一进单模型，配两阶段训练、语义一致性正则与个性化相关性激励，线上CTR+0.4%、GMV+1.1%。</p>
</dd>
<dt><strong><a href="ctr_feature_interaction/rtbagent_a_llm_based_agent_system_for_real_time_bidding/">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></strong></dt>
<dd>
<p>首个 LLM Agent 出价系统，用两步决策+多类记忆+专家策略工具在 iPinYou 上以调整因子方式超越规则/RL/扩散基线。</p>
</dd>
</dl>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (5)</h2>
<dl>
<dt><strong><a href="multi_task_reranking/a_hybrid_cross_stage_coordination_pre_ranking_model_for_online_recommendati/">A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems</a></strong></dt>
<dd>
<p>提出 HCCP 粗排模型，融合上游召回与下游精排多级未曝光样本，用 Margin InfoNCE 联合优化一致性与长尾精度，JD 线上 UCVR+14.9%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/llm4rerank_llm_based_auto_reranking_framework_for_recommendations/">LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations</a></strong></dt>
<dd>
<p>将准确性/多样性/公平性抽象为全连接图节点，用零样本 LLM 以 CoT 多跳自动重排，三数据集上多准则均衡超越 SOTA。</p>
</dd>
<dt><strong><a href="multi_task_reranking/nlgr_utilizing_neighbor_lists_for_generative_rerank_in_personalized_recomme/">NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems</a></strong></dt>
<dd>
<p>用邻居列表相对奖励对齐评估器-生成器目标，并以采样式非自回归生成灵活跳转组合空间，美团线上 CTR +3.25%。</p>
</dd>
<dt><strong><a href="multi_task_reranking/pcl_prompt_based_continual_learning_for_user_modeling_in_recommender_system/">PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems</a></strong></dt>
<dd>
<p>用位置级+上下文提示做持续用户建模，冻结骨干仅训提示，缓解灾难性遗忘，6/8 任务超越 MTL 上界。</p>
</dd>
<dt><strong><a href="multi_task_reranking/xmtf_a_formula_free_model_for_reinforcement_learning_based_multi_task_fusio/">xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems</a></strong></dt>
<dd>
<p>用可学习单调融合单元替代预定义融合公式，配合两阶段混合训练，离线总观看时长提升约7%，线上日观看时长+0.833%</p>
</dd>
</dl>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl>
<dt><strong><a href="rl_bandit_rec/auro_reinforcement_learning_for_adaptive_user_retention_optimization_in_rec/">AURO: Reinforcement Learning for Adaptive User Retention Optimization in Recommender Systems</a></strong></dt>
<dd>
<p>用值对齐状态抽象+拒绝采样守护探索，解决留存优化RL在非平稳环境下的适应与隐式冷启动难题，KuaiSim与线上A/B全面领先。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/interactive_visualization_recommendation_with_hier_sucb/">Interactive Visualization Recommendation with Hier-SUCB</a></strong></dt>
<dd>
<p>将可视化推荐建模为带可学习偏置项的分层上下文组合半臂老虎机，把遗憾界从 $O(\sqrt{T\ln^3(nm^2T\ln T)})$降至$O(\sqrt{T\ln^3(m^2T\ln T)})$，在 Plot.ly 上 80/160 轮超越离线 SOTA。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/reward_balancing_revisited_enhancing_offline_reinforcement_learning_for_rec/">Reward Balancing Revisited: Enhancing Offline Reinforcement Learning for Recommender Systems</a></strong></dt>
<dd>
<p>用扩散世界模型显式建模奖励不确定性，并以衰减加权的交互/熵双惩罚平衡多样性与长期收益，在三个真实数据集上刷新离线RL推荐SOTA。</p>
</dd>
<dt><strong><a href="rl_bandit_rec/value_function_decomposition_in_markov_recommendation_process/">Value Function Decomposition in Markov Recommendation Process</a></strong></dt>
<dd>
<p>将标准 TD 损失分解为策略随机与用户环境随机两个子目标，实现更快收敛、更强探索鲁棒性的长期价值估计。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/creator_side_recommender_system_challenges_designs_and_applications/">Creator-Side Recommender System: Challenges, Designs, and Applications</a></strong></dt>
<dd>
<p>提出创作者侧推荐系统 DualRec，将用户侧召回/排序镜像为物品找用户，并用用户可用性计算 UAC 解决用户可达性问题，快手线上 DAC 提升 2.9%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/datasets_for_navigating_sensitive_topics_in_recommendation_systems/">Datasets for Navigating Sensitive Topics in Recommendation Systems</a></strong></dt>
<dd>
<p>构建 ML-DDD 与 AO3 两个带敏感内容警告标签的推荐数据集，提出 Warning Amplification 指标，发现个性化 CF 反而去放大敏感内容。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/debias_can_be_unreliable_mitigating_bias_issue_in_evaluating_debiasing_reco/">Debias Can be Unreliable: Mitigating Bias Issue in Evaluating Debiasing Recommendation</a></strong></dt>
<dd>
<p>揭示随机曝光数据集上传统 Recall 评估与全曝光真值不一致，提出 URE 无偏估计方案并重估去偏方法。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/distributionally_robust_graph_out_of_distribution_recommendation_via_diffus/">Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model</a></strong></dt>
<dd>
<p>揭示 DRO 推荐对噪声分布过度加权之弊，用 VGAE+扩散去噪、Sinkhorn 距离与熵正则重构最坏情形优化，OOD 场景最高提升 75.5%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/graph_representation_learning_via_causal_diffusion_for_out_of_distribution_/">Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation</a></strong></dt>
<dd>
<p>用结构因果模型定位环境混杂因子，以变分推断+扩散模型学习环境不变图表征，OOD 场景最高提升 36.73%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/joint_evaluation_of_fairness_and_relevance_in_recommender_systems_with_pare/">Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier</a></strong></dt>
<dd>
<p>提出 DPFR：以测试集构造相关性-公平性帕累托前沿，用模型得分到前沿的欧氏距离作为联合评测指标，揭示现有联合指标排序与帕累托最优解不一致。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/on_device_content_based_recommendation_with_single_shot_embedding_pruning_a/">On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspective</a></strong></dt>
<dd>
<p>用 Shapley 值一次性剪枝 CTR 嵌入表，配合场感知码本替代零填充，无需重训即可适配任意端侧存储预算。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/reducing_symbiosis_bias_through_better_a_b_tests_of_recommendation_algorith/">Reducing Symbiosis Bias Through Better A/B Tests of Recommendation Algorithms</a></strong></dt>
<dd>
<p>揭示推荐算法共享训练数据导致的共生偏差，理论建模并仿真对比三种实验设计，用国家级分流实验验证偏差存在。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/unmasking_gender_bias_in_recommendation_systems_and_enhancing_category_awar/">Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness</a></strong></dt>
<dd>
<p>提出6个类别感知的性别偏差度量（含排序折扣）与GBS聚合，并以类别覆盖差作正则项，在3数据集上偏差降50–77%而性能损失有限。</p>
</dd>
</dl>
</div>

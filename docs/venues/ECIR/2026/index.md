---
hide:
  - toc
---

# 🏛️ ECIR 2026 推荐系统论文解读

11 篇 ECIR 2026 论文解读，覆盖 7 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl>
<dt><strong><a href="collaborative_filtering/a_reproducible_and_fair_evaluation_of_partition_aware_collaborative_filteri/">A Reproducible and Fair Evaluation of Partition-aware Collaborative Filtering</a></strong></dt>
<dd>
<p>在统一用户留出划分与 TPE 调参下复现 FPSR/FPSR+，发现 BISM 在 Amazon-CDs 反超，FPSR 家族在长尾显著占优。</p>
</dd>
</dl>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl>
<dt><strong><a href="self_supervised_denoising/post_training_denoising_of_user_profiles_with_llms_in_collaborative_filteri/">Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation</a></strong></dt>
<dd>
<p>用 LLM 零样本/少样本提示在推理前清洗用户画像，无需重训 CF 模型即获最高 13% NDCG/HR 提升。</p>
</dd>
</dl>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl>
<dt><strong><a href="generative_rec/efficient_optimization_of_hierarchical_identifiers_for_generative_recommend/">Efficient Optimization of Hierarchical Identifiers for Generative Recommendation</a></strong></dt>
<dd>
<p>复现SEATER并指出其平衡k-ary树构建是训练瓶颈，提出贪心/混合聚类将建树时间降至2%/5-8%且检索质量持平甚至更优。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl>
<dt><strong><a href="llm_rec/pre_trained_llms_meet_sequential_recommenders_efficient_user_centric_knowle/">Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation</a></strong></dt>
<dd>
<p>用预训练 LLM 生成文本用户画像并蒸馏进 SASRec/BERT4Rec 隐层，两阶段训练实现零推理开销的语义增强，Recall@10 最高提升 23.5%。</p>
</dd>
<dt><strong><a href="llm_rec/scinup_natural_language_user_interest_profiles_for_scientific_literature_re/">SciNUP: Natural Language User Interest Profiles for Scientific Literature Recommendation</a></strong></dt>
<dd>
<p>构建千名作者的自然语言兴趣画像学术推荐基准，系统对比稀疏/稠密/LLM重排，RRF 融合达 NDCG@10 0.4481。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/improving_conversational_recommendation_with_contextual_adaptation_of_exter/">Improving Conversational Recommendation with Contextual Adaptation of External Recommenders and LLM-based Reranking</a></strong></dt>
<dd>
<p>用轻量实体级序列推荐器产出候选、LLM 做上下文适配重排，ReDial/INSPIRED 上 H@5 平均提升 54%/25%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/interplay_training_independent_simulators_for_reference_free_conversational/">Interplay: Training Independent Simulators for Reference-Free Conversational Recommendation</a></strong></dt>
<dd>
<p>训练独立用户/推荐双模拟器实现无参考对话推荐，8B 模型 SR 达 95%、Recall@1 超 32B 基线 15 倍。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/usersimcrs_v2_simulation_based_evaluation_for_conversational_recommender_sy/">UserSimCRS v2: Simulation-Based Evaluation for Conversational Recommender Systems</a></strong></dt>
<dd>
<p>升级对话推荐用户模拟评测工具包，引入 LLM 模拟器、统一数据格式与 LLM-as-a-Judge 指标，案例揭示模拟器间系统排序显著分歧。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/are_multimodal_embeddings_truly_beneficial_for_recommendation_a_deep_dive_i/">Are Multimodal Embeddings Truly Beneficial for Recommendation? A Deep Dive into Whole vs. Individual Modalities</a></strong></dt>
<dd>
<p>用模态剔除（常量/噪声替换）在14个模型上做受控实证，发现文本主导、图像近乎无效，简单融合基线增益源于架构伪影。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/measuring_individual_user_fairness_with_user_similarity_and_effectiveness_d/">Measuring Individual User Fairness with User Similarity and Effectiveness Disparity</a></strong></dt>
<dd>
<p>提出 PUF 指标，用用户相似度加权有效性差异，首次同时刻画个体公平的两大要素，4 数据集 7 推荐器验证其敏感性与计算效率。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/modeling_behavioral_patterns_in_news_recommendations_using_fuzzy_neural_net/">Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks</a></strong></dt>
<dd>
<p>用多层模糊神经网络端到端学习可读逻辑规则预测新闻点击，阈值可调控制规则复杂度，EB-NeRD 上 nDCG@10 相对 NRMS 提升约 21.9%。</p>
</dd>
</dl>
</div>

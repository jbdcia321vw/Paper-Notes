---
hide:
  - toc
---

# 🏛️ ECIR 2025 推荐系统论文解读

11 篇 ECIR 2025 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl>
<dt><strong><a href="sequential_rec/saferec_self_attention_and_frequency_enriched_model_for_next_basket_recomme/">SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation</a></strong></dt>
<dd>
<p>在 SASRec 上并联频率感知打分模块，用物品出现频次与历史指示向量补足重复购买建模，Leave-One-Basket 全库排序下 Recall@10 最高提升 8%。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (3)</h2>
<dl>
<dt><strong><a href="llm_rec/contextualizing_spotify_s_audiobook_list_recommendations_with_descriptive_s/">Contextualizing Spotify&#x27;s Audiobook List Recommendations with Descriptive Shelves</a></strong></dt>
<dd>
<p>用 LLM 抽取十类书目描述符并生成多样化描述性货架标题，两轮线上 A/B 测试验证发现性与互动指标提升。</p>
</dd>
<dt><strong><a href="llm_rec/revisiting_language_models_in_neural_news_recommender_systems/">Revisiting Language Models in Neural News Recommender Systems</a></strong></dt>
<dd>
<p>在 MIND 上统一复现 8 种 LM 作为新闻编码器，发现更大模型不必然更优，但显著利好冷启动用户。</p>
</dd>
<dt><strong><a href="llm_rec/text2playlist_generating_personalized_playlists_from_text_on_deezer/">Text2Playlist: Generating Personalized Playlists from Text on Deezer</a></strong></dt>
<dd>
<p>用 LLM 抽取显隐式标签 + CF 亲和度重排 + LLM 列表精炼，将宽意图文本查询转为个性化歌单，Deezer 线上 20% 灰度中生成歌单后续收听率达 45% vs 人工 27%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/news_without_borders_domain_adaptation_of_multilingual_sentence_embeddings_/">News Without Borders: Domain Adaptation of Multilingual Sentence Embeddings for Cross-lingual News Recommendation</a></strong></dt>
<dd>
<p>构建 PolyNews 多语新闻语料，以 DAE+MT 领域适配 LaBSE 得 NaSE，冻结嵌入+晚期点击融合在 ZS-XLT 冷启动新闻推荐上达 SOTA。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/are_representation_disentanglement_and_interpretability_linked_in_recommend/">Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study</a></strong></dt>
<dd>
<p>复现5个推荐模型×4数据集，量化解耦度与可解释性，发现解耦与效果无关但与表示可解释性强正相关。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/embedding_cultural_diversity_in_prototype_based_recommender_systems/">Embedding Cultural Diversity in Prototype-based Recommender Systems</a></strong></dt>
<dd>
<p>在 ProtoMF 原型矩阵分解上引入 K-近邻原型过滤与原型正交分布正则，以物品国别为文化代理缓解流行度偏差，全库排序下长尾平均排名降 27%、HR@10 反升 2%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/how_to_diversify_any_personalized_recommender/">How to Diversify any Personalized Recommender?</a></strong></dt>
<dd>
<p>提出用户画像级预处理：用逻辑回归识别用户代表类别、userKNN 生成候选，再按 λ% 增删交互，使任意推荐器在保持精度的同时提升多样性与少数类曝光公平。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/improving_minimax_group_fairness_in_sequential_recommendation/">Improving Minimax Group Fairness in Sequential Recommendation</a></strong></dt>
<dd>
<p>将 CVaR-DRO 首次引入 SASRec 序列推荐，无需组标注即可提升小众/短序列用户的最差组 NDCG@20，Retailrocket 整体提升约 6%。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/okra_an_explainable_heterogeneous_multi_stakeholder_job_recommender_system/">OKRA: an Explainable, Heterogeneous, Multi-Stakeholder Job Recommender System</a></strong></dt>
<dd>
<p>异构知识图谱+多头GATv2构建求职者/企业双侧嵌入，谐波均值融合打分，nDCG@10较六基线最高提升约18%，并揭示城乡地域偏差。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/repeat_bias_aware_optimization_of_beyond_accuracy_metrics_for_next_basket_r/">Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation</a></strong></dt>
<dd>
<p>用 MILP 重排同时优化多样性/物品公平与重复偏差，在三个零售数据集上以可接受 Recall 损失显著改善 DS、logDP 与 RepBias。</p>
</dd>
</dl>
</div>

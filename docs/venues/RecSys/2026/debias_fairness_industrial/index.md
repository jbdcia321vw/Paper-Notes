# 🛡️ 去偏公平、可信隐私与工业架构

**RecSys 2026** · **8** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [CIKM 2026 (2)](../../../CIKM/2026/debias_fairness_industrial/) · [KDD 2026 (5)](../../../KDD/2026/debias_fairness_industrial/) · [SIGIR 2026 (11)](../../../SIGIR/2026/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2026 (4)](../../../TORS/2026/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与工业架构 ×3 · 评测基准 ×2 · 重排 ×2

<dl>
<dt><strong><a href="decoupled_learning_and_selection_in_slate_recommendation_for_privacy_and_st/">Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores</a></strong></dt>
<dd>
<p>将列表推荐拆为随机打分器+确定性选择器，给出隐私后处理契约与可审计的 margin 稳定性证书，锚点加权使 Flip@K 降 53–81%。</p>
</dd>
<dt><strong><a href="do_we_care_about_personalization_and_explainability_an_interview_study_with/">Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers</a></strong></dt>
<dd>
<p>对 9 家新闻机构 15 位工程师的半结构化访谈，揭示个性化采纳受编辑控制/隐私/资源约束，可解释性几乎只用于内部调试而非终端用户。</p>
</dd>
<dt><strong><a href="membership_inference_attacks_on_in_context_examples_in_llm_based_recommende/">Membership Inference Attacks on In-Context Examples in LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>首次针对 ICL 式 LLM 推荐系统提出 ItemMem 与 RecInertia 两种成员推理攻击，黑盒下攻击优势最高近 1.0，现有提示抽取防御基本失效。</p>
</dd>
<dt><strong><a href="on_the_convergent_validity_of_offline_evaluation_designs_for_recommender_sy/">On the Convergent Validity of Offline Evaluation Designs for Recommender Systems</a></strong></dt>
<dd>
<p>以稠密真值反馈为参照，用 Kendall&#x27;s τ 检验稀疏离线评测设计的收敛效度，发现无普适最优设计且 KuaiRec 上排序近乎反转。</p>
</dd>
<dt><strong><a href="probabilistic_residual_learning_for_online_recommendations/">Probabilistic Residual Learning for Online Recommendations</a></strong></dt>
<dd>
<p>以即插即用的层次贝叶斯残差模型对固定基座推荐器做概率用户聚类与 do-calculus 混杂去偏，冷启动跨域 Recall@20 最高提升约 7.6 倍。</p>
</dd>
<dt><strong><a href="training_seeds_and_model_selection_stability_in_recommender_system_evaluati/">Training seeds and model-selection stability in recommender-system evaluation</a></strong></dt>
<dd>
<p>固定数据划分与超参、仅变训练种子，从用户级指标敏感性、验证选模稳定性、验证-测试一致性与推荐列表一致性四层量化种子噪声，主张种子应纳入评估协议。</p>
</dd>
<dt><strong><a href="watchlens_a_configurable_platform_for_online_video_recommendation_experimen/">WatchLens: A Configurable Platform for Online Video Recommendation Experiments</a></strong></dt>
<dd>
<p>开源视频推荐在线实验平台，解耦界面/内容池/双表面策略并原生记录曝光上下文，30人被试验证观看页策略对会话留存的影响。</p>
</dd>
<dt><strong><a href="whose_posts_get_ranked_identical_text_exposure_gaps_in_bluesky_custom_feeds/">Whose Posts Get Ranked: Identical-Text Exposure Gaps in Bluesky Custom Feeds</a></strong></dt>
<dd>
<p>基于自然重复帖的曝光审计：同文本跨作者在250个Bluesky自定义Feed中曝光差异达33%，作者历史主导排名</p>
</dd>
</dl>

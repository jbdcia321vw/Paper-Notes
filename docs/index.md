---
hide:
  - navigation
  - toc
---

<style>
.recsys-hero {
  text-align: center;
  padding: 2.6rem 1.5rem 2.2rem;
  margin-bottom: 2rem;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(63,81,181,0.10) 0%, rgba(33,150,243,0.06) 50%, rgba(156,39,176,0.08) 100%);
  border: 1px solid rgba(63,81,181,0.16);
}
.recsys-hero h1 {
  font-size: 2.2rem;
  font-weight: 800;
  margin: 0 0 0.6rem 0;
  letter-spacing: -0.02em;
}
.recsys-hero p {
  font-size: 1.05rem;
  opacity: 0.82;
  max-width: 720px;
  margin: 0 auto 1.4rem;
}
.recsys-stats {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.recsys-stat-pill {
  padding: 0.45rem 1.1rem;
  border-radius: 999px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.2);
  font-size: 0.9rem;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.recsys-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
  margin: 1.2rem 0 2.2rem;
}
.recsys-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 1.15rem 1.25rem;
  border-radius: 12px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.18);
  text-decoration: none !important;
  color: var(--md-default-fg-color) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.recsys-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
  border-color: var(--md-primary-fg-color);
}
.recsys-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
}
.recsys-card-title {
  font-size: 1.02rem;
  font-weight: 700;
}
.recsys-badge {
  font-size: 0.75rem;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: rgba(63,81,181,0.12);
  color: var(--md-primary-fg-color);
  font-weight: 600;
  white-space: nowrap;
}
.recsys-card-desc {
  font-size: 0.84rem;
  opacity: 0.75;
  line-height: 1.45;
}
</style>

<div class="recsys-hero">
  <h1>📚 RecSys Paper Notes</h1>
  <p>5 分钟读懂一篇推荐系统与信息检索顶会/顶刊论文 · 支持「14 大推荐细分方向 × 22 个核心会刊」双维度检索</p>
  <div class="recsys-stats">
    <span class="recsys-stat-pill">📖 已收录 1 篇精读笔记</span>
    <span class="recsys-stat-pill">🔬 14 大推荐系统细分方向</span>
    <span class="recsys-stat-pill">🏛️ 22 个顶会与顶级期刊</span>
  </div>
</div>

## 🔬 按推荐系统细分类别浏览 (14 大核心方向)

<div class="recsys-grid">
  <a class="recsys-card" href="topics/collaborative_filtering/">    <div class="recsys-card-head">      <span class="recsys-card-title">🤝 协同过滤与表征学习</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">矩阵分解、双塔召回、对比学习自监督表征、解耦表征与负采样策略</div>  </a>
  <a class="recsys-card" href="topics/sequential_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">⏱️ 序列与会话推荐</span>      <span class="recsys-badge">1 篇</span>    </div>    <div class="recsys-card-desc">长短期用户行为序列建模、Transformer/Mamba/RNN 架构、会话匿名推荐与 Next-Item 预测</div>  </a>
  <a class="recsys-card" href="topics/generative_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">✨ 生成式推荐与语义 ID</span>      <span class="recsys-badge">1 篇</span>    </div>    <div class="recsys-card-desc">RQ-VAE 离散化语义 Token、TIGER 生成式检索范式、扩散模型推荐 (Diffusion Rec) 与自回归生成</div>  </a>
  <a class="recsys-card" href="topics/llm_agent_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🧠 大模型与智能体推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">大模型指令微调推荐、In-Context Prompting、知识蒸馏对齐、RecSys Agent 用户行为仿真</div>  </a>
  <a class="recsys-card" href="topics/multimodal_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🧩 多模态推荐与内容理解</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">视觉-文本-音频多模态特征融合、多模态图对比学习、短视频/微视频推荐与商品图文匹配</div>  </a>
  <a class="recsys-card" href="topics/graph_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🕸️ 图神经网络与知识图谱推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">LightGCN 谱图卷积改进、超图学习 (Hypergraph Rec)、知识图谱推理推荐 (KGRec) 与异构图建模</div>  </a>
  <a class="recsys-card" href="topics/social_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">👥 社交与群组推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">社交信任网络传播、社交去噪与同质性建模、群组偏好聚合 (Group Rec) 与意见领袖分析</div>  </a>
  <a class="recsys-card" href="topics/cold_start_cross_domain/">    <div class="recsys-card-head">      <span class="recsys-card-title">❄️ 冷启动与跨域/多场景推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">新用户/新物品冷启动、元学习 (Meta-Learning)、跨域迁移学习 (CDR) 与多场景自适应建模</div>  </a>
  <a class="recsys-card" href="topics/ctr_multi_task_ranking/">    <div class="recsys-card-head">      <span class="recsys-card-title">🎯 CTR/CVR 预估与多任务精排</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">高阶特征交叉网络、MMoE/PLE 多目标优化、粗排-精排-重排级联一致性与延迟反馈建模</div>  </a>
  <a class="recsys-card" href="topics/conversational_explainable/">    <div class="recsys-card-head">      <span class="recsys-card-title">🗣️ 对话推荐与可解释性</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">多轮主动追问对话推荐 (CRS)、推荐理由生成、反事实可解释性与用户意图澄清</div>  </a>
  <a class="recsys-card" href="topics/rl_bandit_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🎮 强化学习与动态探索推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存优化、Slate 列表级策略优化</div>  </a>
  <a class="recsys-card" href="topics/debias_fairness_trust/">    <div class="recsys-card-head">      <span class="recsys-card-title">⚖️ 去偏、公平性与可信隐私推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">曝光/流行度/位置偏差消除、因果推断推荐 (Causal Rec)、群体公平性、联邦推荐与鲁棒防投毒</div>  </a>
  <a class="recsys-card" href="topics/poi_spatiotemporal_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">📍 时空与 POI 本地生活推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">Next-POI 兴趣点推荐、时空轨迹建模、LBS 本地生活与移动出行推荐</div>  </a>
  <a class="recsys-card" href="topics/industrial_system_eval/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏭 工业界系统架构与评测基准</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">工业级召回/排序全链路实战、GPU 端到端加速、线上 A/B 测试经验、开源数据集与评测新范式</div>  </a>
</div>

## 🏛️ 按顶会与顶刊浏览 (SIGIR / RecSys / KDD / WWW / ACM MM / TOIS / TKDE)

<div class="recsys-grid">
  <a class="recsys-card" href="venues/SIGIR2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ SIGIR</span>      <span class="recsys-badge">1 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · IR/RecSys 顶会 · ACM SIGIR Conference on Research and Development in Information Retrieval</div>  </a>
  <a class="recsys-card" href="venues/RecSys2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ RecSys</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">旗舰专会 · RecSys 顶会 · ACM Conference on Recommender Systems</div>  </a>
  <a class="recsys-card" href="venues/WSDM2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ WSDM</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-B · 搜索与推荐顶会 · ACM International Conference on Web Search and Data Mining</div>  </a>
  <a class="recsys-card" href="venues/CIKM2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ CIKM</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-B · IR/KM 顶会 · ACM International Conference on Information and Knowledge Management</div>  </a>
  <a class="recsys-card" href="venues/ECIR2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ ECIR</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">IR 重要会议 · European Conference on Information Retrieval</div>  </a>
  <a class="recsys-card" href="venues/SIGIR-AP2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ SIGIR-AP</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">IR 亚太专会 · ACM SIGIR Conference on Information Retrieval in the Asia Pacific</div>  </a>
  <a class="recsys-card" href="venues/KDD2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ KDD</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · 数据挖掘与工业推荐顶会 · ACM SIGKDD Conference on Knowledge Discovery and Data Mining</div>  </a>
  <a class="recsys-card" href="venues/WWW2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ WWW (TheWebConf)</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · Web 与推荐系统顶会 · ACM Web Conference</div>  </a>
  <a class="recsys-card" href="venues/ACM_MM2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ ACM MM</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · 多模态推荐重镇 · ACM International Conference on Multimedia</div>  </a>
  <a class="recsys-card" href="venues/ICDE2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ ICDE</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · 数据工程顶会 · IEEE International Conference on Data Engineering</div>  </a>
  <a class="recsys-card" href="venues/ICDM2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ ICDM</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-B · 数据挖掘顶会 · IEEE International Conference on Data Mining</div>  </a>
  <a class="recsys-card" href="venues/DASFAA2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ DASFAA</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-B · 数据库与推荐应用会议 · International Conference on Database Systems for Advanced Applications</div>  </a>
  <a class="recsys-card" href="venues/AAAI2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ AAAI (RecSys Track)</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · AI 综合顶会 · AAAI Conference on Artificial Intelligence</div>  </a>
  <a class="recsys-card" href="venues/IJCAI2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ IJCAI (RecSys Track)</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · AI 综合顶会 · International Joint Conference on Artificial Intelligence</div>  </a>
  <a class="recsys-card" href="venues/ACL_EMNLP2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ ACL / EMNLP (RecSys & CRS)</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A/B · NLP 与对话推荐顶会 · ACL & EMNLP Conferences</div>  </a>
  <a class="recsys-card" href="venues/ICLR_NeurIPS_ICML2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">🏛️ ICLR / NeurIPS / ICML (RecSys)</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · 机器学习三大顶会 · Top ML Conferences (RecSys Papers)</div>  </a>
  <a class="recsys-card" href="venues/TOIS2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">📖 ACM TOIS</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · 信息系统与推荐顶刊 · ACM Transactions on Information Systems</div>  </a>
  <a class="recsys-card" href="venues/TKDE2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">📖 IEEE TKDE</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-A · 知识与数据工程顶刊 · IEEE Transactions on Knowledge and Data Engineering</div>  </a>
  <a class="recsys-card" href="venues/TORS2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">📖 ACM TORS</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">ACM 推荐系统旗舰专刊 · ACM Transactions on Recommender Systems</div>  </a>
  <a class="recsys-card" href="venues/TMM2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">📖 IEEE TMM</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-B · 多模态推荐顶刊 · IEEE Transactions on Multimedia</div>  </a>
  <a class="recsys-card" href="venues/TKDD2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">📖 ACM TKDD</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-B · 数据挖掘顶刊 · ACM Transactions on Knowledge Discovery from Data</div>  </a>
  <a class="recsys-card" href="venues/TWEB2026/">    <div class="recsys-card-head">      <span class="recsys-card-title">📖 ACM TWEB</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">CCF-B · Web 顶刊 · ACM Transactions on the Web</div>  </a>
</div>

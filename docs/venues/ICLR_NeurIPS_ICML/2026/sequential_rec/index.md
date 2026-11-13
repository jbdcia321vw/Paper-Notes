# ⏱️ 序列与会话推荐

**ICLR / NeurIPS / ICML (RecSys) 2026** · **5** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** 精排 ×5 · NeurIPS ×2

<dl>
<dt><strong><a href="collectivekv_decoupling_and_sharing_collaborative_information_in_sequential/">CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation</a></strong></dt>
<dd>
<p>用可学习全局 K/V 池跨用户共享协同信息，将 KV cache 压缩至 0.8% 且精度不降，解码延迟随 batch 增长近乎平坦。</p>
</dd>
<dt><strong><a href="cramer_control_via_request_aware_masking_for_editing_recommenders/">CRAMER: Control via Request-Aware Masking for Editing Recommenders</a></strong></dt>
<dd>
<p>将自然语言请求编码为稀疏行列门控掩码，对冻结序列推荐骨干做后验行为编辑，采样100负例排序下全面超越四类请求感知基线且推理仅增0.018s。</p>
</dd>
<dt><strong><a href="lime_link_based_user_item_interaction_modeling_with_decoupled_xor_attention/">LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling</a></strong></dt>
<dd>
<p>用低秩 link embedding 解耦用户-候选交互并预计算 QK 权重，配合 XOR 线性注意力将长历史建模降至 O(N)，实现 HSTU 级精度与 10× 推理加速。</p>
</dd>
<dt><strong><a href="massive_memorization_with_hundreds_of_trillions_of_parameters_for_sequentia/">Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders</a></strong></dt>
<dd>
<p>VISTA 将超长用户历史两阶段解耦：离线用准线性注意力压缩为数百种子嵌入并缓存，下游仅做候选-摘要注意力，工业上线推理 GPU 降 94%。</p>
</dd>
<dt><strong><a href="not_all_is_lost_repairing_lossy_user_preference_states_of_personalization_e/">Not All Is Lost: Repairing Lossy User Preference States of Personalization Encoders</a></strong></dt>
<dd>
<p>冻结宿主编码器与任务头，仅训练轻量 REPAIR 模块，用缓存逐步表征对压缩偏好状态做时序结构化残差校正，12 个推荐宿主 MRR/nDCG@10 全面提升。</p>
</dd>
</dl>

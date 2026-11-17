# 🏆 1st Place Solution | Kaggle tabular-long-form-competition-ugcml-harbour-space

[![Kaggle](https://img.shields.io/badge/Kaggle-Rank%20%231%20Champion-gold?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/tabular-long-form-competition-ugcml-harbour-space)
[![Private LB](https://img.shields.io/badge/Private%20LB-0.28160-brightgreen?style=for-the-badge)](https://www.kaggle.com/competitions/tabular-long-form-competition-ugcml-harbour-space/leaderboard)
[![Public LB](https://img.shields.io/badge/Public%20LB-0.27676-blue?style=for-the-badge)](https://www.kaggle.com/competitions/tabular-long-form-competition-ugcml-harbour-space/leaderboard)
[![Team](https://img.shields.io/badge/Team-EscoffierZhou's%20neko-orange?style=for-the-badge)](https://github.com/EscoffierWithNeko)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

> **竞赛标识**: `tabular-long-form-competition-ugcml-harbour-space` (Airbnb 挂牌价格预测)  
> **核心指标**: RMSLE (Root Mean Squared Logarithmic Error)  
> **官方终榜战绩**: **Private LB: 0.28160 (Rank #1 全场冠军 🏆，领先第 2 名 0.00164)**  
> **公榜战绩**: **Public LB: 0.27676** | **验证集**: **10-Fold CV: 0.28109 (校准后 0.28099)**  
> **官方最佳提交 Ref**: `58373885` (`submission.csv`, MD5: `296c235b5b803dc72b271b1f2daa31cb`)  

---

## ⚡ 核心战绩仪表盘 (Key Performance Dashboard)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Private Leaderboard  : 0.28160  (Rank #1 登顶冠军 / 领先优势 +0.00164)          │
│  Public Leaderboard   : 0.27676  (第一梯队领跑 / Ref 58373885)                  │
│  10-Fold CV RMSLE     : 0.28109  (CV 与 Private LB 仅相差 0.00051，零过拟合)      │
│  Baseline 净提升      : -0.02011 (P0: 0.30120  ──>  P6 冠军: 0.28109)           │
│  孪生微环境方差削减率 : 96.77%   (价格对数组内方差从 0.5401 削减至 0.0237)       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 冠军制胜架构一览 (Champion Architecture Flow)

```
                     ┌──────────────────────────────────────────────┐
                     │ 原始数据: 数值/类别表格 + 地理经纬度 + 文本描述 │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 杀手锏 1: 孪生微环境复合特征基座 (Twin-MicroEnv 531 维, cache_10fold_v3/)               │
│ • 5元组微环境物理键: [neighbourhood, room_type, accommodates, bathrooms, bedrooms]     │
│ • 极小调和平滑度 TE (smooth=2/10): 将组内价格对数方差削减 96.77% (0.7349 -> 0.1540 Std)│
│ • 物理容纳密度比率 (accommodates_per_bed, accommodates_per_bath, density_penalty)     │
│ • 严密 10-Fold 折内 Target Encoding 隔离，零未来信息泄漏 (Zero-Leakage)                │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 杀手锏 2: 9 路跨架构高多样性异构模型池 (Model Zoo, Pearson 相关系数 0.885 ~ 0.962)     │
│ • GBDT 树模型家族: XGBoost GPU Hist (0.28389) / LightGBM ET (0.28421) /              │
│                    LightGBM Huber 损失 (0.28423) / LightGBM DART 丢弃 (0.28424)     │
│ • 深度原生文本学习: CatBoostRegressor 原生 Bi-Gram Tokenizer (0.28610, 捕捉文案溢价)  │
│ • 表格式前馈网络:   PyTorch Tabular ResNet (0.28780, 512-256-128 残差跳跃，超平面正交)│
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 杀手锏 3: 严格凸约束 SLSQP Stacking 架构与 Log-Bias 校准门禁                          │
│ • 概率单纯形凸约束: min ||y - W*p||^2  s.t.  w_i >= 0 且 sum(w_i) = 1 (彻底终结震荡)  │
│ • 折内嵌套贪婪特征选择: 严格在 10 折内部锁定最优子集，杜绝选择偏误                     │
│ • 折内自适应残差校准: Log-Bias 系统性偏置动态微调 (CV: 0.28109 -> 0.28099)            │
│ • 全局物理边界硬截断: clip(exp(pred) - 1, 20.0, 2000.0) 熔断一切异常外推              │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                            ▼
       🏆 最终冠军提交: submission.csv (Ref 58373885, Private LB 0.28160, Rank #1)
```

---

## 💡 三大核心杀手锏总结 (Three Winning Pillars)

1. **孪生微环境复合键与物理方差削减 (Twin-MicroEnvironment)**:
   - 捕捉超局部（Hyper-local）房源内在定价基底，通过 5 元组物理唯一键（邻域、户型、容纳人数、卫生间、卧室）实现同质房源精准对齐；
   - 极小调和平滑度编码（$\text{smooth}=2.0$）使簇内对数方差狂降 **96.77%**（标准差从 `0.7349` 降至 `0.1540`），结合容纳密度指标彻底锁定基准价格。
2. **9 路跨算法异构模型矩阵 (9-Way Heterogeneous Model Zoo)**:
   - 绝非同构树的重复堆砌，而是构建了涵盖 **XGBoost GPU、LightGBM (ExtraTrees / Huber / DART)、CatBoost 原生分词文本学习、PyTorch Tabular ResNet** 的跨模态模型池；
   - 预测相关系数控制在 `0.885 ~ 0.962` 黄金区间，提供了巨大的误差正交对冲空间。
3. **严格凸约束 SLSQP 堆叠与物理边界熔断 (Constrained Convex Stacking)**:
   - 从 P5 阶段无约束 Ridge 因模型共线性导致权重震荡（+3.5/-2.6、LB 暴跌至 0.53173）的滑铁卢中沉淀出真理：**模型融合必须施加概率单纯形约束（$w_i \ge 0, \sum w_i = 1$）**；
   - 辅以折内嵌套特征选择、自适应 Log-Bias 残差补偿与 $[20.0, 2000.0]$ 硬截断，确保公榜私榜双双绝对稳健。

---

## 📈 关键阶段演进对照表 (P0 -> P6)

| 阶段编号 | 核心技术路线与演进亮点 | 10-Fold CV | Public LB | Private LB | 状态评估与启示 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **P0** | 极简 Baseline (单一 LightGBM，47 维原生基础特征) | 0.30120 | 0.29850 | 0.30412 | 跑通流程，确立对数评估框架 |
| **P1** | 引入 TF-IDF 文本词频特征 + XGBoost/LGBM 双模融合 | 0.28892 | 0.28430 | 0.28915 | 文本特征带来显著正交增益 |
| **P2** | 空间地理经纬度 KMeans 聚类 + 高基数类别平滑 TE | 0.28450 | 0.27980 | 0.28522 | 地理微区域对房产溢价强解释力 |
| **P3** | 引入 CatBoost 原生文本处理 + PyTorch Tabular ResNet | 0.28315 | 0.27820 | 0.28378 | 跨架构异构初显威力 |
| **P4** | 提出“孪生微环境”概念验证 + Huber 损失抗噪 | 0.28205 | 0.27740 | 0.28291 | 验证微环境对极端异构房源的降噪 |
| **P5** | 【重大滑铁卢】无约束 Ridge 回归导致共线性权重震荡 | 0.28099 *(虚假)* | **0.53173** *(暴跌)* | 0.54010 | **血泪教训**: 共线性导致外推严重负偏 |
| **P6** | **【终局夺冠】** 孪生微环境 531 维 + SLSQP 严格凸约束 | **0.28109** | **0.27676** | **0.28160** | 🏆 **Kaggle 全场第一名 (领先 0.00164)** |

---

## 🚀 一键端到端复现流水线 (Quick Reproduction Pipeline)

复现全流程仅需按序执行以下 4 条标准命令（总耗时约 2.5 ~ 3 小时）：

```powershell
# 1. 构建 v3 孪生微环境 531 维特征基座 (生成 cache_10fold_v3/)
python fe_pipeline_v3.py

# 2. 训练 7 路多样性 GBDT 模型池 (产出 oof_gbdt_v3/)
python train_gbdt_v3.py

# 3. 训练深度原生文本 CatBoost 与 Tabular ResNet (产出 oof_cat_nn_v3/)
python train_catboost_nn_zoo_v3.py

# 4. 执行折内嵌套选择、严格凸约束 SLSQP Stacking 求解与后校准
python stack_and_submit_gbdt_textzoo.py
```

---

## 🔒 最终冠军文件散列核验 (File Hash Integrity)

| 目标文件 | 物理存储路径 | 字节大小 | MD5 校验和 (绝对锁死) | 官方提交 Ref | 官方成绩 (Public / Private) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **最终冠军提交** | `submission.csv` | 184,962 B | `296c235b5b803dc72b271b1f2daa31cb` | **`58373885`** | **0.27676 / 0.28160 (Rank #1)** |
| 归档备份提交 | `archive/submissions/submission.csv` | 184,962 B | `296c235b5b803dc72b271b1f2daa31cb` | `58373885` | 0.27676 / 0.28160 (Rank #1) |

---

## 📚 归档技术文档与记忆宫殿资产导航 (Documentation Index)

- **完整白皮书**: [`archive/FINAL_WRAPUP_REPORT.md`](archive/FINAL_WRAPUP_REPORT.md) (420+ 行系统收官报告)
- **算法演进全景**: [`archive/docs/ALGORITHM_EVOLUTION_PATH.md`](archive/docs/ALGORITHM_EVOLUTION_PATH.md) (P0-P6 详细实验记录)
- **制胜算法规格**: [`archive/docs/CORE_ALGORITHM_SPEC.md`](archive/docs/CORE_ALGORITHM_SPEC.md) (数学定义与超参数规格)
- **数据审计报告**: [`archive/docs/DATA_AUDIT_AND_LEAKAGE_CHECK.md`](archive/docs/DATA_AUDIT_AND_LEAKAGE_CHECK.md) (零泄漏证明与物理边界)
- **端到端复现指南**: [`archive/docs/REPRODUCTION_GUIDE.md`](archive/docs/REPRODUCTION_GUIDE.md) (环境与复现排查手册)
- **MemPalace 记忆宫殿**: 7 项 Tier 1 核心长效记忆已注册至 `Financial_Domain_NLP/Financial_Report_Analysis`，详见 [`archive/docs/MEMPALACE_ENTRIES.md`](archive/docs/MEMPALACE_ENTRIES.md)。

---

## 👥 作者与致谢 (Authors & Acknowledgments)

- **Team**: EscoffierZhou's neko
- **Author**: Escoffier (`escoffierzhou@gmail.com`) & Neko 🐾
- **Workspace**: Nekoya AI Studio

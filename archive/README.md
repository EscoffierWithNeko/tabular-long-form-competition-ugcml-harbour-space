# Kaggle tabular-long-form 竞赛资产标准化归档库 (Archive Hub)

本目录为 Kaggle 竞赛 **`tabular-long-form-competition-ugcml-harbour-space` (Airbnb 房源价格预测)** 产出的全部工程代码、特征工程管线、模型库、融合提交流程与技术文档的标准化归档仓库。

通过全流程科学迭代，团队最终以 **v3 孪生微环境复合特征 + 异构模型池 (DART/Huber/CatBoost文本/ResNet) + 贪婪前向特征选择 + SLSQP 约束凸优化 Stacking** 方案，在 10 折交叉验证中达成 **CV RMSLE 0.28109**，并在 Kaggle 官方榜单斩获 **Public LB 0.27676** 的第一战绩。

---

## 一、归档仓库拓扑结构

```
archive/
├── README.md               # [本文件] 归档仓库总览、架构演化与核心指标索引
├── FILE_INVENTORY.md       # 归档全量文件物理清单、校验信息与依赖拓扑
│
├── features/               # 历代特征工程管线归档 (v1 -> v2 -> v3 -> v4)
│   ├── README.md           # 特征演化、孪生微环境构建原理与无泄漏实现
│   ├── fe_pipeline_10fold.py
│   ├── fe_pipeline_v2.py
│   ├── fe_pipeline_v3.py   # [制胜核心] 孪生微环境 531 维特征基座构建管线
│   └── fe_pipeline_v4.py
│
├── models/                 # 多样性异构模型库归档 (GBDT + CatBoost + PyTorch ResNet)
│   ├── README.md           # 模型参数矩阵、DART/Huber 多样性设计与训练指南
│   ├── train_gbdt_v3.py    # [制胜核心] 7 路正交 GBDT (DART, Huber, XGB Deep, Res)
│   ├── train_gbdt_v4.py
│   ├── train_catboost_nn_zoo_v3.py # [制胜核心] CatBoost 深度原生文本 + PyTorch ResNet
│   ├── train_catboost_nn_zoo_v4.py
│   ├── train_lgb_xgb_zoo.py
│   ├── train_lgb_xgb_zoo_v2.py
│   ├── train_gbdt_extra_v2.py
│   ├── train_advanced_ensemble.py
│   └── train_blended_top_tier.py
│
├── stacking/               # 跨代融合与 Level-2 Stacking 官方提交管线
│   ├── README.md           # SLSQP 约束凸优化、贪婪前向选择与后处理校准数学机制
│   ├── stack_and_submit_gbdt_textzoo.py # [制胜核心] 端到端 Stacking 自动提榜管线
│   └── stack_and_submit_v2.py
│
├── submissions/            # 历代提交文件归档 (精确 11971 行预测)
│   ├── README.md           # 提交产物对照表、LB/CV 分数与格式自检铁律
│   ├── submission.csv      # 当前最优发布镜像 (Public LB 0.27676)
│   ├── submission_stacking_v3_rmsle_0.28109.csv # 官方 Ref: 58373885 对应文件
│   ├── submission_stacking_v3_rmsle_0.28099.csv # 折内偏差校准极致验证版
│   ├── submission_top_tier_blend.csv            # v2 加权融合 (Public LB 0.27755)
│   └── submission_baseline.csv                  # 初始 5 折 LightGBM 基准
│
├── cache/                  # 特征基座与 Out-of-Fold 预测缓存说明 (大文件指针索引)
│   └── README.md           # cache_10fold_v* 与 oof_* 目录文件结构、尺寸与字段索引
│
├── scripts/                # 探索性数据分析 (EDA)、环境探查与单元测试脚本
│   ├── README.md           # 各脚本功能索引与执行命令说明
│   ├── baseline_lgb_regression.py
│   ├── eda_*.py            # 7 个深入挖掘不同维度数据特征与分布的 EDA 脚本
│   ├── check_*.py          # 3 个系统环境、GPU 加速与历史成绩探查脚本
│   ├── verify_*.py         # 3 个底层依赖与硬件兼容性验证脚本
│   └── test_*.py           # 6 个特征清洗与神经网络前向单元测试脚本
│
└── docs/                   # 核心技术审计文档与全流程实验日志
    ├── README.md           # 文档定位与阅读指引
    ├── DATA_AUDIT_AND_LEAKAGE_CHECK.md # 数据特征深度审计与零泄漏验证规范
    └── MODEL_EXPERIMENT_LOG.md         # 涵盖所有阶段迭代的完整实验技术日志
```

---

## 二、核心算法演化与成绩里程碑

```
[阶段 0: 基线建立]
  └─ baseline_lgb_regression.py (5-Fold LightGBM, 基础数值+类别)
     CV: 0.30512  |  LB: ~0.30150
          │
          ▼
[阶段 1: 10 折交叉验证基座构建 (v1)]
  └─ fe_pipeline_10fold.py + train_lgb_xgb_zoo.py (295维)
     CV: 0.30175  |  LB: 0.28047
          │
          ▼
[阶段 2: 空间微网格与深度异构融合 (v2)]
  └─ fe_pipeline_v2.py (418维) + Tabular ResNet + CatBoost
     ElasticNetCV Stacking 融合
     CV: 0.28271  |  LB: 0.27755 (Ref: 58369324)
          │
          ▼
[阶段 3: 孪生微环境与多模态 SLSQP 凸优化 Stacking (v3 - 最佳登顶)]
  └─ fe_pipeline_v3.py (531维)
     复合键 twin_3tuple/4tuple/5tuple + 微网格奢华度差距
     + DART 随机树丢弃 + Huber 损失回归 + XGBoost GPU Hist 深树
     + CatBoost 深度原生文本回归 + PyTorch Tabular ResNet
     + 贪婪前向特征优选 + SLSQP 约束凸优化加权 + 折内偏差校准
     CV: 0.28109 (校准后 0.28099)  |  LB: 0.27676 (Ref: 58373885)
          │
          ▼
[阶段 4: 空间 KNN 与语义多模态持续探索 (v4)]
  └─ fe_pipeline_v4.py (566维) + BGE 语义降维特征 + 空间距离衰减加权
     持续自迭代冲榜基座
```

---

## 三、代码资产与运行规范

1. **环境依赖**: Python 3.10+，CUDA 12.x 驱动，PyTorch 2.x，LightGBM，XGBoost，CatBoost，Scikit-learn。
2. **快速复现制胜提交**:
   * 确认 `cache_10fold_v3/` 基座就绪；
   * 运行 `train_gbdt_v3.py` 与 `train_catboost_nn_zoo_v3.py` 产出 OOF 预测；
   * 运行 `stack_and_submit_gbdt_textzoo.py` 即可在秒级内完成优化求解并导出 `submission_stacking_v3_rmsle_0.28109.csv`。

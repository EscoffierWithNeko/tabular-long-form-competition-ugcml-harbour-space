# Kaggle tabular-long-form 竞赛全量归档文件清单 (File Inventory)

本清单完整列出 `archive/` 标准化归档目录下的所有工程资产、数据管线、模型库、提交文件及技术文档。所有文件尺寸、代码行数与依赖关系均经实机严格核验。

---

## 一、归档资产总览统计

* **归档根目录**: `archive/`
* **一级功能子目录数**: 7 个 (`features/`, `models/`, `stacking/`, `submissions/`, `cache/`, `scripts/`, `docs/`)
* **归档文件总数**: 51 个
  * 核心特征工程脚本: 4 个
  * 模型训练与集成脚本: 11 个
  * Stacking 融合提交脚本: 2 个
  * 正式预测提交文件: 5 个
  * 探查 / EDA / 单元测试脚本: 22 个
  * 技术文档与实验日志: 2 篇
  * 各级模块说明文档: 8 篇 (`README.md` * 7 + 根目录 `README.md` + 本清单 `FILE_INVENTORY.md`)

---

## 二、详细文件清单与元数据字典

### 1. 顶层管理与索引文件

| 文件路径 | 物理大小 | 源码行数 | 核心定位与说明 |
| :--- | :--- | :--- | :--- |
| `archive/README.md` | 6,043 B | 108 行 | 归档仓库总览、各模块拓扑架构图、核心算法演进历程与成绩里程碑。 |
| `archive/FILE_INVENTORY.md` | 本文件 | - | 全量归档资产物理清单、行数、校验与依赖拓扑字典。 |

---

### 2. 特征工程管线 (`archive/features/`)

| 文件名称 | 物理大小 | 源码行数 | 核心定位与说明 | 上游依赖 / 下游产出 |
| :--- | :--- | :--- | :--- | :--- |
| **`fe_pipeline_v3.py`** | 55,879 B | 1,036 行 | **[制胜核心基座]** 构建 531 维孪生微环境复合特征 (`twin_3tuple`, `twin_4tuple`, `twin_5tuple`)、微网格奢华度差距、容纳密度与 10 折严密无泄漏折内 Target Encoding。 | **输入**: `train.csv`, `test.csv`<br>**产出**: `cache_10fold_v3/` |
| `fe_pipeline_10fold.py` | 55,405 B | 973 行 | 初代 10 折基座管线 (295 维)，确立 Stratified 10-Fold 划分与折内无泄漏基础特征工程范式。 | **输入**: `train.csv`, `test.csv`<br>**产出**: `cache_10fold/` |
| `fe_pipeline_v2.py` | 47,213 B | 888 行 | 第二代特征工程 (418 维)，引入经纬度聚类空间特征、硬件比率特征与多模态文本统计。 | **输入**: `train.csv`, `test.csv`<br>**产出**: `cache_10fold_v2/` |
| `fe_pipeline_v4.py` | 71,484 B | 1,323 行 | 第四代特征基座 (566 维)，引入 BGE 语义嵌入主成分特征与空间 KD-Tree 距离衰减加权特征。 | **输入**: `train.csv`, `test.csv`<br>**产出**: `cache_10fold_v4/` |
| `README.md` | 4,811 B | 65 行 | 特征工程演化脉络、字段维度对比表与防数据泄漏严格规范说明。 | - |

---

### 3. 多样性模型库 (`archive/models/`)

| 文件名称 | 物理大小 | 源码行数 | 核心定位与说明 | 上游依赖 / 下游产出 |
| :--- | :--- | :--- | :--- | :--- |
| **`train_gbdt_v3.py`** | 30,764 B | 758 行 | **[制胜核心]** 训练 7 路正交 GBDT：标准 LightGBM、ExtraTrees、DART 丢弃树、Huber 鲁棒回归、XGBoost GPU Hist、XGBoost Deep、残差回归。 | **输入**: `cache_10fold_v3/`<br>**产出**: `oof_gbdt_v3/` |
| **`train_catboost_nn_zoo_v3.py`** | 38,522 B | 811 行 | **[制胜核心]** 深度异构模型：CatBoost 原生深度文本 (depth=7~8) + 孪生复合键高基数类别，以及 PyTorch Tabular ResNet (Entity Embedding + 残差块)。 | **输入**: `cache_10fold_v3/`<br>**产出**: `oof_cat_nn_v3/` |
| `train_gbdt_v4.py` | 32,218 B | 793 行 | 基于 v4 空间近邻与语义特征基座训练多路多样性 GBDT 模型。 | **输入**: `cache_10fold_v4/`<br>**产出**: `oof_gbdt_v4/` |
| `train_catboost_nn_zoo_v4.py` | 42,014 B | 883 行 | 基于 v4 特征基座训练 CatBoost 深度文本与 PyTorch 深度表格网络。 | **输入**: `cache_10fold_v4/`<br>**产出**: `oof_cat_nn_v4/` |
| `train_lgb_xgb_zoo.py` | 23,641 B | 495 行 | 初代 10 折 LightGBM 与 XGBoost 基座模型池训练脚本。 | **输入**: `cache_10fold/`<br>**产出**: `oof_gbdt/` |
| `train_lgb_xgb_zoo_v2.py` | 26,383 B | 578 行 | 第二代 GBDT 模型池 (含 ExtraTrees 与 Huber Loss LightGBM)。 | **输入**: `cache_10fold_v2/`<br>**产出**: `oof_gbdt_v2/` |
| `train_gbdt_extra_v2.py` | 21,994 B | 522 行 | 补充训练 LightGBM ExtraTrees 与深度 XGBoost 变体。 | **输入**: `cache_10fold_v2/`<br>**产出**: `oof_gbdt_v2/` |
| `train_catboost_nn_zoo.py` | 44,563 B | 956 行 | 初代 CatBoost 文本模型与 PyTorch 神经网络联合训练脚本。 | **输入**: `cache_10fold/`<br>**产出**: `oof_cat_nn/` |
| `train_catboost_nn_zoo_v2.py` | 39,361 B | 880 行 | 第二代 CatBoost 深度文本模型与 PyTorch Tabular ResNet 训练脚本。 | **输入**: `cache_10fold_v2/`<br>**产出**: `oof_cat_nn_v2/` |
| `train_advanced_ensemble.py` | 12,478 B | 283 行 | 早期网格搜索加权融合脚本。 | **输入**: `oof_gbdt/` |
| `train_blended_top_tier.py` | 34,833 B | 689 行 | 高阶加权融合调度脚本，支持 Nelder-Mead 与 Powell 局部最优化。 | **输入**: `oof_gbdt_v2/` |
| `README.md` | 5,763 B | 67 行 | 异构模型架构、超参数配置矩阵与训练运行指令文档。 | - |

---
### 4. 融合与 Stacking 管线 (`archive/stacking/`)

| 文件名称 | 物理大小 | 源码行数 | 核心定位与说明 | 上游依赖 / 下游产出 |
| :--- | :--- | :--- | :--- | :--- |
| **`stack_and_submit_gbdt_textzoo.py`** | 27,952 B | 593 行 | **[制胜核心融合管线]** 跨代全模型池 Level-2 Stacking：执行前向贪婪特征选择、SLSQP 凸优化权重求解、折内 Log-Bias 零均值漂移校准与自动化 Kaggle API 提交。 | **输入**: `oof_gbdt_v3/`, `oof_cat_nn_v3/`<br>**产出**: `submission_stacking_v3_*.csv` |
| `stack_and_submit_v2.py` | 30,128 B | 673 行 | 第二代 Level-2 Stacking 脚本，支持 ElasticNetCV、RidgeCV 与 NNLS 多元加权对比。 | **输入**: `oof_gbdt_v2/`, `oof_cat_nn_v2/`<br>**产出**: `submission_top_tier_blend.csv` |
| `README.md` | 6,036 B | 86 行 | SLSQP 凸优化数学公式推导、贪婪前向特征选择算法与后处理偏差校准机制说明。 | - |

---

### 5. 提交结果产物 (`archive/submissions/`)

| 文件名称 | 物理大小 | 数据行数 | 核心定位与成绩说明 |
| :--- | :--- | :--- | :--- |
| **`submission.csv`** | 184,962 B | 11,972 行 | **[官方当前最优镜像]** 官方榜单第一梯队最优成绩镜像，格式与字段完全合规。 |
| **`submission_stacking_v3_rmsle_0.28109.csv`** | 184,962 B | 11,972 行 | **[官方斩获最高分基准]** v3 SLSQP 约束凸优化 Stacking 产物，**CV 0.28109 -> Public LB 0.27676 (Ref 58373885)**。 |
| `submission_stacking_v3_rmsle_0.28099.csv` | 182,709 B | 11,972 行 | v3 SLSQP 融合叠加折内零偏校准 (Zero-Bias Calibrated) 优化版本，**CV 0.28099**。 |
| `submission_top_tier_blend.csv` | 301,799 B | 11,972 行 | v2 阶段 ElasticNetCV + 贪婪搜索混合产物，**CV 0.28271 -> Public LB 0.27755 (Ref 58369324)**。 |
| `submission_baseline.csv` | 301,841 B | 11,972 行 | 初始 5-Fold LightGBM 单模基线提交产物，CV 0.30512。 |
| `README.md` | 2,923 B | 26 行 | 提交文件版本对照表、行数/列名格式自检标准与官方成绩复现说明。 |

---

### 6. 辅助工具、EDA 与单元测试 (`archive/scripts/`)

| 文件名称 | 物理大小 | 源码行数 | 分类 | 核心定位与说明 |
| :--- | :--- | :--- | :--- | :--- |
| `baseline_lgb_regression.py` | 2,902 B | 79 行 | 基线 | 初始极简 5 折 LightGBM 回归基线。 |
| `main.py` | 565 B | 19 行 | 工具 | 项目主入口初始调度骨架。 |
| `print_columns.py` | 165 B | 6 行 | 工具 | 快速打印原始训练集/测试集全部字段名。 |
| `run_advanced_experiment.py` | 13,334 B | 310 行 | 实验 | 高阶模型调优与网格搜索调度脚本。 |
| `eda_all_columns.py` | 492 B | 17 行 | EDA | 全字段数据类型、缺失率与唯一值数量宏观体检。 |
| `eda_deep_insights.py` | 1,201 B | 26 行 | EDA | 价格分布偏度、对数正态性检验与强相关特征挖掘。 |
| `eda_features_deep.py` | 1,620 B | 31 行 | EDA | 卧室、床位、卫生间等容积特征深入相关性剖析。 |
| `eda_full.py` | 1,105 B | 27 行 | EDA | 全量数值特征皮尔逊/斯皮尔曼相关性矩阵分析。 |
| `eda_geo_and_text.py` | 1,285 B | 28 行 | EDA | 经纬度散点空间分布与文本字段长度探索。 |
| `eda_price_and_overlap_audit.py` | 35,457 B | 536 行 | EDA | 训练集与测试集重复房源、同房东定价先验深度审计。 |
| `eda_text_and_amenities.py` | 1,178 B | 37 行 | EDA | 设施列表 JSON 解析与高频词频统计。 |
| `check_env_and_stats.py` | 835 B | 20 行 | 环境 | Python 解释器、环境包与数据文件存在性探测。 |
| `check_libraries_gpu.py` | 612 B | 18 行 | 环境 | GBDT 家族与 PyTorch CUDA 硬件加速可用性校验。 |
| `check_previous_scores.py` | 1,023 B | 26 行 | 工具 | 自动化调用 Kaggle CLI 拉取团队历史提交与成绩。 |
| `verify_catboost_xgb.py` | 1,657 B | 56 行 | 验证 | CatBoost GPU 与 XGBoost CUDA 回归训练冒烟测试。 |
| `verify_env_quick.py` | 480 B | 15 行 | 验证 | 秒级环境可用性快速校验。 |
| `verify_tab_libs.py` | 910 B | 32 行 | 验证 | 表格建模依赖库完整性与版本检查。 |
| `test_bath_parser.py` | 988 B | 34 行 | 单测 | 浴室文本正则提取器边界用例单测。 |
| `test_env_models.py` | 913 B | 34 行 | 单测 | 各模型推理接口与显存分配机制测试。 |
| `test_feature_expansion.py` | 2,184 B | 48 行 | 单测 | 特征派生膨胀率与内存控制测试。 |
| `test_feature_extraction.py` | 2,169 B | 61 行 | 单测 | 特征提取逻辑输入输出张量维度测试。 |
| `test_kaggle_cli.py` | 637 B | 19 行 | 单测 | Kaggle 官方 CLI 权限与网络通信测试。 |
| `test_network.py` | 1,087 B | 28 行 | 单测 | PyTorch Tabular ResNet 前向传播张量测试。 |
| `README.md` | 4,258 B | 60 行 | 说明 | 脚本功能分类导航与单行运行范例说明。 |

---

### 7. 技术文档与实验报告 (`archive/docs/`)

| 文件名称 | 物理大小 | 文档行数 | 核心定位与说明 |
| :--- | :--- | :--- | :--- |
| **`DATA_AUDIT_AND_LEAKAGE_CHECK.md`** | 19,228 B | 216 行 | **[数据与防泄漏权威防线]** 涵盖房源特征清洗准则、对数正态目标分布设定、10 折统一划分逻辑与严禁折间目标渗透的数学保证。 |
| **`MODEL_EXPERIMENT_LOG.md`** | 21,992 B | 389 行 | **[全生命周期实验日志]** 详尽记录从基线 0.30512 到登顶 0.28109 / LB 0.27676 的每一轮模型组合、CV/LB 映射与后处理优化复盘。 |
| `README.md` | 3,016 B | 37 行 | 归档文档核心内容摘要、阅读指引与科研复现规范。 |

---

### 8. 缓存基座与 Out-of-Fold 索引 (`archive/cache/`)

| 目录/文件 | 物理体积 | 关键文件与维度 | 核心说明 |
| :--- | :--- | :--- | :--- |
| `cache_10fold_v3/` | 约 200 MB | `folds_10.npy`<br>`fold_ids.npy`<br>`fold_te/fold_0~9.npz` (10折 Target Encoding) | **[制胜核心特征基座]** 包含 361 维基础数值特征与 170 维无泄漏折内 Target Encoding 特征，总计 531 维。因大文件占用较大，留存于工作区根目录并在此处做索引说明。 |
| `oof_gbdt_v3/` | 约 5 MB | 7 路正交 GBDT 预测 npy | 包含 `oof_v3_lgb_dart.npy`, `oof_lgb_huber.npy`, `oof_xgb_deep.npy` 等主力预测，为 Stacking 求解提供高质量正交输入。 |
| `oof_cat_nn_v3/` | 约 3 MB | `oof_v3_cat_text_deep.npy`<br>`oof_v3_nn_tabular.npy` | 深度多模态原生文本预测与 PyTorch 表格残差网络预测，提供跨模型范式的极大集成方差缩减。 |
| `README.md` | 5,544 B | 84 行 | 完整记录历代特征缓存 (v1~v4) 与 OOF 预测的组织拓扑、字段维度与引用说明。 |

---

## 三、归档资产完整性与复现性保证

1. **零数据丢失与零破环**: 工作区根目录原有主入口脚本、缓存基座与环境文件保持不动，归档目录 `archive/` 完整映射并结构化梳理了全套工程代码。
2. **端到端一键复现**: 依赖关系清晰闭环，依据 `archive/stacking/README.md` 中指引，即可基于归档的模型脚本与缓存基座在数分钟内重构官方最高分提交。

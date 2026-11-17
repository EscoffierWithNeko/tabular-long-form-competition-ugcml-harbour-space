# 探索性数据分析 (EDA)、环境验证与工具脚本归档说明 (Scripts Archive)

本目录归档了在项目探索、环境准备、数据审计及单元测试阶段编写的全部辅助与验证脚本。这些脚本构成了整个竞赛项目严谨、可复现、无泄漏建模工程的技术基石。

---

## 一、脚本功能分类清单

### 1. 探索性数据分析 (EDA) 与特征探索
| 脚本名称 | 核心功能与分析要点 |
| :--- | :--- |
| **`eda_price_and_overlap_audit.py`** | **最核心数据审计脚本**：系统性探查训练集与测试集在空间微网格、同房东 (`host_id`)、同房型上的定价重叠与孪生特征先验，为后期的孪生微环境复合键 (`twin_3tuple`, `twin_4tuple`, `twin_5tuple`) 提供了最关键的实验实证支撑。 |
| **`eda_all_columns.py`** | 快速扫描全量原始字段的数据类型、缺失率、基数统计与样本初览。 |
| **`eda_deep_insights.py`** | 深入剖析目标变量 `price` 的右偏分布特征，确立 $\log(1 + \text{price})$ 目标变换的必要性；评估离群点对评估指标 RMSLE 的影响。 |
| **`eda_features_deep.py`** | 深入分析 `bathrooms`, `bedrooms`, `beds`, `accommodates` 之间的非线性容量关系与人均比率。 |
| **`eda_full.py`** | 跨字段多变量相关性热力图与多重共线性 (VIF) 初筛。 |
| **`eda_geo_and_text.py`** | 地理坐标 (`latitude`, `longitude`) 空间分布离散化与标题/描述长度统计。 |
| **`eda_text_and_amenities.py`** | 针对 `amenities` 设施 JSON 文本进行分词清洗与高频词频统计分析。 |

### 2. 初始基准模型 (Baselines) 与调度入口
| 脚本名称 | 核心功能与分析要点 |
| :--- | :--- |
| **`baseline_lgb_regression.py`** | 初代标准 5 折 LightGBM 回归基线，完成第一个端到端 `train.csv` -> 预测 -> `submission_baseline.csv` 闭环。 |
| **`main.py`** | 早期主执行入口骨架。 |
| **`run_advanced_experiment.py`** | 早期端到端多模型实验流程串联与超参数网格搜索调度脚本。 |
| **`print_columns.py`** | 格式化打印原始特征列名字典与索引。 |

### 3. 环境探查与硬件加速诊断
| 脚本名称 | 核心功能与分析要点 |
| :--- | :--- |
| **`check_env_and_stats.py`** | 探测 Python 解释器版本、内存容量、物理磁盘余量以及关键数据文件完整性。 |
| **`check_libraries_gpu.py`** | 全面诊断 LightGBM、XGBoost、CatBoost 与 PyTorch 的 CUDA / GPU 硬件加速可用性与驱动适配情况。 |
| **`check_previous_scores.py`** | 自动化调用 Kaggle API 查询竞赛官方历史提交记录与 Public Leaderboard 最高得分。 |
| **`verify_catboost_xgb.py`** | 针对 CatBoost GPU 与 XGBoost `device='cuda'` 进行小批量显存吞吐与稳定性冒烟测试。 |
| **`verify_env_quick.py`** | 秒级执行的环境冒烟探查工具，快速检查核心模块导入。 |
| **`verify_tab_libs.py`** | 校验 Optuna、Scikit-learn、Category Encoders 等表格增强类库的版本兼容性。 |

### 4. 单元测试与特征提取防泄漏单测 (Unit Tests)
| 脚本名称 | 核心功能与分析要点 |
| :--- | :--- |
| **`test_bath_parser.py`** | 单元测试浴室文本解析逻辑，覆盖 `half-bath`, `shared`, `private bath`, 纯数字与缺失值等所有极端边界用例。 |
| **`test_feature_expansion.py`** | 验证特征工程矩阵扩展后的列名唯一性与无非法高维稀疏爆炸。 |
| **`test_feature_extraction.py`** | 严格验证特征提取函数在训练集与测试集上的输出维度一致性与无标签泄漏。 |
| **`test_network.py`** | 单元测试 PyTorch Tabular ResNet 的前向传播张量形状、残差连接正确性与梯度反向传播。 |
| **`test_env_models.py`** | 跨模型推理单测，确保统一输出格式。 |
| **`test_kaggle_cli.py`** | 验证系统底层 Kaggle CLI 命令的可调用性与身份认证 token 状态。 |

---

## 二、使用示例

运行浴室文本解析单测：
```powershell
F:\Anaconda\envs\pytorch\python.exe test_bath_parser.py
```

运行环境加速能力自检：
```powershell
F:\Anaconda\envs\pytorch\python.exe check_libraries_gpu.py
```

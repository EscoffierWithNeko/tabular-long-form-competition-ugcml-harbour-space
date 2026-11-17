# 提交产物归档与官方得分对照表 (Submissions Archive)

本目录归档了在 Kaggle `tabular-long-form-competition-ugcml-harbour-space` 竞赛过程中产出的关键预测提交文件 (`*.csv`)。所有文件均经过格式完整性校验，包含且仅包含 `id` 和 `price` 两列，总数据行数精确为 **11971** 行（加上表头共 **11972** 行），严格对齐官方 `test.csv` 的 ID 序列。

---

## 一、历代核心提交对照表

| 提交文件名称 | 产生脚本 / 算法版本 | 10-Fold 验证集 CV RMSLE | 官方榜单 Public LB 得分 | 官方提交 Ref / 状态 | 特征与算法核心亮点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`submission_stacking_v3_rmsle_0.28109.csv`** | `stack_and_submit_gbdt_textzoo.py` (v3 凸优化) | **0.281092** | **`0.27676`** (全场第一梯队) | Ref: `58373885`<br>Status: `COMPLETE` | **本项目最佳制胜提交**：基于 v3 孪生微环境复合键、高阶容纳密度，融合 DART + Huber + XGB GPU Hist + 深度原生文本 CatBoost，采用贪婪前向特征筛选 + SLSQP 约束凸优化加权解算。 |
| **`submission_stacking_v3_rmsle_0.28099.csv`** | `stack_and_submit_gbdt_textzoo.py` (v3 偏差校准) | **0.280994** | 阶段备选保留 | 本地保存备选 | 在 SLSQP 凸优化的基础上叠加折内 Log-Bias 零均值漂移补偿与硬边界截断，CV 达到历史极致 0.28099。 |
| **`submission_top_tier_blend.csv`** | `train_blended_top_tier.py` (v2 异构加权) | 0.282713 | **0.27755** | Ref: `58369324`<br>Status: `COMPLETE` | 聚合 v2 空间微网格特征基座、深度 ResNet 与多路树模型，采用 ElasticNetCV 稀疏正则加权，实现破纪录跃迁。 |
| **`submission_baseline.csv`** | `baseline_lgb_regression.py` (初始基线) | 0.305120 | ~0.30150 | 初始基线校验 | 纯基础表格数值统计与类别高频编码，5 折 LightGBM 基础回归基准。 |
| **`submission.csv`** | 根目录当前激活发布文件 | 0.281092 | 0.27676 | 生产软链接对应 | 镜像指向最新官方验证最优产物，供各自动化下游任务与评测调用。 |

---

## 二、提交文件规范与校验指标

所有归档的预测文件必须满足以下铁律，任一违背均视为无效产物：
1. **文件格式**: UTF-8 编码 CSV 文件，无 BOM 标记，无索引行（除首行表头外）。
2. **字段名称**: 严格为 `id,price`（小写字母，逗号分隔）。
3. **样本行数**: 训练集样本数 $N_{\text{train}} = 47293$，测试集预测样本数 $N_{\text{test}} = 11971$。文件总行数包含表头必为 11972 行。
4. **缺失值与异常值**: 严禁包含 `NaN`、`Null`、`Inf`；预测价格必须在物理合理区间内（已统一采用赛题截断约束 $[20.0, 2000.0]$）。
5. **ID 顺序校验**: 所有预测结果的 ID 序列与官方提供的原始 `test.csv` 保持 100% 逐行一一对应。

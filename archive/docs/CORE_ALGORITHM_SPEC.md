# Kaggle 房源定价赛题制胜算法技术规格说明书 (v3 Twin-MicroEnv SLSQP Stacking)

> **算法名称**: 孪生微环境特征工程与严格凸约束 SLSQP 堆叠融合算法  
> **代码标识**: `v3_twin_microenv_slsqp_stacking`  
> **官方最优成绩**: Public LB: **0.27676** | Private LB: **0.28160** (Ref: 58373885)  
> **归属工程管线**: `fe_pipeline_v3.py` + `train_gbdt_v3.py` + `train_catboost_nn_zoo_v3.py` + `stack_and_submit_gbdt_textzoo.py`  
> **编制日期**: 2026-11-15  

---

## 一、算法总体架构与流程图 (System Architecture)

制胜算法系统由三大核心阶段构成：
1. **严格折内无泄漏 10-Fold 孪生微环境特征基座 (531 维)**；
2. **异构多目标 9 路基元模型库 (GBDT 多样性 + CatBoost 原生深度文本 + PyTorch Tabular ResNet)**；
3. **Level-2 SLSQP 约束凸组合优化引擎 + 折内 Log-Bias 质检校准与后处理截断**。

```
[原始数据: train.csv (47,293) / test.csv (11,971)]
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段一：特征工程与基座构建 (fe_pipeline_v3.py)               │
│ - 数据清洗与预处理 (硬边界截断感知 [20, 2000])               │
│ - 全局微环境实体复合键构建 (twin_3tuple, twin_4tuple, 5tuple)│
│ - 基础表格特征矩阵 (361 维)                                  │
│ - 10-Fold 折内高阶 OOF 特征抽取 (170 维):                    │
│   * 孪生复合键多平滑度 Target Encoding (m=0.2, 1.0, 2.0, 10) │
│   * 核心实体 Unit TE, Residual TE 与分位数统计              │
│   * Text Zoo 16 维 TF-IDF + Ridge 回归折内 OOF 语义表征      │
│ 产出: 严格对齐 531 维特征矩阵 (cache_10fold_v3/)            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段二：9 路高多样性异构基元模型库训练 (10-Fold CV)          │
│ 1. v3_xgb_main      (XGBoost GPU Hist, depth=7)   CV 0.28389│
│ 2. v3_lgb_et        (LGBM 极端随机化 ExtraTrees)  CV 0.28421│
│ 3. v3_lgb_huber     (LGBM Huber 鲁棒损失)         CV 0.28423│
│ 4. v3_lgb_main      (LGBM 标准主力回归器)         CV 0.28503│
│ 5. v3_xgb_deep      (XGBoost GPU Hist 深树 depth=9)CV 0.28507│
│ 6. v3_cat_text_deep (CatBoost 原生深度文本)       CV 0.28866│
│ 7. v3_lgb_res       (LGBM 级联残差回归器)         CV 0.28922│
│ 8. v3_lgb_dart      (LGBM DART 随机树丢弃)        CV 0.29085│
│ 9. v3_nn_tabular    (PyTorch Tabular ResNet)      CV 0.29729│
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段三：Level-2 堆叠优化与后处理门禁 (stack_and_submit_...) │
│ - 严格跨代模型池质检与分布审计                              │
│ - 折内前向贪婪模型优选 (Forward Stepwise Selection, K=8)   │
│ - SLSQP 严格凸优化求解: min ||y - Xw||^2 s.t. w>=0, sum(w)=1│
│ - 折内无偏 Log-Bias 动态校准 (|bias| <= 0.03 熔断保护)       │
│ - 硬边界后处理截断 np.clip(pred, 20.0, 2000.0)              │
│ - 四维统计指标门禁校验 (均值区间 [195, 235], NaN/Inf 检查)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
        [最终提交: submission.csv -> 官方 LB: 0.27676]
```

---

## 二、数据预处理与目标定义

1. **赛题目标转换**:
   - 原始目标为房源价格 `price`（美元），官方评估指标为 RMSLE；
   - 模型训练全流程在对数空间进行：$y_{\text{train}} = \log(1 + \text{price}) = \text{log1p}(\text{price})$；
   - 目标统计特性：`n = 47293`, 均值 `5.1508`, 对应原空间均值 `227.13 美元`，中位数 `175.00 美元`。
2. **硬边界截断感知与数据清洗**:
   - 数据审计证实主办方对测试集与训练集价格均施加了 $[20.0, 2000.0]$ 的硬截断；
   - 任何小于 20.0 的预测值或大于 2000.0 的预测值均属于无效长尾外推；
   - 缺失值处理：数值列缺失补充中位数与缺失指示标记（`_isna`）；分类列缺失填充特殊标记 `'__MISSING__'`；文本缺失填充空字符串 `''`。

---

## 三、531 维特征工程基座详解 (cache_10fold_v3)

特征工程基座共包含 **531 维** 完全严格折内生成的无泄漏特征：

### 1. 基础表格特征矩阵 (361 维)
- **物理与规模特征**:
  - `accommodates`, `bedrooms`, `bathrooms`, `beds` 等基础容量；
  - 交互比例：`accommodates_per_bedroom`, `beds_per_bedroom`, `bathrooms_per_bedroom`, `bathrooms_per_guest`；
  - 空间容量冗余度指标与独享/合住布尔标记。
- **房东历史与信誉特征**:
  - `hosts_time_as_user_years`, `hosts_time_as_host_years`；
  - `host_response_rate_num`, `host_acceptance_rate_num`；
  - 房东名下房源数 `host_listings_count` 及其分段等级。
- **空间地理特征**:
  - `latitude`, `longitude` 经纬度绝对坐标；
  - 经纬度主成分投影与欧氏几何变换；
  - 微地理网格实体 `geo_cell`（经纬度按高精度网格切片编码，覆盖率达 98.30%）。
- **文本统计与元数据**:
  - `name_len`, `description_len`, `amenities_count` 等长度与数量统计；
  - 关键设施布尔标记（Wifi, Kitchen, Air conditioning, Pool, Hot tub, Free parking 等 30+ 维）。

### 2. 孪生微环境复合键与高阶 OOF 特征 (170 维)
这是本方案击穿高分壁垒的关键创新：
- **孪生微环境复合键定义**:
  - `twin_3tuple`: `host_id + '__' + geo_cell + '__' + room_type`
  - `twin_4tuple`: `host_id + '__' + geo_cell + '__' + room_type + '__' + accommodates`
  - `twin_5tuple`: `host_id + '__' + geo_cell + '__' + room_type + '__' + accommodates + '__' + bedrooms`
  - **业务机理**: 在同一房东、相同地理极小微区、相同房间类型和容纳规模下的房源，属于典型的“孪生房源”（如同一栋公寓楼内的多套同户型短租房），其历史定价具有极高置信度的锚定价值。
- **折内多平滑度 Target Encoding (Multi-Smooth TE)**:
  - 针对 `twin_3tuple`, `twin_4tuple`, `twin_5tuple`, `host_id`, `geo_cell`, `neighbourhood`, `zipcode`:
  - 采用平滑经验贝叶斯公式：
    $$\hat{\mu}_g = \frac{\sum_{i \in g} y_i + m \cdot \mu_{\text{global}}}{|g| + m}$$
  - 平滑参数梯度：$m \in [0.2, 1.0, 2.0, 10.0]$。对于孪生复合键，极小的平滑度（$m=0.2, 1.0$）能够直接近乎无损地还原同房东同户型的微环境精准溢价；
  - 严格在 10 折每折的训练集内部进行 5 折内层循环编码，验证集与测试集直接基于训练集先验做映射，未见过的微环境退化为全局先验。
- **Unit TE 与 Residual TE**:
  - `unit_te`: 以每人均价格 $\log(1 + \text{price} / \text{accommodates})$ 为目标计算折内 TE；
  - `residual_te`: 建立轻量级基线回归器，计算折内预测残差并基于空间网格计算局部残差偏置。
- **Text Zoo 16 维 Ridge OOF 语义表征**:
  - 在每折训练集中，分别提取 `name`、`description` 和拼接全文本的 TF-IDF 矩阵（含词级 `(1, 2)` ngram 与字符级 `(3, 5)` ngram）；
  - 训练正则化无截距 Ridge 回归器（$\alpha \in [1.5, 2.0, 6.0]$）；
  - 输出 16 维折内 OOF 对数价格先验预测特征（如 `full_word_ridge_a1.5`, `name_word_ridge_a2.0` 等），其单维特征与真实标签相关系数高达 0.85+。

---

## 四、9 路基元模型库规格清单 (Level-1 Model Zoo)

模型库由 9 路具备高度算法多样性、结构差异与不同损失函数的模型构成，所有模型均在 `cache_10fold_v3/` 统一特征和切分上独立训练：

| 模型标识 | 算法引擎 | 核心结构与超参数设定 | 损失函数 / 目标 | 单模 10-Fold CV | 角色与多样性来源 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`v3_xgb_main`** | XGBoost (GPU Hist) | `max_depth=7`, `lr=0.03`, `colsample=0.7`, `subsample=0.8`, `min_child_weight=5`, 树数 4000 | RMSE | **0.28389** | 单模第一主力，梯度直方图精确切分 |
| **`v3_lgb_et`** | LightGBM | `extra_trees=True`, `num_leaves=95`, `lr=0.025`, `colsample=0.5`, `subsample=0.7`, 树数 5000 | RMSE | **0.28421** | 极端随机化树，显著降低特征共线性过拟合 |
| **`v3_lgb_huber`** | LightGBM | `objective='huber'`, `alpha=0.8`, `num_leaves=63`, `lr=0.03`, `colsample=0.7`, `subsample=0.8` | Huber Loss | **0.28423** | 鲁棒损失函数，有效抵御极端高端房源扰动 |
| **`v3_lgb_main`** | LightGBM | `num_leaves=63`, `lr=0.03`, `colsample=0.7`, `subsample=0.8`, 树数 5000 | RMSE | **0.28503** | 标准主力回归器，学习全局稳定趋势 |
| **`v3_xgb_deep`** | XGBoost (GPU Hist) | `max_depth=9`, `lr=0.02`, `colsample=0.6`, `subsample=0.75`, `min_child_weight=10` | RMSE | **0.28507** | 深度复杂特征交互探索 |
| **`v3_cat_text_deep`**| CatBoost (GPU/CPU) | `depth=7`, `lr=0.035`, `text_features=['desc_clean','name_clean']`, `l2_leaf_reg=3.0`, 迭代 3500 | RMSE | **0.28866** | 原生深度文本语义提取 + 类别高阶字典 |
| **`v3_lgb_res`** | LightGBM | 基于基线残差级联训练，`num_leaves=47`, `lr=0.025` | Residual RMSE | **0.28922** | 专注于拟合前序模型未学到的高难度残差 |
| **`v3_lgb_dart`** | LightGBM (DART) | `boosting='dart'`, `drop_rate=0.10`, `skip_drop=0.50`, `num_leaves=63`, `lr=0.03` | RMSE | **0.29085** | 随机树丢弃正则化，防止局部过拟合 |
| **`v3_nn_tabular`** | PyTorch ResNet | 4 层残差块（Dense-BN-ReLU-Dropout(0.2)），维度 256->128->64，AdamW 优化器，余弦退火学习率 | Smooth L1 | **0.29729** | 纯连续表征与全连接深度交互，异构拓扑 |

---

## 五、Level-2 SLSQP 约束凸优化 Stacking 引擎

为了在融合多模型的同时彻底杜绝 P5 阶段遇到的外推漂移，本算法设计了严密的两阶段融合策略：

### 1. 折内前向贪婪模型优选 (Forward Stepwise Selection)
- **输入**: 经质检合规的所有候选 OOF 预测池 $\mathcal{M}$，真实目标 $y_{\text{train}}$；
- **初始化**: 选择单模验证集最优模型 $m_1 = \text{v3\_xgb\_main}$（RMSLE = 0.283888），已选集合 $S = \{m_1\}$；
- **贪婪迭代**: 
  - 在第 $k$ 步，遍历所有未选模型 $m \in \mathcal{M} \setminus S$；
  - 构造候选特征矩阵 $X = [\hat{y}_{s_1}, \dots, \hat{y}_{s_{k-1}}, \hat{y}_m]$；
  - 运行 SLSQP 求解约束凸组合权重；
  - 选出使 10-Fold RMSLE 降幅最大的候选模型加入 $S$；
  - 直至达到最大容量限制（$K=8$）或收益小于阈值。
- **优选路径与最终选定子集**:
  1. 起始: `v3_xgb_main` (0.283888)
  2. 选入: `v1_xgb` (改善后 0.282004, $\Delta = -0.001884$)
  3. 选入: `v3_lgb_et` (改善后 0.281339, $\Delta = -0.000665$)
  4. 选入: `v3_nn_tabular` (改善后 0.281144, $\Delta = -0.000195$)
  5. 选入: `v3_lgb_huber` (改善后 0.281039, $\Delta = -0.000106$)
  6. 选入: `v1_lgb_et` (改善后 0.280965, $\Delta = -0.000073$)
  7. 选入: `v1_lgb_std` (改善后 0.280942, $\Delta = -0.000023$)
  8. 选入: `v3_xgb_deep` (改善后 0.280934, $\Delta = -0.000009$)

### 2. SLSQP 约束凸优化数学表述
对于选出的 8 个模型，求解以下二次规划凸优化问题：
$$\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N \left( y_i - \sum_{j=1}^K w_j \hat{y}_{ij} \right)^2$$
$$\text{subject to} \quad \sum_{j=1}^K w_j = 1 \quad \text{and} \quad w_j \ge 0, \quad \forall j \in \{1, \dots, K\}$$

通过 `scipy.optimize.minimize` (SLSQP 求解器)，初始权重设为 $w_0 = [1/K, \dots, 1/K]$，容差设定为 `tol=1e-9`。

**全局最优解权重分布**:
- `v3_xgb_main`: **0.2444** (24.44% - 主力核心)
- `v3_lgb_et`: **0.1711** (17.11% - 极端随机化互补)
- `v3_lgb_huber`: **0.1523** (15.23% - 鲁棒损失抗异常)
- `v1_xgb`: **0.1495** (14.95% - 跨代基座差异)
- `v3_nn_tabular`: **0.0856** (8.56% - 深度网络异构先验)
- `v1_lgb_et`: **0.0818** (8.18% - 历史平滑树)
- `v1_lgb_std`: **0.0675** (6.75% - 基础树基准)
- `v3_xgb_deep`: **0.0480** (4.80% - 深度交互微调)

所有权重严格非负且和精确为 1.0000，彻底消除外推灾难。

---

## 六、折内 Log-Bias 动态校准与安全熔断

在对数空间中，模型的整体期望微偏会对最终指标造成系统性影响。为此设计严格折内无偏校准算法：
1. **折内残差偏置估计**:
   对于第 $k$ 折，基于该折训练集与验证集，计算对数均值残差：
   $$b_k = \frac{1}{|V_k|} \sum_{i \in V_k} (y_i - \hat{y}_i^{\text{convex}})$$
2. **安全漂移熔断**:
   若 $|b_k| > 0.03$，判定该折存在异常漂移，直接强制熔断将偏置置零：$b_k = 0.0$；
3. **加权微调应用**:
   - 验证集预测: $\hat{y}_{\text{cal}, i} = \hat{y}_i + b_k$
   - 测试集预测: $\hat{y}_{\text{test\_cal}} = \hat{y}_{\text{test}} + \frac{1}{K_{\text{folds}}} \sum_{k=1}^{K_{\text{folds}}} b_k$

---

## 七、后处理流程与输出质量检验门禁 (Post-Processing & QA Gates)

生成的最终预测必须经过以下确定性后处理管线：

1. **指数反变换还原到真实价格空间**:
   $$\hat{P}_{\text{test}} = \exp(\hat{y}_{\text{test\_cal}}) - 1 = \text{expm1}(\hat{y}_{\text{test\_cal}})$$
2. **赛题硬边界截断**:
   $$\hat{P}_{\text{final}} = \text{np.clip}(\hat{P}_{\text{test}}, 20.0, 2000.0)$$
3. **出厂全自动四维质检门禁 (Quality Assurance Check)**:
   - **Check 1: 格式与维度**: 预测样本数必须严格对齐 `11971` 行，列名为 `id` 与 `price`；
   - **Check 2: 数值合法性**: 严禁出现任何 `NaN`、`Inf` 或负数；
   - **Check 3: 边界合法性**: $\min(\hat{P}) \ge 20.0$ 且 $\max(\hat{P}) \le 2000.0$；
   - **Check 4: 宏观统计学区间质检**:
     - 预测价格均值必须处于 $[195.0, 235.0]$ 范围（实际实测: **213.87 美元**，高度贴合真实世界 227.13 美元）；
     - 对数均值必须处于 $[5.05, 5.25]$ 范围（实际实测: **5.14**）。

任何一项检查未通过，流程将直接抛出异常中断，确保交付物 100% 具备工业级稳健性。

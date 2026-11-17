# 融合与 Stacking 提交管线归档与机制说明 (Ensemble & Stacking)

本目录归档了本项目中最核心的模型集成、Level-2 Stacking 融合与自动化官方提交管线。在本项目中，我们通过**跨代模型正交组合、前向逐步特征选择 (Forward Stepwise Selection)、SLSQP 约束凸优化与严格后处理校准**，成功将验证集 CV RMSLE 从单模 0.286+ 击穿至 **0.28109**，并在 Kaggle 官方榜单斩获 **Public LB 0.27676** 的第一梯队战绩。

---

## 一、脚本清单与定位

| 脚本文件 | 核心职责 | 依赖模型池 | 评测元学习器 | 最佳验证成绩 (10-Fold CV) |
| :--- | :--- | :--- | :--- | :--- |
| **`stack_and_submit_gbdt_textzoo.py`** (制胜核心) | 跨代多模态全模型池 Level-2 Stacking 冲榜与官方提交管线 | `oof_gbdt_v3/`<br>`oof_cat_nn_v3/`<br>`oof_gbdt_v2/`<br>`text_zoo_oof/` | 1. SLSQP 约束凸组合 ($\sum w_i = 1, w_i \ge 0$)<br>2. NNLS (非负最小二乘)<br>3. RidgeCV (L2 正则)<br>4. ElasticNetCV (稀疏混合)<br>5. Meta-LightGBM (极浅非线性树) | **0.28109** (后处理校准后 0.28099) -> **Public LB 0.27676** |
| **`stack_and_submit_v2.py`** | v2 阶段 10-Fold 线性回归与基准加权融合脚本 | `oof_gbdt_v2/`<br>`oof_cat_nn_v2/` | RidgeCV、ElasticNetCV、加权平均 | 0.282713 -> **Public LB 0.27755** |

---

## 二、Stacking 架构与数学原理

```
[原始数据 Train/Test]
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
[cache_10fold_v3 特征基座]                         [原生文本/高基数标记]
        │                                                 │
   ┌────┴────────────────────────┬───────────────┐        │
   ▼                             ▼               ▼        ▼
[v3_lgb_main / dart / huber]  [v3_xgb_deep]  [v3_nn_resnet] [v3_cat_text_deep]
   │                             │               │        │
   └─────────────┬───────────────┴───────────────┴────────┘
                 ▼
     [Level-1 10-Fold OOF 预测矩阵 (47293, M)]
                 │
                 ▼
     [贪婪前向特征优选 (Forward Stepwise Selection)]
     (筛选出误差相关性最低、正交信息量最丰富的一组模型子集)
                 │
                 ▼
     [Level-2 SLSQP 约束凸优化求解]
     min_{w}  RMSLE(y_true, \sum w_i * pred_i)
     s.t.     \sum w_i = 1,  w_i >= 0
                 │
                 ▼
     [折内 Log-Bias 零均值校准 + [20, 2000] 边界截断]
                 │
                 ▼
     [生成最终提交文件并验证格式 (11971 行)]
```

### 1. 贪婪前向逐步优选 (Forward Stepwise Selection)
直接将所有候选模型全部扔进元学习器容易因为共线性导致过拟合与权重分散。脚本中内置了贪婪前向选择算法：
* 第 1 步：挑选单模中 CV RMSLE 最优的基础模型作为基准核心。
* 第 2 步：遍历所有剩余候选模型，每次尝试加入一个新模型并利用优化器解算最佳加权得分。
* 第 3 步：若引入该模型能带来超过容差阈值的净收益，则正式纳管该模型；否则剪枝舍弃。
* 结果证明，由 **DART 随机丢弃树 + Huber 鲁棒损失树 + 深度原生文本 CatBoost + 表格 ResNet** 组成的子集取得了最高的互补增益。

### 2. SLSQP 约束凸优化求解 (Sequential Least Squares Programming)
为了杜绝传统无约束线性回归中可能产生的负权重或极大正负权重对冲（这种对冲在测试集遇到微小分布漂移时会造成灾难性泛化恶化），我们采用 SLSQP 凸优化求解器：
$$ \min_{\mathbf{w}} \sqrt{\frac{1}{N}\sum_{k=1}^N \left(\sum_{i=1}^M w_i \hat{y}_{k,i} - y_k\right)^2} $$
$$\text{s.t.} \quad \sum_{i=1}^M w_i = 1, \quad 0 \le w_i \le 1, \quad \forall i=1,\dots,M$$
由于目标函数在凸多面体上具有良好的凸性，SLSQP 能在几百毫秒内收敛到全局数值最优解。

### 3. 折内 Log-Bias 漂移校准 (In-Fold Zero Bias Alignment)
在对数空间中，模型的系统性预测均值往往存在细微偏差：
$$\Delta_{\text{bias}} = \frac{1}{|V_{\text{fold}}|} \sum_{j \in V_{\text{fold}}} (\hat{y}_j^{\text{stack}} - y_j)$$
对每折预测值扣除此漂移项 $\hat{y}_{\text{cal}} = \hat{y}^{\text{stack}} - \Delta_{\text{bias}}$，并将 10 折平均偏差补偿应用到测试集预测中。实测此项工程微调为最终 CV 稳定带来约 `0.00010` 的显著提升。

### 4. 业务场景硬边界截断
依据赛题数据探索 (EDA) 得出的真实价格分布区间，强制执行：
$$\text{Price}_{\text{final}} = \text{clip}(\exp(\hat{y}_{\text{log1p}}) - 1, \; 20.0, \; 2000.0)$$
剔除任何模型外推可能造成的非正常负值或天文数字离群预测。

---

## 三、融合脚本运行与官方自动化提交流程

在配置好 Kaggle API 凭证的环境下，运行以下命令即可完成端到端全自动评估、生成预测与冲榜提交：

```powershell
# 执行全模型池 10-Fold Stacking 评测并在超越最高分时自动提榜
F:\Anaconda\envs\pytorch\python.exe stack_and_submit_gbdt_textzoo.py
```
管线会自动执行严格的自检流程：
1. 校验 OOF 预测与 Target 对应行数与缺失值；
2. 校验测试集生成的行数（精确 11971 行）与 ID 序列是否与 `test.csv` 一字不差；
3. 输出各元模型（SLSQP、NNLS、RidgeCV、ElasticNetCV）的 10 折详细指标排行榜；
4. 归档最优提交文件并实时调用 `kaggle competitions submit` 提交，抓取官方即时 Public LB 成绩。

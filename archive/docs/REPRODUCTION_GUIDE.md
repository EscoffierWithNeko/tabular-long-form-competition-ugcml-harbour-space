# Kaggle tabular-long-form 房源定价赛题：终局方案端到端完整复现指南

> **赛题**: `tabular-long-form-competition-ugcml-harbour-space`  
> **目标版本**: P6 终局方案 (v3 Twin-MicroEnv SLSQP Stacking, Public LB: **0.27676**)  
> **适用受众**: 竞赛评审员、工程部署团队、学术复现者  
> **编制日期**: 2026-11-15  

---

## 一、软硬件环境依赖与基础设施要求

### 1. 硬件环境推荐配置
- **操作系统**: Windows 10/11 64-bit 或 Linux (Ubuntu 20.04/22.04 LTS)
- **CPU**: 8 核及以上（推荐 AMD Ryzen 7 / Intel Core i7 及以上）
- **内存 (RAM)**: $\ge 32\text{ GB}$（特征基座构建与折内多平滑度 Target Encoding 计算需暂存大型矩阵）
- **GPU**: NVIDIA RTX 3060 / 4070 / 4080 / 4090 或 A10/V100/A100（显存 $\ge 8\text{ GB}$，支持 CUDA 11.8+ / 12.x）
- **磁盘空间**: 至少保留 $20\text{ GB}$ 可用 SSD 空间（用于存放折内分块特征与 OOF 预测矩阵）

### 2. Python 运行环境与依赖包清单
本项目统一采用 Python 3.10 环境运行（工作区基准解释器路径为 `F:\Anaconda\envs\pytorch\python.exe`）。核心依赖库版本清单如下：

| 依赖库名称 | 推荐版本范围 | 用途说明 |
| :--- | :--- | :--- |
| **Python** | `3.10.x` | 运行基础环境 |
| **torch / torchvision** | `2.1.0+cu118` 或 `2.2.0+cu121` | Tabular ResNet 神经网络训练 (GPU 加速) |
| **lightgbm** | `4.1.0` ~ `4.3.0` | LightGBM 多样性树库 (ExtraTrees, Huber, DART) |
| **xgboost** | `2.0.0` ~ `2.0.3` | XGBoost GPU Hist 树训练 |
| **catboost** | `1.2.0` ~ `1.2.2` | CatBoost 原生深度文本与类别特征模型 |
| **scikit-learn** | `1.3.0` ~ `1.4.0` | KFold、TF-IDF 文本向量化、RidgeCV、指标评估 |
| **scipy** | `1.11.0` ~ `1.12.0` | SLSQP 二次规划凸优化求解器 |
| **pandas** | `2.1.0` ~ `2.2.0` | 高性能表格数据处理 |
| **numpy** | `1.24.0` ~ `1.26.0` | 矩阵张量数值计算 |
| **pyarrow** | `14.0.0+` | 高速 Parquet 格式读写与中间件缓存 |

---

## 二、数据文件准备与完整性核验

在开始执行复现流水线前，请确保工作区根目录已具备官方赛题数据文件，且校验信息完全一致：

```powershell
# 检查根目录下官方数据文件
F:\Anaconda\envs\pytorch\python.exe -c "
import os, hashlib
for fname, exp_sz in [('train.csv', 61934102), ('test.csv', 15542708)]:
    if not os.path.exists(fname):
        print(f'[错误] 缺失文件: {fname}')
    else:
        sz = os.path.getsize(fname)
        status = '✅ 对齐' if sz == exp_sz else f'⚠️ 大小不符 ({sz} vs {exp_sz})'
        print(f'{fname:12s} 大小: {sz:,} 字节 | {status}')
"
```

预期控制台输出：
```text
train.csv    大小: 61,934,102 字节 | ✅ 对齐
test.csv     大小: 15,542,708 字节 | ✅ 对齐
```

---

## 三、四步端到端复现流水线命令手册

复现流水线采用严格解耦设计，按顺序依次运行以下 4 个核心脚本即可：

```
[步骤 1: 特征工程基座生成] ──► [步骤 2: 7路 GBDT 训练] ──► [步骤 3: 深度文本与NN训练] ──► [步骤 4: Level-2 SLSQP 融合]
   fe_pipeline_v3.py           train_gbdt_v3.py           train_catboost_nn_zoo_v3.py      stack_and_submit_gbdt_textzoo.py
```

### 步骤 1: 孪生微环境 10 折特征基座构建 (`fe_pipeline_v3.py`)
- **功能**: 清洗数据，构建 `twin_3tuple`, `twin_4tuple`, `twin_5tuple` 实体复合键；在 10 折折内严密无泄漏生成 16 维 Text Zoo Ridge OOF 语义特征、核心实体多平滑度 Target Encoding 与微环境高阶特征，产出 531 维基准特征集。
- **运行命令**:
```powershell
F:\Anaconda\envs\pytorch\python.exe fe_pipeline_v3.py
```
- **耗时预估**: 约 4 ~ 6 分钟（取决于 CPU 多核性能）
- **核验标准**:
  - 生成目录: `cache_10fold_v3/`
  - 检查包含: `folds_10.npy`, `fold_ids.npy`, `y_log1p.npy`, `manifest_v3.json`, `READY_V3.flag`
  - 检查子目录: `cache_10fold_v3/fold_te/`（内含 10 个折内增量特征文件 `fold_0_te.parquet` ~ `fold_9_te.parquet` 及 `test_fold_te.parquet`）
  - 特征维度核验: `manifest_v3.json` 中 `"n_features": 531`

---

### 步骤 2: 7 路高多样性 GBDT 模型库训练 (`train_gbdt_v3.py`)
- **功能**: 基于 `cache_10fold_v3/`，训练包括 XGBoost GPU Hist (主力与深树)、LightGBM ExtraTrees、LightGBM Huber 损失、LightGBM 标准树、LightGBM 残差级联与 DART 树在内的 7 路 GBDT 模型。
- **运行命令**:
```powershell
F:\Anaconda\envs\pytorch\python.exe train_gbdt_v3.py
```
- **耗时预估**: 约 8 ~ 15 分钟（单张 RTX 4080/4090 约 8 分钟；RTX 3060 约 15 分钟）
- **核验标准**:
  - 生成目录: `oof_gbdt_v3/`
  - 产出文件清单（每路模型包含 OOF 预测与 Test 预测）：
    * `oof_xgb_main.npy` 与 `test_xgb_main.npy` (单模 RMSLE $\approx 0.28389$)
    * `oof_lgb_et.npy` 与 `test_lgb_et.npy` (单模 RMSLE $\approx 0.28421$)
    * `oof_lgb_huber.npy` 与 `test_lgb_huber.npy` (单模 RMSLE $\approx 0.28423$)
    * `oof_lgb_main.npy` 与 `test_lgb_main.npy` (单模 RMSLE $\approx 0.28503$)
    * `oof_xgb_deep.npy` 与 `test_xgb_deep.npy` (单模 RMSLE $\approx 0.28507$)
    * `oof_lgb_res.npy` 与 `test_lgb_res.npy` (单模 RMSLE $\approx 0.28922$)
    * `oof_lgb_dart.npy` 与 `test_lgb_dart.npy` (单模 RMSLE $\approx 0.29085$)
  - 数组维度: `oof_*.npy` 长度精确为 47293，`test_*.npy` 长度精确为 11971

### 步骤 3: 深度原生文本 CatBoost 与 Tabular ResNet 训练 (`train_catboost_nn_zoo_v3.py`)
- **功能**: 
  1. CatBoostRegressor: 原生深度解析房源描述与名称文本，并结合微环境高基数类别字典，迭代 3500 轮；
  2. PyTorch Tabular ResNet: 4 层连续特征全连接深度残差块网络，采用 AdamW 与余弦退火学习率。
- **运行命令**:
```powershell
F:\Anaconda\envs\pytorch\python.exe train_catboost_nn_zoo_v3.py
```
- **耗时预估**: 约 10 ~ 18 分钟（GPU 加速模式下）
- **核验标准**:
  - 生成目录: `oof_cat_nn_v3/`
  - 产出文件清单:
    * `oof_cat_text.npy` 与 `test_cat_text.npy` (单模 RMSLE $\approx 0.28866$)
    * `oof_nn_tabular.npy` 与 `test_nn_tabular.npy` (单模 RMSLE $\approx 0.29729$)
  - 标志文件: `READY_V3.flag` 与 `manifest_v3.json`

---

### 步骤 4: Level-2 约束凸优化 Stacking 与最终预测交付 (`stack_and_submit_gbdt_textzoo.py`)
- **功能**:
  1. 跨代纳管模型池并执行数据完整性与分布漂移审计；
  2. 运行 8 步折内前向贪婪模型优选（Forward Stepwise Selection）；
  3. 执行 SLSQP 约束二次规划凸优化，求解非负归一化凸权重（$w_i \ge 0, \sum w_i = 1$）；
  4. 运行严格折内 Log-Bias 动态微调与防漂移熔断（$|b| \le 0.03$）；
  5. 执行赛题硬边界截断 `np.clip(price, 20.0, 2000.0)`；
  6. 执行端到端四维统计门禁检测，产出提交文件并自动备份归档。
- **运行命令**:
```powershell
F:\Anaconda\envs\pytorch\python.exe stack_and_submit_gbdt_textzoo.py
```
- **耗时预估**: 约 30 ~ 60 秒
- **核验标准**:
  - 预期控制台关键输出指标：
    ```text
    >>> 前向贪婪选出模型: ['v3_xgb_main', 'v1_xgb', 'v3_lgb_et', 'v3_nn_tabular', 'v3_lgb_huber', 'v1_lgb_et', 'v1_lgb_std', 'v3_xgb_deep']
    >>> SLSQP 约束凸组合权重:
        v3_xgb_main: 0.2444
        v3_lgb_et: 0.1711
        v3_lgb_huber: 0.1523
        v1_xgb: 0.1495
        v3_nn_tabular: 0.0856
        v1_lgb_et: 0.0818
        v1_lgb_std: 0.0675
        v3_xgb_deep: 0.0480
    >>> 折内 Log-Bias 微调平均偏置: +0.000412
    >>> [20.0, 2000.0] 截断后 10-Fold OOF RMSLE: 0.281090
    >>> 测试集预测均值: 213.87 美元 (对数均值: 5.14)
    >>> [门禁质检] 全部 4 项质检通过！
    ```
  - 产出目标交付文件:
    * 根目录: `submission.csv` (184,962 字节, 11972 行)
    * 历史备份: `submission_stacking_v3_rmsle_0.28109.csv` (184,962 字节, 11972 行)

---

## 四、最终交付物指标校验对照清单 (Inspection Checklist)

完成上述 4 步流程后，请运行以下自动化脚本，对所有关键产出进行一次性闭环核验：

```powershell
F:\Anaconda\envs\pytorch\python.exe -c "
import os, numpy as np, pandas as pd

print('=== 1. 提交文件质检 ===')
sub_path = 'submission.csv'
assert os.path.exists(sub_path), 'submission.csv 不存在！'
df_sub = pd.read_csv(sub_path)
assert len(df_sub) == 11971, f'行数异常: {len(df_sub)}'
assert list(df_sub.columns) == ['id', 'price'], f'列名异常: {df_sub.columns}'
assert not df_sub.isna().any().any(), '存在 NaN 空值！'
assert df_sub['price'].min() >= 20.0, '存在低于 20.0 的价格！'
assert df_sub['price'].max() <= 2000.0, '存在高于 2000.0 的价格！'
mean_p = df_sub['price'].mean()
print(f'✅ 提交文件正常: 行数={len(df_sub)}, 均值=${mean_p:.2f}, 最小值=${df_sub.price.min():.2f}, 最大值=${df_sub.price.max():.2f}')

print('\n=== 2. 模型 OOF RMSLE 核验 ===')
y_log1p = np.load('cache_10fold_v3/y_log1p.npy')
def calc_rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

for m_name, path in [
    ('v3_xgb_main', 'oof_gbdt_v3/oof_xgb_main.npy'),
    ('v3_lgb_et', 'oof_gbdt_v3/oof_lgb_et.npy'),
    ('v3_lgb_huber', 'oof_gbdt_v3/oof_lgb_huber.npy'),
    ('v3_cat_text', 'oof_cat_nn_v3/oof_cat_text.npy'),
    ('v3_nn_tabular', 'oof_cat_nn_v3/oof_nn_tabular.npy')
]:
    arr = np.load(path)
    score = calc_rmsle(y_log1p, arr)
    print(f'✅ {m_name:15s} OOF RMSLE: {score:.5f}')
"
```

预期输出必须全部打勾通过：
```text
=== 1. 提交文件质检 ===
✅ 提交文件正常: 行数=11971, 均值=$213.87, 最小值=$26.25, 最大值=$1564.13

=== 2. 模型 OOF RMSLE 核验 ===
✅ v3_xgb_main      OOF RMSLE: 0.28389
✅ v3_lgb_et        OOF RMSLE: 0.28421
✅ v3_lgb_huber     OOF RMSLE: 0.28423
✅ v3_cat_text      OOF RMSLE: 0.28866
✅ v3_nn_tabular    OOF RMSLE: 0.29729
```

---

## 五、常见问题与排障指引 (Troubleshooting FAQ)

### Q1: 训练 XGBoost / CatBoost 时报 CUDA 显存不足 (CUDA Out of Memory)？
- **原因**: 批量并行处理时 GPU 显存被其他程序占用，或显存小于 8GB。
- **解决方案**:
  1. 打开 `train_gbdt_v3.py`，将 `tree_method='hist'` 的 `device='cuda'` 修改为 `device='cpu'`；
  2. 或在 `train_catboost_nn_zoo_v3.py` 中将 Tabular ResNet 的 `batch_size` 从 `256` 调减至 `128`。

### Q2: 为什么生成提交文件时测试集均值偏离很远？
- **警惕**: 请务必使用 `stack_and_submit_gbdt_textzoo.py` 中的 SLSQP 凸优化求解器，切勿使用无约束的 LinearRegression 或未限制深度的复杂树模型，否则可能重蹈 P5 阶段测试集负偏覆辙！

### Q3: 能否在单卡环境下分批次运行？
- **可以**: 每一个脚本均具备独立的缓存和断点落盘机制。例如完成 `train_gbdt_v3.py` 后，所有的预测矩阵已经落盘在 `oof_gbdt_v3/` 下，后续随时可以单独运行 `train_catboost_nn_zoo_v3.py` 或二阶堆叠。

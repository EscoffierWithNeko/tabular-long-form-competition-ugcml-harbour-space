# 模型训练脚本库归档与多样性架构说明 (Model Training Zoo)

本目录归档了在比赛推进过程中研发的全部模型训练脚本。为了打破单一 GBDT 的预测方差壁垒，我们构建了**高正交性、多损失函数、异构混合的模型库体系 (Model Zoo)**，涵盖 LightGBM (多种变体)、XGBoost (GPU Hist 与深树)、CatBoost (深度原生文本回归) 以及 PyTorch (Tabular ResNet 表格神经网络)。

---

## 一、模型库分类与脚本清单

| 类别 | 归档脚本 | 适用特征基座 | 模型家族 / 核心算法 | 训练机制与多样性设计 |
| :--- | :--- | :--- | :--- | :--- |
| **GBDT v3 (核心)** | `train_gbdt_v3.py` | `cache_10fold_v3/` | 7 大正交 GBDT 模型：<br>1. `v3_lgb_main`<br>2. `v3_lgb_et`<br>3. `v3_lgb_dart`<br>4. `v3_lgb_huber`<br>5. `v3_xgb_main`<br>6. `v3_xgb_deep`<br>7. `v3_lgb_res` | DART 树随机丢弃、Huber 鲁棒回归防极值过拟合、ExtraTrees 极端随机化切分、XGBoost max_depth=9 深树捕捉高阶交互、残差补偿回归 |
| **GBDT v4** | `train_gbdt_v4.py` | `cache_10fold_v4/` | 空间与语义增强 7 路 GBDT 族 | 继承 v3 架构，引入空间 KNN 与 BGE 降维特征基座，支持断点续训与折并行 |
| **GBDT 早期版本** | `train_lgb_xgb_zoo.py`<br>`train_lgb_xgb_zoo_v2.py`<br>`train_gbdt_extra_v2.py` | `cache_10fold/`<br>`cache_10fold_v2/` | 经典 LightGBM、XGBoost、HistGBDT 等基础基线模型库 | 初期参数搜索与基准建立 |
| **深度异构 v3 (核心)** | `train_catboost_nn_zoo_v3.py` | `cache_10fold_v3/` | 1. `v3_cat_text_deep`<br>2. `v3_nn_tabular` | **CatBoost 深度原生文本**：直接处理 `name`, `description`, `amenities` 词袋/Token 字典，搭配孪生高基数键；<br>**PyTorch Tabular ResNet**：实体嵌入 (Entity Embedding) + 残差块 + 批量归一化 + Dropout |
| **深度异构 v4** | `train_catboost_nn_zoo_v4.py` | `cache_10fold_v4/` | 深度语义与多模态扩展版异构库 | 支持文本字典深层分词与 PyTorch 混合精度训练 |
| **深度异构早期版本** | `train_catboost_nn_zoo.py`<br>`train_catboost_nn_zoo_v2.py` | `cache_10fold/`<br>`cache_10fold_v2/` | 初代 CatBoost 与 MLP / ResNet | 验证神经网络与原生文本模型在表格任务中的互补收益 |
| **集成/混合实验** | `train_advanced_ensemble.py`<br>`train_blended_top_tier.py` | 跨代 OOF | 早期模型融合与简单加权实验脚本 | 验证初步融合假设 |

---

## 二、关键模型算法与多样性设计

### 1. LightGBM DART (`v3_lgb_dart`)
* **原理**: 借鉴深度学习的 Dropout 机制，在每次迭代时以一定概率 (`drop_rate=0.1`) 随机丢弃已经建立的树，防止后期弱学习器对前期残差过度微调而陷入过拟合。
* **参数亮点**: `boosting_type='dart'`, `drop_rate=0.1`, `skip_drop=0.5`, `learning_rate=0.04`。
* **效果**: 相比标准 GBDT，为 Stacking 提供了具有高度正交预测误差的基元。

### 2. LightGBM Huber 鲁棒损失 (`v3_lgb_huber`)
* **原理**: 房源价格存在大量高价离群点（如奢华别墅、特殊商业租赁）。Huber 损失在误差较小时表现为 L2 损失，误差较大时线性衰减为 L1 损失，极大增强了模型对极端噪音标签的抗干扰性。
* **参数亮点**: `objective='huber'`, `alpha=0.8`。

### 3. ExtraTrees 极端随机化树 (`v3_lgb_et`)
* **原理**: 在树分裂时不仅随机选择特征子集，且在连续特征的分裂阈值选择上引入极端随机性（`extra_trees=True`），极大地丰富了树结构的多样性。

### 4. XGBoost GPU Hist 深度交互树 (`v3_xgb_deep`)
* **原理**: 采用 `tree_method='hist'`, `device='cuda'`, `max_depth=9`。深树能够充分捕获地理微网格与容纳人数、设施之间的多层阶梯交互，弥补浅层树表达能力的不足。

### 5. CatBoost 深度原生文本回归器 (`v3_cat_text_deep`)
* **原理**: 利用 CatBoost 独创的在线文本标记与字典嵌入（Tokenization & Dictionaries），将 `clean_name`, `clean_desc`, `clean_amenities` 作为原生文本特征直接喂入，同时结合 `twin_3tuple`, `twin_4tuple`, `host_id` 等高基数类别，端到端学习文本模式与房东属性对价格的综合调制。
* **参数亮点**: `depth=7~8`, `iterations=1200`, `text_processing` 包含 Bi-Gram / Tri-Gram 提取与 TF-IDF 频率加权。

### 6. PyTorch Tabular ResNet (`v3_nn_tabular`)
* **网络架构**:
  * **输入层**: 稠密数值特征（经过 QuantileTransformer / RobustScaler 归一化） + 类别特征 Embedding 映射层。
  * **隐藏层**: 3 组带有 Skip-Connection 的残差块（`Linear -> BatchNorm -> LeakyReLU -> Dropout -> Linear -> Residual Add`）。
  * **训练细节**: AdamW 优化器，CosineAnnealingLR 学习率衰减，平滑 L1 损失函数。
  * **价值**: 神经网络具有完全不同于决策树轴对齐切分的决策边界，在跨模型 Stacking 中提供了极高价值的互补性。

---

## 三、模型训练标准化运行命令

各脚本均提供了灵活的命令行参数，支持指定模型、指定折数及断点续训：

```powershell
# 训练指定 GBDT 模型（支持 v3_lgb_dart, v3_xgb_deep, v3_lgb_huber 等）
F:\Anaconda\envs\pytorch\python.exe train_gbdt_v3.py --model v3_lgb_dart

# 训练全套 7 大 GBDT 模型
F:\Anaconda\envs\pytorch\python.exe train_gbdt_v3.py --model all

# 训练深度原生文本 CatBoost 模型（支持针对特定折增量调试）
F:\Anaconda\envs\pytorch\python.exe train_catboost_nn_zoo_v3.py --model v3_cat_text_deep

# 训练 PyTorch 表格残差网络
F:\Anaconda\envs\pytorch\python.exe train_catboost_nn_zoo_v3.py --model v3_nn_tabular
```

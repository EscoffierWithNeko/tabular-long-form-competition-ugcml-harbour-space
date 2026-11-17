# 关键缓存基座与 OOF 预测库说明 (Cache & OOF Archive Reference)

本项目在探索和建模迭代过程中，针对 10 折交叉验证体系生成并持久化了多代特征缓存基座与模型 OOF 预测结果。为了避免冗余复制导致磁盘空间膨胀及破坏正在进行的模型训练管道与相对路径依赖，原始海量特征文件与数组缓存保留在工作区对应目录中，本说明文件作为其权威索引与架构规范字典。

---

## 一、特征工程缓存目录 (Feature Caches)

### 1. `cache_10fold/` (初代经典特征基座 - v1)
* **定位**: 赛题初期建立的经典数值、统计与基础目标编码基座。
* **核心内容**:
  * `folds_10.npy`: 10 折索引划分定义（划分依据为空间分层与 Target 分布，保证跨模型评估口径绝对一致）。
  * `fold_ids.npy`: 训练集样本对应的 Fold ID 数组。
  * `fold_te/`: 10 折严格无泄漏目标编码与聚合统计子目录（`fold_0.npz` ~ `fold_9.npz`）。
  * `manifest.json`: 包含特征列名列表、数据形状与元数据描述。
  * `READY.flag`, `READY_CLEAN.flag`: 特征生成成功流水线哨兵文件。

### 2. `cache_10fold_v2/` (高阶交叉与微网格空间基座 - v2)
* **定位**: 引入空间经纬度微网格聚类、高阶非线性交互比率（如每床容纳人数、每室卫生间比例等）。
* **核心内容**:
  * `folds_10.npy`, `fold_ids.npy`: 继承自 v1 的一致性划分标准。
  * `fold_te/`: 逐折 Target Encoding 与聚类统计。
  * `test_base_features.parquet`: 预提取并压缩的测试集基础数值与交叉特征。
  * `manifest_v2.json`, `READY_V2.flag`: v2 版本元数据配置与校验哨兵。

### 3. `cache_10fold_v3/` (孪生微环境复合基座 - v3制胜核心)
* **定位**: **取得 Kaggle Public LB 0.27676 / CV 0.28109 顶尖成绩的关键核心特征基座**。
* **核心技术**:
  * 孪生微环境复合键构建：`twin_3tuple`, `twin_4tuple`, `twin_5tuple`（如 `host_id__geo_cell__room_type__accommodates__bedrooms`），精确捕捉同房东、同微区、同房型下的基准定价。
  * 微网格豪华度差距 (Luxury Delta) 与容纳密度计算。
  * 严格折内无泄漏 Target Encoding (`fold_0.npz` ~ `fold_9.npz`)。
* **核心内容**:
  * `folds_10.npy`, `fold_ids.npy`: 权威 10 折切分定义。
  * `fold_te/`: 10 折独立的 Out-of-Fold 目标编码矩阵。
  * `manifest_v3.json`: 记录 v3 特征清单（基础数值 361 列 + OOF 170 列，共 531 维）。
  * `READY_V3.flag`: v3 基座就绪状态标记。

### 4. `cache_10fold_v4/` (语义嵌入与空间 KNN 邻域基座 - v4)
* **定位**: 融合预训练语言模型（BGE 嵌入降维向量）与空间地理 KNN 邻域特征的高阶基座。
* **核心内容**:
  * `bge_embeddings.npz`: 文本名称与描述的多模态 BGE 稠密嵌入表示。
  * `folds_10.npy`, `fold_ids.npy`: 跨代一致折定义。
  * `fold_te/`: 适配 v4 特征集合的逐折目标编码。
  * `manifest_v4.json`, `READY_V4.flag`: v4 元数据与就绪哨兵。

---

## 二、模型 OOF 预测与测试集推断结果库 (OOF & Test Predictions)

所有模型输出均遵循严格统一的命名与对齐规范：
* 训练集 OOF 形状：`(47293,)`，与 `y_log1p` 完全对齐。
* 测试集推断形状：`(11971,)`，取自 10 折验证模型的等权平均测试集预测。

### 1. GBDT 系列预测库
* **`oof_gbdt/` (v1)**:
  * `oof_lgb_std.npy`, `test_lgb_std.npy`: 标准 LightGBM 基线预测。
  * `oof_lgb_et.npy`, `test_lgb_et.npy`: ExtraTrees 极度随机化树 LightGBM 预测。
  * `oof_xgb.npy`, `test_xgb.npy`: 初始 XGBoost 回归器预测。
  * `summary.json`: 记录各模型单模 10 折 RMSLE 评分。
* **`oof_gbdt_v2/` (v2)**:
  * `oof_lgb_std.npy`, `oof_lgb_et.npy`, `oof_xgb.npy`, `oof_xgb_deep.npy` (深度树 XGBoost), `oof_lgb_huber.npy` (Huber 鲁棒损失树), `oof_lgb_res.npy` (残差拟合树)。
* **`oof_gbdt_v3/` (v3)**:
  * `oof_v3_lgb_main.npy`, `oof_v3_lgb_et.npy`, `oof_v3_lgb_dart.npy` (DART 随机丢弃), `oof_v3_lgb_huber.npy`, `oof_v3_xgb_main.npy`, `oof_v3_xgb_deep.npy`, `oof_v3_lgb_res.npy`。
  * 以及对应的 `test_v3_*.npy` 预测数组。
* **`oof_gbdt_v4/` (v4)**:
  * 空间 KNN 与 BGE 特征驱动的各分支模型预测。

### 2. 深度异构与文本模型预测库 (CatBoost + PyTorch NN)
* **`oof_cat_nn/` (v1)**:
  * `oof_cat_text.npy`, `test_cat_text.npy`: CatBoost 原生文本特征挖掘。
  * `oof_nn_tabular.npy`, `test_nn_tabular.npy`: PyTorch 表格神经网络初代预测。
* **`oof_cat_nn_v2/` (v2)**:
  * 深度表格 ResNet 与多字段 CatBoost 文本增强预测。
* **`oof_cat_nn_v3/` (v3)**:
  * `oof_v3_cat_text_deep.npy`, `test_v3_cat_text_deep.npy`: 融合复合键高基数类别的深度 CatBoost 原生文本模型。
  * `oof_v3_nn_tabular.npy`, `test_v3_nn_tabular.npy`: 10 折 PyTorch Tabular ResNet (含 Entity Embedding 与 Skip Connection)。
* **`oof_cat_nn_v4/` (v4)**:
  * 接入高阶多模态语义空间的持续迭代模型输出。

---

## 三、跨代 Stacking 调度指引

在 Level-2 Stacking 融合管线（如 `stack_and_submit_gbdt_textzoo.py`）中，系统会自动跨目录扫描上述 OOF 预测库，通过贪婪前向特征优选算法（Greedy Forward Stepwise Selection）在超 15+ 异构模型池中寻找正交性互补组合，最终执行 SLSQP 凸优化权重解算，确保达到理论极限泛化性能。

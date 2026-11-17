# 特征工程管线归档与架构演化说明 (Feature Engineering Pipelines)

本目录归档了在 Kaggle `tabular-long-form-competition-ugcml-harbour-space` 房源价格预测竞赛中研发的历代核心特征工程管道。所有特征工程均严格秉承**零数据泄漏 (Zero Leakage)** 铁律，通过严格统一的 10 折分层交叉验证进行特征提取、标准化与目标编码。

---

## 一、特征工程版本演进总览

| 版本脚本 | 产出缓存目录 | 特征维度 (Base / TE / 总维数) | 核心算法与特征设计突破 | 对应模型最佳成绩 (Public LB) |
| :--- | :--- | :--- | :--- | :--- |
| **`fe_pipeline_10fold.py` (v1)** | `cache_10fold/` | 185 / 110 / ~295 维 | 基础数值解析、房型文本长度统计、标准 Out-of-Fold 目标编码 (Target Encoding)、地理坐标粗粒度离散化 | 0.28047 |
| **`fe_pipeline_v2.py` (v2)** | `cache_10fold_v2/` | 278 / 140 / ~418 维 | 引入微网格经纬度空间聚类 (Geo Hashing / Clustering)、非线性容量比率 (accommodates/bedrooms, beds/baths)、多字段文本 TF-IDF SVD 降维 | 0.27755 |
| **`fe_pipeline_v3.py` (v3 - 顶尖核心)** | `cache_10fold_v3/` | 361 / 170 / **531 维** | **制胜核心发明**：孪生微环境复合键 (`twin_3tuple`, `twin_4tuple`, `twin_5tuple`)、微区奢华度差值、折内严密无泄漏平滑 TE | **`0.27676` (Top 1% 突破)** |
| **`fe_pipeline_v4.py` (v4)** | `cache_10fold_v4/` | 386 / 180 / ~566 维 | 地理空间 KNN 邻域特征（K 最近邻均价与密度）、多模态预训练模型（BGE 嵌入降维向量）、嵌套交叉验证特征筛选基座 | 探索与持续迭代 |

---

## 二、核心脚本功能与实现细节

### 1. `fe_pipeline_10fold.py` (初代经典特征工程)
* **核心职责**: 构建全项目的交叉验证骨架与基准特征库。
* **主要特征**:
  * 浴室类型清洗与数值提取 (`bathrooms_text` 解析出浴缸/独立卫浴/共享卫浴标志位)。
  * 日期特征解析：房东入驻天数 (`host_days`)、首条/最近评论间隔天数、评论活跃度。
  * 房源硬件比率：床位数/卧室数比例、容纳人数/床位比例。
  * 目标编码：基于 10 折交叉验证划分，仅在 `Train[fold_train_idx]` 上计算分组均值与平滑权重，分别映射到 `Train[fold_val_idx]` 和 `Test`，杜绝一切标签泄露。

### 2. `fe_pipeline_v2.py` (空间聚类与多维交互增强)
* **核心职责**: 引入空间非线性拓扑与多模态初探。
* **主要特征**:
  * 空间网格化：对经纬度应用高斯扰动鲁棒网格编码 (`geo_cell_l1`, `geo_cell_l2`)，捕捉地理空间局部溢价。
  * 文本基础挖掘：对 `amenities` 设施列表进行高频词热编码与设施数量聚类。
  * 房东声誉交叉：超赞房东身份与评论得分的复合加权指数。

### 3. `fe_pipeline_v3.py` (孪生微环境复合基座 —— 核心制胜算法)
* **核心职责**: 实现突破性高精定价建模，直接贡献了 Public LB 0.27676 的成绩。
* **关键技术突破**:
  1. **孪生微环境复合键 (Twin-MicroEnvironment Tuples)**:
     * `twin_3tuple`: `geo_cell + room_type + accommodates`
     * `twin_4tuple`: `host_id + geo_cell + room_type + accommodates`
     * `twin_5tuple`: `host_id + geo_cell + room_type + accommodates + bedrooms`
     同一房东在同一微小街区、针对相同房型与容纳规格的房源，其基准定价具有极高的先验锁定效应。
  2. **容纳密度与微网格奢华度差距 (Luxury Delta)**:
     * 计算 `accommodates_per_bedroom`、`bathrooms_per_guest`。
     * 提取同街区平均硬件指标并计算当前房源相对街区中位数的离差率 (Neighborhood Premium / Luxury Ratio)。
  3. **严格 10-Fold 独立张量保存**:
     * 逐折导出 `fold_te/fold_0.npz` ~ `fold_9.npz`，避免训练时重复内存计算，大幅加速跨模型训练。

### 4. `fe_pipeline_v4.py` (高阶空间 KNN 与语义嵌入管线)
* **核心职责**: 进一步挖掘地理近邻与深度语义信息。
* **主要特征**:
  * 空间 KNN 邻域聚合：利用 KD-Tree / BallTree 搜索空间半径与 K 近邻，计算距离衰减加权价格先验。
  * 预训练语言模型多模态特征：整合 BGE (BAAI General Embedding) 对房源描述的密集向量表征。
  * 强健方差滤波与相关性降噪。

---

## 三、代码运行与验证方法

所有特征工程脚本均支持命令行独立运行，内置自动化防泄漏检查与形状校验：
```powershell
# 运行特征提取管线（示例为 v3）
F:\Anaconda\envs\pytorch\python.exe fe_pipeline_v3.py
```
运行后将自动生成并校验哨兵标志文件 `cache_10fold_v3/READY_V3.flag`，确保后续模型无缝加载。

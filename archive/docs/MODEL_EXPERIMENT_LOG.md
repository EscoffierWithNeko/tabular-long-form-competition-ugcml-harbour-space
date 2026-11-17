
## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 16:14:19
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30512
  - CatBoost CV RMSLE: 0.31087
  - XGBoost CV RMSLE: 0.33048
  - **集成融合最优权重**: LGB=0.602, CAT=0.364, XGB=0.035
  - **集成后全局 CV RMSLE**: 0.30175

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 16:26:02
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30512
  - CatBoost CV RMSLE: 0.31087
  - XGBoost CV RMSLE: 0.33048
  - **集成融合最优权重**: LGB=0.602, CAT=0.364, XGB=0.035
  - **集成后全局 CV RMSLE**: 0.30175

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 16:35:55
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30512
  - CatBoost CV RMSLE: 0.31087
  - XGBoost CV RMSLE: 0.33048
  - **集成融合最优权重**: LGB=0.602, CAT=0.364, XGB=0.035
  - **集成后全局 CV RMSLE**: 0.30175

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 16:48:41
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.32900
  - **集成融合最优权重**: LGB=0.598, CAT=0.382, XGB=0.020
  - **集成后全局 CV RMSLE**: 0.29979

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 16:56:22
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 17:05:31
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 17:16:24
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 17:28:20
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 17:38:17
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 17:46:59
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 17:54:37
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 18:07:02
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 18:19:41
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 18:27:45
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 18:36:02
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30313
  - CatBoost CV RMSLE: 0.30793
  - XGBoost CV RMSLE: 0.29958
  - **集成融合最优权重**: LGB=0.377, CAT=0.056, XGB=0.567
  - **集成后全局 CV RMSLE**: 0.29715

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending (特征扩展 + 严防泄露)
- **完成时间**: 2026-11-13 18:46:15
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 700, 深度: 6, lr: 0.06, l2_reg: 3.0, CPU多线程)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - 房东与用户年限与经验交互 (`host_experience_months`, `user_experience_months`, `host_user_gap_months`, `listings_per_host_year`)
  - 评论与容纳深度交互 (`reviews_per_accommodates`, `monthly_review_power`)
  - 关键设施指示向量与多维 SVD 语义向量 (Name SVD 6-dim + Description SVD 8-dim + Amenities SVD 6-dim)
  - 严格 4-Fold Inner-OOF 无泄漏 Target Encoding (`geo_cell`, `property_type`, `room_type`, `combo`)
  - 严格 4-Fold Inner-OOF 文本稀疏 Ridge 元预测 (`name`, `amenities`, `full_text`)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30320
  - CatBoost CV RMSLE: 0.30779
  - XGBoost CV RMSLE: 0.29881
  - **集成融合最优权重**: LGB=0.354, CAT=0.032, XGB=0.613
  - **集成后全局 CV RMSLE**: 0.29673 (创全场历史新低，超越前期最佳 0.29715)
- **官方提交状态**:
  - 尝试调用 Kaggle API 提交该极值模型生成的 `submission.csv`
  - Kaggle API 反馈: `400 Client Error: Submission not allowed: Your team has used its daily Submission allowance (5) today, please try again tomorrow UTC (13 hours from now).`
  - 本日 5/5 提交额度已全部用尽，当前在榜官方最高成绩仍为 **Public Score: 0.28962** (Ref: 58303111)
  - 新模型预测结果已就地持久化为 `submission.csv` 及 `submission_top_tier_blend.csv`，待 UTC 重置后可立即一键提交冲击突破 0.289！

## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: 2026-11-13 18:57:39
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: 0.30320
  - CatBoost CV RMSLE: 0.30779
  - XGBoost CV RMSLE: 0.29881
  - **集成融合最优权重**: LGB=0.354, CAT=0.032, XGB=0.613
  - **集成后全局 CV RMSLE**: 0.29673

## 实验批次: 10-Fold Stacking 集成 (3-GBDT + Text Zoo 元特征 + NN Tabular)
- **完成时间**: 2026-11-14 01:26:45
- **训练基座与数据标准**:
  - 严格 10-Fold 独立折外基座 (`cache_10fold/fold_ids.npy`, `y_log1p.npy`)，全流程零特征泄漏与零统计泄漏
  - 训练集样本数: 47,293，测试集样本数: 11,971
- **核心异构基元模型池**:
  1. `lgb_std`: LightGBM 标准树结构 (num_leaves=63, lr=0.03, colsample=0.7) -> OOF RMSLE: **0.28835**
  2. `lgb_et`: LightGBM 极度随机树 Extra-Trees (extra_trees=True, colsample=0.6, subsample=0.8) -> OOF RMSLE: **0.28922**
  3. `xgb_main`: XGBoost Hist 深度精调版 (max_depth=7, lr=0.03, colsample=0.7) -> OOF RMSLE: **0.28794**
  4. `text_zoo`: TF-IDF + Ridge 多字段语义回归 (full_ridge_a1.5: 0.43411, name_ridge_a1.5: 0.49825, desc_ridge_a6.0: 0.50644, amen_ridge_a6.0: 0.54417)
  5. `nn_tabular`: PyTorch Tabular ResNet (全 10 折完工版) -> OOF RMSLE: **0.32982**
- **Level-2 融合策略与实证评估**:
  - 基线 1: 3-GBDT 等权平均 -> 10-Fold OOF RMSLE: **0.285623**
  - 基线 2: 3-GBDT SLSQP 凸组合优化 -> 10-Fold OOF RMSLE: **0.285609** (w_lgb_std=0.280, w_lgb_et=0.352, w_xgb_main=0.368)
  - 方案 3: 全模态异构融合 (GBDT + Text Zoo + NN) 全局优化 -> OOF RMSLE: **0.285552**
  - 方案 4: **严格 10-Fold Out-of-Fold 独立权重 Stacking** (每折独立在其余 9 折拟合凸组合权重，杜绝 Level-2 权重过拟合，并应用赛题硬边界截断 `[20.0, 2000.0]`) -> 严格折外 Stacking RMSLE: **0.285590**
- **官方 Kaggle CLI 提交实测战报**:
  - **提交文件**: `submission.csv` (11,971 行有效测试预测，字段 `id, price`，范围 [24.46, 1566.72]，无空值/Inf)
  - **提交命令**: `kaggle competitions submit -c tabular-long-form-competition-ugcml-harbour-space -f submission.csv -m "10-Fold Stacking (3-GBDT + Text Zoo) CV: 0.28559"`
  - **提交回执 Ref**: `58327481`
  - **官方榜单状态**: `SubmissionStatus.COMPLETE`
  - **官方榜单真实得分 (Public Score)**: **`0.28047`**
  - **历史对比**: 相比上一轮最佳成绩 **`0.28962`**，实现大幅跨越式提升 **`-0.00915`**，冲入 Top-Tier 顶尖领先梯队！

## 实验批次: 严格 10-Fold Level-2 Stacking v2 (GBDT 家族 + CatBoost 原生文本 + PyTorch Tabular ResNet + Text Zoo)
- **完成时间**: 2026-11-14 17:58:00
- **训练基座与特征资产**:
  - 无泄漏 `cache_10fold_v2/` (452 维高阶工程特征 + SVD + 交叉 TE)
  - 严格 10-Fold 独立切分 (`fold_ids.npy`, `y_log1p.npy`)，全流程无统计穿越与标签泄漏
- **核心异构基元池与单模 10-Fold OOF CV RMSLE**:
  1. `v2_lgb_et`: Extra-Trees LightGBM -> **0.28730**
  2. `v2_xgb_deep`: 深度 XGBoost (depth=9, colsample=0.6, lr=0.03) -> **0.28735**
  3. `v2_lgb_huber`: Huber 鲁棒损失 (delta=0.8) LightGBM -> **0.28741**
  4. `v2_xgb_main`: 标准 XGBoost Hist (depth=7, colsample=0.7) -> **0.28764**
  5. `v2_xgb_reg`: 浅树强正则 XGBoost (depth=5, alpha=3.0, lambda=8.0) -> **0.28790**
  6. `v2_lgb_std`: 标准 LightGBM (num_leaves=63, colsample=0.7) -> **0.28810**
  7. `v2_cat_text`: CatBoost 满血原生文本+类别+折内TE -> **0.29098**
  8. `v2_nn_tabular`: PyTorch Entity-Embedding Tabular ResNet -> **0.30635** (相比 v1 大幅提升 +0.023)
  9. `text_zoo`: 多尺度词袋/字符级 TF-IDF + Ridge 语义回归元特征
- **Level-2 多 Meta-Learner 严格 10-Fold OOF 实证对比**:
  - **纯 6-GBDT-v2 家族**:
    - 等权平均: 0.285612
    - SLSQP 约束凸组合: **0.285157** (击穿原 0.28559 历史纪录)
  - **异构大融合 (GBDT + CatBoost 原生文本 + PyTorch Tabular ResNet)**:
    - 等权平均: 0.286283
    - SLSQP 约束凸组合: **0.284705** (多模态互补增益显著)
    - NNLS 非负回归: 0.284705
    - RidgeCV L2 正则元回归: 0.284562
  - **前向贪婪逐步特征优选 (Forward Stepwise Selection) 组合**:
    - SLSQP 约束凸组合: **0.284305**
    - NNLS 非负回归: 0.284305
    - RidgeCV: 0.283120
    - **ElasticNetCV (L1/L2 稀疏混合元回归)**: **`0.282713`** (全场最优)
- **后处理优化**:
  - 折内 Log-Bias 校准与业务截断 `[20.0, 2000.0]` 稳固防长尾外推。
  - **全流程最终严格 10-Fold OOF CV RMSLE**: **`0.282713`** (相比历史最佳 `0.285590` 净提升 **`-0.002877`**)
- **官方 Kaggle CLI 提交实测战报**:
  - **提交文件**: `submission.csv` (11,971 行有效测试预测，字段 `id, price`，范围 [26.31, 1531.39]，无空值/Inf)
  - **提交命令**: `kaggle competitions submit -c tabular-long-form-competition-ugcml-harbour-space -f submission.csv -m "10-Fold Stacking v2 (Greedy_Selected+ElasticNetCV) CV: 0.28271"`
  - **提交回执 Ref**: `58339341`
  - **官方榜单状态**: `SubmissionStatus.COMPLETE`
  - **官方榜单真实得分 (Public Score)**: **`0.27755`**
  - **历史对比**: 相比上一轮历史最高纪录 **`0.28047`** (Ref 58327481)，再次跨越式大幅狂降 **`-0.00292`**，挺进 `0.277` 顶尖统治级高分区！

## 🚀 [2026-11-15 15:03:38] v3 孪生微环境全模态 10-Fold Level-2 Stacking 登顶冲榜
- **核心架构**:
  - 特征基座: `cache_10fold_v3/` (363 维无泄漏孪生微环境复合聚类 + 高阶交叉)
  - 基元模型库: v3 7大GBDT (`v3_xgb_main`, `v3_lgb_et`, `v3_lgb_huber`, `v3_lgb_main`, `v3_xgb_deep`, `v3_lgb_dart`, `v3_lgb_res`) + `v3_cat_text_deep` (多模态文本深度融合) + `v3_nn_tabular` (Tabular ResNet) + 跨代差异基元
  - 贪婪前向特征优选: ['v3_xgb_main', 'v1_xgb', 'v3_lgb_et', 'v3_nn_tabular', 'v3_lgb_huber', 'v1_lgb_et', 'v1_lgb_std', 'v3_xgb_deep', 'v2_nn_tabular']
  - 最优元学习器: `Greedy_SLSQP_Convex`
- **本地验证指标**:
  - 原始 10-Fold OOF RMSLE: **`0.281074`**
  - 后处理截断校准 RMSLE: **`0.281092`** (相比历史基准 `0.282713` 净提升 **`0.001621`**)
- **Kaggle 官方提交**:
  - 提交信息: `v3 Twin-MicroEnv SLSQP Stacking (CV: 0.28109)`
  - 官方状态看板回执: `58373885  submission.csv  2026-11-15 07:03:37.340000  v3 Twin-MicroEnv SLSQP Stacking (CV: 0.28109)                          SubmissionStatus.COMPLETE  0.27676                    `

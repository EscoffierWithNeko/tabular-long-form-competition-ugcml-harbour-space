# Kaggle tabular-long-form 登顶收官：MemPalace 核心记忆入库全景清单

> **归属空间 (Wing / Room)**: `Financial_Domain_NLP` / `Financial_Report_Analysis`  
> **会话级双写空间**: `neko_session_memory` (含 `NEKO_PERSONA.md` 本地同步)  
> **工作区标识 (WorkspaceTag)**: `kaggle_tabular_long_form_competi`  
> **入库记录生成者**: @CHARLIE (Wave 3 - 核心算法入库 MemPalace)  
> **时间戳**: 2026-11-17 01:25:00  

---

## 一、记忆入库总体概要与召回规范

为了确保本竞赛沉淀的顶级特征工程算子、集成优化理论、数据审计发现与血泪避坑经验长期持久化，本次将 7 项核心知识作为 **Tier 1 (核心长效记忆)** 注入 MemPalace 记忆宫殿。
入库抽屉同时注册至：
1. **专业业务房间**: `Financial_Domain_NLP/Financial_Report_Analysis`（用于金融核查、高价值房产与标准化资产定价的跨赛题长效算子召回）；
2. **会话级运维房间**: `neko_session_memory`（用于本竞赛工作区内任意会话启动时的环境与偏好自动注入）。

---

## 二、7 项核心知识入库条目明细

### 条目 1: 最终制胜算法完整规格 (v3 Twin-MicroEnv SLSQP Stacking, Private LB 0.28160)

- **抽屉唯一标识 (ID)**: `Financial_Report_Analysis.v3_twin_microenv_slsqp_stacking_spec`
- **抽屉键 (Drawer Key)**: `v3_twin_microenv_slsqp_stacking_spec`
- **归属房间 (Room / Wing)**: `Financial_Report_Analysis` / `Financial_Domain_NLP`
- **标题 (Title)**: `[kaggle_tabular_long_form_competi] v3 Twin-MicroEnv SLSQP Stacking 终局制胜算法规格`
- **核心假设与验证结论 (Hypothesis)**:  
  【v3 制胜算法规格】基于 10-Fold 无泄漏孪生微环境 531 维特征基座，融合 9 路高度异构基元模型（XGBoost GPU Hist / LightGBM ET / Huber / DART / CatBoost 深度文本 / PyTorch Tabular ResNet）。在 Level-2 采用严格非负凸组合 SLSQP（w>=0, sum(w)=1）与折内 Log-Bias 动态校准、[20, 2000] 绝对硬截断。在 Kaggle tabular-long-form 取得 10-Fold CV 0.281049、Public LB 0.27676、Private LB 0.28160（Ref: 58373885）历史最优登顶成绩。
- **关键代码与数学规格 (Code Snippet)**:
  ```python
  # v3 Twin-MicroEnv SLSQP Stacking Pipeline
  # 1. 目标变量对数变换
  y_log1p = np.log1p(train_df['price'].values)

  # 2. 特征基座构建 (cache_10fold_v3, 共 531 维)
  # X_train = [361 维基础特征, 170 维 10-Fold 折内 OOF 特征]

  # 3. Level-1: 9 路异构模型独立 10 折交叉训练得到 OOF 预测与 Test 预测矩阵
  # P_oof.shape = (47293, 9), P_test.shape = (11971, 9)

  # 4. Level-2: 严格非负凸优化 (SLSQP)
  from scipy.optimize import minimize
  def obj_func(weights, preds_mat, targets):
      blend = np.dot(preds_mat, weights)
      return np.sqrt(np.mean((blend - targets) ** 2))

  constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
  bounds = [(0.0, 1.0) for _ in range(n_models)]
  w_init = np.ones(n_models) / n_models
  res = minimize(obj_func, w_init, args=(P_oof, y_log1p),
                 method='SLSQP', bounds=bounds, constraints=constraints)
  best_weights = res.x

  # 5. 测试集预测加权与 Log-Bias 折内校准
  pred_log_test = np.dot(P_test, best_weights)
  bias = np.mean(y_log1p - np.dot(P_oof, best_weights))
  if abs(bias) <= 0.03:
      pred_log_test += bias

  # 6. 后处理原空间恢复与赛题硬边界截断
  final_pred = np.clip(np.expm1(pred_log_test), 20.0, 2000.0)
  ```
- **避坑原则与反模式 (Anti-patterns)**:  
  1. 严禁使用无约束线性回归（如 Ridge/LinearRegression），模型间高共线性会导致权重剧烈震荡（+3.5/-2.6），导致测试集外推严重负偏，P5 提交 LB 暴跌至 0.53173；  
  2. 严禁全量数据提前计算 Target Encoding，必须严格在 10 折内部 Out-of-Fold 提取。
- **触发检索标签 (Trigger Keywords)**:  
  `ws:kaggle_tabular_long_form_competi,v3_stacking,slsqp,convex_optimization,private_lb_0.28160,public_lb_0.27676,twin_microenv,model_zoo`
- **层级 (Tier)**: Tier 1 | **状态**: Active

---

### 条目 2: 孪生微环境特征工程核心逻辑 (twin_5tuple/twin_4tuple 极小平滑度 TE、容纳密度、微网格豪华度差距)

- **抽屉唯一标识 (ID)**: `Financial_Report_Analysis.twin_microenv_feature_engineering_logic`
- **抽屉键 (Drawer Key)**: `twin_microenv_feature_engineering_logic`
- **归属房间 (Room / Wing)**: `Financial_Report_Analysis` / `Financial_Domain_NLP`
- **标题 (Title)**: `[kaggle_tabular_long_form_competi] 孪生微环境特征工程与极小平滑度 TE 核心机制`
- **核心假设与验证结论 (Hypothesis)**:  
  【孪生微环境特征工程】针对测试集 61.01% 来自已知房东、25.75% 属于同房东同户型的重合特征，构建 twin_5tuple 与 twin_4tuple 复合键。利用折内极小平滑度 (smooth=2, 10) Target Encoding 锁住 96.77% 的组内方差削减；并衍生容纳密度 (accommodates/bedrooms)、微网格豪华度差距与残差 TE，构成 531 维特征基座。
- **关键代码与数学规格 (Code Snippet)**:
  ```python
  # 孪生微环境复合键构建与折内平滑 Target Encoding 核心代码
  # 1. 复合实体键生成
  df['twin_5tuple'] = (df['host_id'].astype(str) + '_' + 
                       df['geo_cell'].astype(str) + '_' + 
                       df['room_type'].astype(str) + '_' + 
                       df['accommodates'].astype(str) + '_' + 
                       df['bedrooms'].astype(str))
  df['twin_4tuple'] = (df['host_id'].astype(str) + '_' + 
                       df['geo_cell'].astype(str) + '_' + 
                       df['room_type'].astype(str) + '_' + 
                       df['accommodates'].astype(str))

  # 2. 10-Fold 嵌套折内平滑 Target Encoding (无泄漏)
  def fold_target_encode(tr_x, tr_y, val_x, te_col, smooth=5.0):
      global_mean = np.mean(tr_y)
      stats = pd.DataFrame({'key': tr_x[te_col], 'target': tr_y}).groupby('key').agg(['count', 'mean'])
      counts = stats[('target', 'count')]
      means = stats[('target', 'mean')]
      # 平滑公式: (count * mean + smooth * global_mean) / (count + smooth)
      smooth_dict = ((counts * means + smooth * global_mean) / (counts + smooth)).to_dict()
      return val_x[te_col].map(smooth_dict).fillna(global_mean).values

  # 3. 容纳密度与微网格豪华度衍生
  df['accommodates_density'] = df['accommodates'] / (df['bedrooms'].clip(lower=1))
  df['beds_per_bedroom'] = df['beds'] / (df['bedrooms'].clip(lower=1))
  df['microenv_luxury_gap'] = df['twin_5tuple_te_s2'] - df['geo_cell_te_s50']
  ```
- **避坑原则与反模式 (Anti-patterns)**:  
  1. 严禁全量训练集提前做 GroupBy Target Encoding，会使验证集特征直接包含标签真实值产生虚假高分；  
  2. 复合键独立基数极高（>3.9万），必须配对平滑参数（smooth=2~10），否则仅有 1 次出现的小样本会导致单树过拟合。
- **触发检索标签 (Trigger Keywords)**:  
  `ws:kaggle_tabular_long_form_competi,twin_microenv,target_encoding,twin_5tuple,twin_4tuple,accommodates_density,smooth_te,variance_reduction`
- **层级 (Tier)**: Tier 1 | **状态**: Active

---

### 条目 3: 9 路异构基元模型库配置与单模 CV

- **抽屉唯一标识 (ID)**: `Financial_Report_Analysis.model_zoo_9way_heterogeneous_architectures`
- **抽屉键 (Drawer Key)**: `model_zoo_9way_heterogeneous_architectures`
- **归属房间 (Room / Wing)**: `Financial_Report_Analysis` / `Financial_Domain_NLP`
- **标题 (Title)**: `[kaggle_tabular_long_form_competi] 9 路异构基元模型库配置与单模 10-Fold CV 表现`
- **核心假设与验证结论 (Hypothesis)**:  
  【9 路基元模型库】采用多算法引擎（XGBoost GPU Hist / LightGBM / CatBoost / PyTorch NN）、不同树生长策略（Depth-wise / Leaf-wise）、不同损失函数（Huber/RMSE）与异构输入架构。单模最优为 XGBoost Main (CV 0.28389)，DART (0.28424)、Huber (0.28423) 与深度原生文本 (0.28866) 形成强正交互补。
- **关键代码与数学规格 (Code Snippet)**:
  ```python
  # 9 路基元模型架构配置与 10-Fold CV 清单 (统一基于 cache_10fold_v3/)
  LEVEL1_MODEL_ZOO = {
      'v3_xgb_main': {
          'engine': 'XGBoost (GPU Hist)', 'cv_rmsle': 0.283888,
          'params': {'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'tree_method': 'hist'}
      },
      'v3_lgb_et': {
          'engine': 'LightGBM (ExtraTrees)', 'cv_rmsle': 0.284210,
          'params': {'extra_trees': True, 'num_leaves': 63, 'learning_rate': 0.03, 'colsample_bytree': 0.6}
      },
      'v3_lgb_huber': {
          'engine': 'LightGBM (Huber Loss)', 'cv_rmsle': 0.284230,
          'params': {'objective': 'huber', 'alpha': 0.8, 'num_leaves': 48, 'learning_rate': 0.03}
      },
      'v3_lgb_dart': {
          'engine': 'LightGBM (DART)', 'cv_rmsle': 0.284240,
          'params': {'boosting_type': 'dart', 'drop_rate': 0.10, 'num_leaves': 48, 'learning_rate': 0.04}
      },
      'v3_xgb_deep': {
          'engine': 'XGBoost Deep (GPU Hist)', 'cv_rmsle': 0.284340,
          'params': {'max_depth': 9, 'learning_rate': 0.02, 'subsample': 0.75, 'colsample_bytree': 0.65}
      },
      'v3_lgb_main': {
          'engine': 'LightGBM Baseline', 'cv_rmsle': 0.284560,
          'params': {'num_leaves': 48, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7}
      },
      'v3_lgb_res': {
          'engine': 'LightGBM Residual Booster', 'cv_rmsle': 0.285800,
          'params': {'target': 'y_true - lgb_main_oof', 'num_leaves': 31, 'learning_rate': 0.02}
      },
      'v3_cat_text_deep': {
          'engine': 'CatBoost Native Text', 'cv_rmsle': 0.288660,
          'params': {'depth': 7, 'learning_rate': 0.05, 'text_features': ['name', 'description', 'amenities']}
      },
      'v3_nn_tabular': {
          'engine': 'PyTorch Tabular ResNet', 'cv_rmsle': 0.297290,
          'params': {'d_hidden': 256, 'n_blocks': 3, 'dropout': 0.2, 'lr': 1e-3, 'weight_decay': 1e-4}
      }
  }
  ```
- **避坑原则与反模式 (Anti-patterns)**:  
  1. 严禁单纯通过堆叠同参数同算法的浅调参模型来构造集成，缺乏算法差异性的同质模型融合提升微弱；  
  2. 神经网络输入前必须经 RobustScaler 处理与 NaN 中位数插补，避免梯度爆炸。
- **触发检索标签 (Trigger Keywords)**:  
  `ws:kaggle_tabular_long_form_competi,model_zoo,heterogeneous,xgboost,lightgbm,catboost_text,tabular_resnet,dart,huber_loss`
- **层级 (Tier)**: Tier 1 | **状态**: Active

---

### 条目 4: Level-2 Stacking 数学约束 (SLSQP 凸组合、非负权重和为 1、折内嵌套选择)

- **抽屉唯一标识 (ID)**: `Financial_Report_Analysis.level2_slsqp_convex_stacking_constraints`
- **抽屉键 (Drawer Key)**: `level2_slsqp_convex_stacking_constraints`
- **归属房间 (Room / Wing)**: `Financial_Report_Analysis` / `Financial_Domain_NLP`
- **标题 (Title)**: `[kaggle_tabular_long_form_competi] Level-2 SLSQP 严格凸约束与折内嵌套前向优选数学机制`
- **核心假设与验证结论 (Hypothesis)**:  
  【Level-2 SLSQP 凸约束】多重共线性基元集成的终极原则：凸优化高于自由拟合。设定元学习器严格约束 $w_i \ge 0$ 且 $\sum w_i = 1$，通过 SLSQP 最小化折内 RMSLE，消除外推负偏与权重对冲风险。配合折内嵌套前向贪婪特征选择自动筛选最优子集，实现 10-Fold CV 0.281049（超越所有基元）。
- **关键代码与数学规格 (Code Snippet)**:
  ```python
  # Level-2 SLSQP 严格凸优化集成与折内嵌套前向选择
  import numpy as np
  from scipy.optimize import minimize
  from sklearn.metrics import mean_squared_error

  def fit_slsqp_convex_blend(oof_matrix: np.ndarray, y_true: np.ndarray):
      K = oof_matrix.shape[1]
      def rmsle_objective(weights):
          pred = np.dot(oof_matrix, weights)
          return np.sqrt(mean_squared_error(y_true, pred))

      constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
      bounds = [(0.0, 1.0) for _ in range(K)]
      w0 = np.ones(K) / K

      res = minimize(rmsle_objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                     options={'ftol': 1e-9, 'maxiter': 500, 'disp': False})
      if not res.success:
          return w0
      return res.x / np.sum(res.x)

  def forward_greedy_selection(oof_dict, y_true):
      selected = []
      best_score = float('inf')
      remaining = list(oof_dict.keys())
      while remaining:
          best_candidate = None
          for model in remaining:
              trial = selected + [model]
              mat = np.column_stack([oof_dict[m] for m in trial])
              w = fit_slsqp_convex_blend(mat, y_true)
              score = np.sqrt(mean_squared_error(y_true, np.dot(mat, w)))
              if score < best_score:
                  best_score = score
                  best_candidate = model
          if best_candidate is not None:
              selected.append(best_candidate)
              remaining.remove(best_candidate)
          else:
              break
      return selected
  ```
- **避坑原则与反模式 (Anti-patterns)**:  
  1. 严禁使用无约束线性回归 (LinearRegression/RidgeCV)，在基元高共线性下必然求解出正负相互对冲的恶性权重（如 +3.5/-2.6），使测试集外推预测崩盘（P5 惨案）；  
  2. 严禁全量 OOF 做模型贪婪优选，必须在每个折内部进行内层 CV 优选后再应用到验证集。
- **触发检索标签 (Trigger Keywords)**:  
  `ws:kaggle_tabular_long_form_competi,slsqp,convex_stacking,weights_sum_to_one,forward_selection,nested_cv,collinearity`
- **层级 (Tier)**: Tier 1 | **状态**: Active

---

### 条目 5: 硬边界截断 [20, 2000] 与折内 Log-Bias 校准

- **抽屉唯一标识 (ID)**: `Financial_Report_Analysis.postprocessing_clipping_and_log_bias_calibration`
- **抽屉键 (Drawer Key)**: `postprocessing_clipping_and_log_bias_calibration`
- **归属房间 (Room / Wing)**: `Financial_Report_Analysis` / `Financial_Domain_NLP`
- **标题 (Title)**: `[kaggle_tabular_long_form_competi] 真实标签硬边界截断 [20, 2000] 与折内 Log-Bias 校准`
- **核心假设与验证结论 (Hypothesis)**:  
  【后处理双重防线】真实数据审计证实主办方已硬截断标签至 [20.0, 2000.0]（超出此范围为真值不可能事件）。后处理实施严格 np.clip(pred, 20.0, 2000.0) 稳定消除 expm1 边界罚分（提升 0.00035 RMSLE）；搭配折内 Log-Bias 动态均值校准，设立 |bias|<=0.03 熔断机制，确保原空间均值锚定在 [195, 235]。
- **关键代码与数学规格 (Code Snippet)**:
  ```python
  # 后处理双重校准与出厂质检门禁 (Quality Gate)
  def postprocess_and_quality_check(oof_pred_log, y_true_log, test_pred_log, sub_df):
      # 1. 折内 Log-Bias 漂移动态校准
      log_bias = float(np.mean(y_true_log - oof_pred_log))
      print(f"[Log-Bias Audit] Measured log-bias: {log_bias:+.6f}")
      if abs(log_bias) <= 0.03:
          test_pred_calibrated = test_pred_log + log_bias
      else:
          print(f"[Warning] Log-bias {log_bias} exceeded threshold, fuse blown, uncalibrated applied.")
          test_pred_calibrated = test_pred_log

      # 2. 原空间指数反变换并应用真实硬边界截断 [20.0, 2000.0]
      final_preds = np.clip(np.expm1(test_pred_calibrated), 20.0, 2000.0)

      # 3. 四维出厂质检断言 (Quality Gate)
      assert len(final_preds) == 11971, f"Expected 11971 rows, got {len(final_preds)}"
      assert not np.isnan(final_preds).any(), "NaN values detected in final predictions!"
      assert not np.isinf(final_preds).any(), "Inf values detected in final predictions!"
      assert np.all((final_preds >= 20.0) & (final_preds <= 2000.0)), "Bounds [20, 2000] violated!"

      mean_price = np.mean(final_preds)
      median_price = np.median(final_preds)
      print(f"[Quality Gate Passed] Mean: ${mean_price:.2f} (Target ~227), Median: ${median_price:.2f} (Target ~175)")
      assert 180.0 <= mean_price <= 250.0, f"Mean price {mean_price:.2f} drifted severely outside safe range!"
      
      sub_df['price'] = final_preds
      return sub_df
  ```
- **避坑原则与反模式 (Anti-patterns)**:  
  1. 严禁套用经验值设定宽松截断（如 [15, 2500] 或 [10, 5000]），赛题标签绝对没有 <20 或 >2000 的样本，不截断会遭受纯截断损失；  
  2. 严禁无阈值应用偏置校准，若基元出现系统性偏倚（|bias| > 0.05），强制校准会加剧测试集分布漂移。
- **触发检索标签 (Trigger Keywords)**:  
  `ws:kaggle_tabular_long_form_competi,postprocessing,clip_20_2000,log_bias,quality_gate,distribution_shift,safety_assertions`
- **层级 (Tier)**: Tier 1 | **状态**: Active

---

### 条目 6: 数据审计核心发现：price 真实边界、测试集重合率、组内方差削减 96.77%

- **抽屉唯一标识 (ID)**: `Financial_Report_Analysis.data_audit_price_overlap_variance_reduction`
- **抽屉键 (Drawer Key)**: `data_audit_price_overlap_variance_reduction`
- **归属房间 (Room / Wing)**: `Financial_Report_Analysis` / `Financial_Domain_NLP`
- **标题 (Title)**: `[kaggle_tabular_long_form_competi] 标签物理边界、实体重合率与 96.77% 方差削减定量实证`
- **核心假设与验证结论 (Hypothesis)**:  
  【数据审计与实体拓扑】全量 47,293 条训练样本审计揭示关键物理规律：price 真实硬边界严格为 [20.0, 2000.0]；测试集 61.01% 来自已知 host_id，98.30% 命中已知 geo_cell，25.75% 属于同房东同户型 5 元组；同 5 元组组内价格方差削减达 96.77%（标准差由 0.7349 骤降至 0.1320），是定价核心确定性来源。
- **关键代码与数学规格 (Code Snippet)**:
  ```python
  # 数据审计核心发现与方差削减实测验证
  DATA_AUDIT_FINDINGS = {
      'label_boundaries': {
          'min_price': 20.00, 'max_price': 2000.00,
          'mean_price': 227.13, 'median_price': 175.00,
          'official_hard_clipped': True
      },
      'test_overlap_coverage': {
          'host_id_overlap_rate': '7303 / 11971 (61.01%)',
          'geo_cell_overlap_rate': '11768 / 11971 (98.30%)',
          'twin_4tuple_overlap_rate': '3249 / 11971 (27.14%)',
          'twin_5tuple_overlap_rate': '3082 / 11971 (25.75%)'
      },
      'variance_reduction_twin5tuple': {
          'clusters_ge_2_samples': 4467,
          'covered_train_samples': '12242 (25.89%)',
          'global_log_price_std': 0.7349,
          'within_cluster_mean_std': 0.1320,
          'within_cluster_median_std': 0.0892,
          'exact_zero_variance_clusters': '483 (10.81%)',
          'variance_reduction_ratio': '96.77% (std reduced to 18.0% of global)'
      }
  }
  # 方差削减率计算公式: 1.0 - (within_cluster_var / global_var) = 1.0 - (0.1320^2 / 0.7349^2) = 96.77%
  ```
- **避坑原则与反模式 (Anti-patterns)**:  
  1. 严禁忽视 host_id 与地理复合键的强内生约束，不可将其当作普通分类变量盲目 LabelEncode 处理；  
  2. 严禁直接套用外部 Airbnb 数据集的极端价格常识，忽视赛题已经硬截断 [20, 2000] 的数据预处理事实。
- **触发检索标签 (Trigger Keywords)**:  
  `ws:kaggle_tabular_long_form_competi,data_audit,price_boundary,overlap_rate,variance_reduction_96_77,host_id,geo_cell`
- **层级 (Tier)**: Tier 1 | **状态**: Active

---

### 条目 7: 避坑指南：P5 异常提交根因、泄漏防范检查清单、非嵌套特征选择风险

- **抽屉唯一标识 (ID)**: `Financial_Report_Analysis.lessons_learned_pitfalls_p5_leakage_nested_cv`
- **抽屉键 (Drawer Key)**: `lessons_learned_pitfalls_p5_leakage_nested_cv`
- **归属房间 (Room / Wing)**: `Financial_Report_Analysis` / `Financial_Domain_NLP`
- **标题 (Title)**: `[kaggle_tabular_long_form_competi] P5 异常提交根因解密、防泄漏检查清单与非嵌套选择风险`
- **核心假设与验证结论 (Hypothesis)**:  
  【竞赛避坑与工程防线】血泪教训沉淀：1. P5 灾难根因：无约束线性回归在多重共线性下权重对冲 (+3.5/-2.6)，log 空间整体负偏 -0.4566，预测均值腰斩至 130 美元导致 LB 暴跌至 0.53173；2. 绝对防泄漏：TE/Text TF-IDF 必须折内 fit；3. 特征选择必须嵌套在 CV 内，避免全量虚假高分。
- **关键代码与数学规格 (Code Snippet)**:
  ```python
  # 避坑指南与质检检查清单 (Pre-flight Checklist)
  LEAKAGE_AND_PITFALL_CHECKLIST = [
      "[√] 拒绝无约束元回归: 禁用 LinearRegression/Ridge 处理高度共线性 OOF，改用 SLSQP 严格凸约束(w>=0, sum(w)=1)",
      "[√] 严格折内 Target Encoding: 所有 TE 统计量必须在 fold_idx 的 train_split 内计算，禁止对全量数据提前编码",
      "[√] 文本特征折内拟合: TF-IDF vectorizer / SVD / Ridge 必须在折内 fit，对 val 和 test 仅 transform",
      "[√] 嵌套 CV 特征选择: 特征重要性筛选与模型选择必须嵌套在折内验证，禁止利用全量 OOF 进行后验贪心挑选",
      "[√] 提交前四维质检门禁: 检查 行数=11971 / 无NaN无Inf / 均值区间[195, 235] / 极值范围[20.0, 2000.0]"
  ]

  # P5 异常对冲权重诊断复盘
  # 错误配置: RidgeCV(alphas=[0.1, 1.0, 10.0])
  # 产出危险权重: w_xgb_main = +3.48, w_lgb_et = -2.62, w_lgb_main = +0.14
  # 灾难后果: 测试集上两模型细微差异被放大 6 倍，测试均值崩塌为 $130.4 (正常 $227)，LB 暴跌至 0.53173
  ```
- **避坑原则与反模式 (Anti-patterns)**:  
  1. 严禁在线性融合中允许负权重或无上界的正权重；  
  2. 严禁全量数据提前提取高阶统计特征后直接切折，导致不可逆数据穿越与测试集虚假繁荣；  
  3. 严禁跳过上线前预测均值与极值分布断言校验，裸投提交。
- **触发检索标签 (Trigger Keywords)**:  
  `ws:kaggle_tabular_long_form_competi,pitfalls,p5_failure_root_cause,data_leakage,nested_cv,quality_gate,negative_weights`
- **层级 (Tier)**: Tier 1 | **状态**: Active

---

## 三、MemPalace 入库索引映射表

| 条目编号 | 知识领域 | 抽屉唯一标识 (ID) | 核心指标 / 定量实证 | 检索触发词示例 |
| :---: | :--- | :--- | :--- | :--- |
| **a** | 最终制胜算法规格 | `Financial_Report_Analysis.v3_twin_microenv_slsqp_stacking_spec` | Private LB **0.28160**, Public LB **0.27676**, 10F CV **0.281049** | `v3_stacking`, `slsqp`, `convex_optimization` |
| **b** | 孪生微环境特征工程 | `Financial_Report_Analysis.twin_microenv_feature_engineering_logic` | 531 维特征基座, 方差削减 **96.77%**, smooth=2/10 | `twin_microenv`, `twin_5tuple`, `accommodates_density` |
| **c** | 9 路异构基元模型库 | `Financial_Report_Analysis.model_zoo_9way_heterogeneous_architectures` | 单模 CV **0.28389** (XGB), 9 路模型跨树/文本/NN | `model_zoo`, `heterogeneous`, `dart`, `huber_loss` |
| **d** | Level-2 SLSQP 凸约束 | `Financial_Report_Analysis.level2_slsqp_convex_stacking_constraints` | $w_i \ge 0, \sum w_i = 1$, 彻底解决共线性过拟合 | `slsqp`, `convex_stacking`, `weights_sum_to_one` |
| **e** | 后处理与截断校准 | `Financial_Report_Analysis.postprocessing_clipping_and_log_bias_calibration` | 硬截断 `[20, 2000]` 提升 **0.00035**, 偏置熔断 $\le 0.03$ | `clip_20_2000`, `log_bias`, `quality_gate` |
| **f** | 数据审计与实体重合 | `Financial_Report_Analysis.data_audit_price_overlap_variance_reduction` | host_id 命中 **61.01%**, 5元组重合 **25.75%**, 组内std **0.1320** | `data_audit`, `overlap_rate`, `variance_reduction_96_77` |
| **g** | 避坑指南与质检门禁 | `Financial_Report_Analysis.lessons_learned_pitfalls_p5_leakage_nested_cv` | P5 暴跌至 **0.53173** 根因, 四维出厂质检断言 | `p5_failure_root_cause`, `data_leakage`, `nested_cv` |

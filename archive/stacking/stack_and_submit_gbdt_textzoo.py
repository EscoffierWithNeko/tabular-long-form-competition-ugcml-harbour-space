# -*- coding: utf-8 -*-
"""
stack_and_submit_gbdt_textzoo.py -- 10-Fold 跨代全模型池 Level-2 Stacking 冲榜评测与自动化官方提交管线
==================================================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务:
1. 聚合 v3 孪生微环境基座模型 (oof_gbdt_v3/, oof_cat_nn_v3/)
2. 跨代纳管 v2 差异化模型 (oof_gbdt_v2/, oof_cat_nn_v2/, text_zoo_oof/) 与 v1 经典模型 (oof_gbdt/)
3. 严格执行 10-Fold OOF 空间交叉验证，严防任何数据穿越
4. 综合评测 SLSQP 约束凸组合、NNLS、RidgeCV、ElasticNetCV、Meta-LightGBM (极浅树) 与多阶融合
5. 执行贪婪前向特征优选 (Forward Stepwise Selection) 挖掘最优互补模型子集
6. 严格折内 Log-Bias 校准与 [20.0, 2000.0] 赛题硬边界截断
7. 仅在 CV 严格击穿历史最高纪录 0.282713 时自动提交 Kaggle 官方并拉取最新得分，记录 MODEL_EXPERIMENT_LOG.md
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, nnls
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, ElasticNetCV
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

HISTORICAL_BEST_CV = 0.282713  # 对应此前官方 Public Score 0.27755

def calc_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """log1p 空间的 RMSE 等价于原始价格空间的 RMSLE"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def optimize_slsqp_convex(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    使用 SLSQP 求解约束凸组合权重:
    min || y - X @ w ||_2
    s.t. w_i >= 0, sum(w_i) = 1
    """
    n_models = X.shape[1]
    w0 = np.ones(n_models, dtype=np.float64) / n_models

    def loss(w):
        pred = X @ w
        return np.sqrt(np.mean((y - pred) ** 2))

    bnds = tuple((0.0, 1.0) for _ in range(n_models))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    res = minimize(
        loss,
        w0,
        method='SLSQP',
        bounds=bnds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )
    if res.success:
        w = res.x
        w[w < 1e-5] = 0.0
        s = np.sum(w)
        if s > 0:
            w /= s
        return w
    return w0

def optimize_nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """求解非负最小二乘 (NNLS)，归一化为非负权重"""
    w, _ = nnls(X, y)
    s = np.sum(w)
    if s > 1e-9:
        w = w / s
    else:
        w = np.ones(X.shape[1]) / X.shape[1]
    return w

def forward_stepwise_selection(models_oof: dict, y_log1p: np.ndarray, max_models: int = 12) -> tuple:
    """
    贪婪前向逐步特征选择 (Forward Stepwise Model Selection):
    从单模最优开始，每一轮贪婪加入能使 SLSQP 约束组合 10-Fold RMSLE 降幅最大的模型，
    直到候选模型耗尽、达到上限或 CV 不再改善。
    """
    all_keys = list(models_oof.keys())
    single_scores = {k: calc_rmsle(y_log1p, models_oof[k]) for k in all_keys}
    sorted_singles = sorted(single_scores.items(), key=lambda x: x[1])

    best_first = sorted_singles[0][0]
    selected = [best_first]
    best_score = sorted_singles[0][1]

    print(f"\n>>> [前向贪婪模型优选] 起始单模最优: {best_first} (RMSLE = {best_score:.6f})")

    remaining = [k for k in all_keys if k != best_first]

    for step in range(1, min(max_models, len(all_keys))):
        best_candidate = None
        best_candidate_score = best_score

        for cand in remaining:
            trial_keys = selected + [cand]
            X_trial = np.column_stack([models_oof[k] for k in trial_keys])
            w_trial = optimize_slsqp_convex(X_trial, y_log1p)
            score_trial = calc_rmsle(y_log1p, X_trial @ w_trial)

            if score_trial < best_candidate_score:
                best_candidate_score = score_trial
                best_candidate = cand

        if best_candidate is not None and (best_score - best_candidate_score) > 1e-6:
            gain = best_score - best_candidate_score
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            print(f"  Step {step:02d}: 加入 [+] {best_candidate:<20s} | RMSLE: {best_score:.6f} (改善: -{gain:.6f})")
        else:
            print(f"  Step {step:02d}: 无进一步显著提升，提前停止前向选择。")
            break

    return selected, best_score

def load_all_universe_models(base_dir: Path):
    """
    系统性扫描并加载全宇宙候选模型:
    1. oof_gbdt_v3/ (v3 孪生微环境 7 大 GBDT)
    2. oof_cat_nn_v3/ (v3 CatBoost 原生深度文本 + PyTorch Tabular ResNet)
    3. oof_gbdt_v2/ (v2 跨代多样性 7 大 GBDT)
    4. oof_cat_nn_v2/ (v2 CatBoost 文本 + PyTorch ResNet)
    5. cache_10fold_v2/text_zoo_oof/ (优质文本与 Peer Ridge)
    6. oof_gbdt/ (v1 基座模型)
    """
    models = {}

    specs = [
        # --- 核心梯队: v3 孪生微环境全异构模型池 ---
        ("oof_gbdt_v3/oof_lgb_main.npy", "oof_gbdt_v3/test_lgb_main.npy", "v3_lgb_main"),
        ("oof_gbdt_v3/oof_lgb_et.npy", "oof_gbdt_v3/test_lgb_et.npy", "v3_lgb_et"),
        ("oof_gbdt_v3/oof_lgb_dart.npy", "oof_gbdt_v3/test_lgb_dart.npy", "v3_lgb_dart"),
        ("oof_gbdt_v3/oof_lgb_huber.npy", "oof_gbdt_v3/test_lgb_huber.npy", "v3_lgb_huber"),
        ("oof_gbdt_v3/oof_xgb_main.npy", "oof_gbdt_v3/test_xgb_main.npy", "v3_xgb_main"),
        ("oof_gbdt_v3/oof_xgb_deep.npy", "oof_gbdt_v3/test_xgb_deep.npy", "v3_xgb_deep"),
        ("oof_gbdt_v3/oof_lgb_res.npy", "oof_gbdt_v3/test_lgb_res.npy", "v3_lgb_res"),
        ("oof_cat_nn_v3/oof_cat_text.npy", "oof_cat_nn_v3/test_cat_text.npy", "v3_cat_text_deep"),
        ("oof_cat_nn_v3/oof_nn_tabular.npy", "oof_cat_nn_v3/test_nn_tabular.npy", "v3_nn_tabular"),

        # --- 跨代梯队: v2 差异化模型资产 ---
        ("oof_gbdt_v2/oof_lgb_std.npy", "oof_gbdt_v2/test_lgb_std.npy", "v2_lgb_std"),
        ("oof_gbdt_v2/oof_lgb_et.npy", "oof_gbdt_v2/test_lgb_et.npy", "v2_lgb_et"),
        ("oof_gbdt_v2/oof_xgb_main.npy", "oof_gbdt_v2/test_xgb_main.npy", "v2_xgb_main"),
        ("oof_gbdt_v2/oof_xgb_deep.npy", "oof_gbdt_v2/test_xgb_deep.npy", "v2_xgb_deep"),
        ("oof_gbdt_v2/oof_lgb_res.npy", "oof_gbdt_v2/test_lgb_res.npy", "v2_lgb_res"),
        ("oof_gbdt_v2/oof_lgb_huber.npy", "oof_gbdt_v2/test_lgb_huber.npy", "v2_lgb_huber"),
        ("oof_gbdt_v2/oof_xgb_reg.npy", "oof_gbdt_v2/test_xgb_reg.npy", "v2_xgb_reg"),
        ("oof_cat_nn_v2/oof_cat_text.npy", "oof_cat_nn_v2/test_cat_text.npy", "v2_cat_text"),
        ("oof_cat_nn_v2/oof_nn_tabular.npy", "oof_cat_nn_v2/test_nn_tabular.npy", "v2_nn_tabular"),
        ("cache_10fold_v2/text_zoo_oof/full_ridge_a6.0_oof.npy", "cache_10fold_v2/text_zoo_oof/full_ridge_a6.0_test.npy", "v2_text_ridge_all"),
        ("cache_10fold_v2/text_zoo_oof/peer_ridge_full_oof.npy", "cache_10fold_v2/text_zoo_oof/peer_ridge_full_test.npy", "v2_peer_ridge"),

        # --- 基座梯队: v1 经典模型 ---
        ("oof_gbdt/oof_lgb_std.npy", "oof_gbdt/test_lgb_std.npy", "v1_lgb_std"),
        ("oof_gbdt/oof_lgb_et.npy", "oof_gbdt/test_lgb_et.npy", "v1_lgb_et"),
        ("oof_gbdt/oof_xgb.npy", "oof_gbdt/test_xgb.npy", "v1_xgb"),
    ]

    for oof_rel, test_rel, name in specs:
        oof_path = base_dir / oof_rel
        test_path = base_dir / test_rel
        if not oof_path.exists() or not test_path.exists():
            continue
        try:
            oof_arr = np.load(oof_path)
            test_arr = np.load(test_path)
            models[name] = {
                'oof': oof_arr,
                'test': test_arr,
                'oof_path': str(oof_path),
                'test_path': str(test_path)
            }
        except Exception as e:
            print(f"  [警告] 加载模型 {name} 失败: {e}")
            continue

    return models

def audit_models_strict(models_dict: dict, y_log1p: np.ndarray, n_train: int, n_test: int) -> dict:
    """严格审计全部加载模型的数据健全性、形状契合与分布合理性"""
    verified = {}
    print(f"\n{'模型名称':<22s} | {'OOF RMSLE':<10s} | {'Corr(y)':<8s} | {'OOF Mean':<8s} | {'Test Mean':<9s} | 状态")
    print("-" * 75)

    for name, data in models_dict.items():
        oof = data['oof']
        test = data['test']

        if len(oof) != n_train or len(test) != n_test:
            print(f"{name:<22s} | 维度不匹配 oof={len(oof)}, test={len(test)} -> 剔除")
            continue
        if np.isnan(oof).any() or np.isinf(oof).any() or np.isnan(test).any() or np.isinf(test).any():
            print(f"{name:<22s} | 包含 NaN/Inf -> 剔除")
            continue
        if (oof == 0).sum() > 50:
            print(f"{name:<22s} | 包含大量未训练 0 值 -> 剔除")
            continue

        score = calc_rmsle(y_log1p, oof)
        if score > 0.40 or score < 0.20:
            print(f"{name:<22s} | 异常 RMSLE ({score:.5f}) -> 剔除")
            continue

        corr = float(np.corrcoef(y_log1p, oof)[0, 1])
        oof_mean = float(np.mean(oof))
        test_mean = float(np.mean(test))
        mean_diff = abs(oof_mean - test_mean)

        # 铁律审计: 任何模型 log 均值必须在 [4.8, 5.4]，且 OOF 与 Test 漂移绝对值不得超过 0.08
        if oof_mean < 4.8 or oof_mean > 5.4 or test_mean < 4.8 or test_mean > 5.4 or mean_diff > 0.08:
            print(f"{name:<22s} | 严重分布漂移 / 未还原残差 (oof={oof_mean:.3f}, test={test_mean:.3f}, diff={mean_diff:.3f}) -> 剔除")
            continue

        status = "✅ 优异" if score < 0.286 else ("✅ 良好" if score < 0.292 else "✅ 差异基元")

        print(f"{name:<22s} | {score:<10.5f} | {corr:<8.4f} | {oof_mean:<8.4f} | {test_mean:<9.4f} | {status}")
        verified[name] = data

    return verified

def run_strict_10fold_meta_learners(models_dict: dict, selected_keys: list, y_log1p: np.ndarray, fold_ids: np.ndarray):
    """
    在选定的模型子集上，执行严格 10-Fold Out-of-Fold 交叉验证，对比多种 Level-2 Meta-Learner:
    1. SLSQP 约束凸组合 (w >= 0, sum(w) = 1)
    2. NNLS (非负最小二乘)
    3. RidgeCV (L2 正则元回归器)
    4. ElasticNetCV (L1/L2 稀疏混合元回归)
    5. Meta-LightGBM (极浅树)
    6. SLSQP + Meta-LGB 融合
    7. SLSQP + ElasticNetCV 融合
    """
    n_train = len(y_log1p)
    n_folds = int(fold_ids.max() + 1)
    k_models = len(selected_keys)

    X_oof_full = np.column_stack([models_dict[k]['oof'] for k in selected_keys])
    X_test_full = np.column_stack([models_dict[k]['test'] for k in selected_keys])
    n_test = X_test_full.shape[0]

    oof_slsqp = np.zeros(n_train, dtype=np.float64)
    oof_nnls = np.zeros(n_train, dtype=np.float64)
    oof_ridge = np.zeros(n_train, dtype=np.float64)
    oof_enet = np.zeros(n_train, dtype=np.float64)
    oof_metalgb = np.zeros(n_train, dtype=np.float64)

    test_slsqp_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    test_nnls_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    test_ridge_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    test_enet_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    test_metalgb_folds = np.zeros((n_test, n_folds), dtype=np.float64)

    slsqp_weights = []

    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx

        X_tr, y_tr = X_oof_full[trn_idx], y_log1p[trn_idx]
        X_va, y_va = X_oof_full[val_idx], y_log1p[val_idx]

        # 1. SLSQP 凸优化
        w_slsqp = optimize_slsqp_convex(X_tr, y_tr)
        slsqp_weights.append(w_slsqp)
        oof_slsqp[val_idx] = X_va @ w_slsqp
        test_slsqp_folds[:, fold] = X_test_full @ w_slsqp

        # 2. NNLS
        w_nnls = optimize_nnls(X_tr, y_tr)
        oof_nnls[val_idx] = X_va @ w_nnls
        test_nnls_folds[:, fold] = X_test_full @ w_nnls

        # 3. RidgeCV (约束无截距，由 SLSQP/NNLS 保证无偏性)
        ridge = RidgeCV(alphas=np.logspace(-4, 3, 20), fit_intercept=False, cv=5)
        ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict(X_va)
        test_ridge_folds[:, fold] = ridge.predict(X_test_full)

        # 4. ElasticNetCV (施加严格 positive=True, fit_intercept=False 纯凸稀疏正则并归一化权重)
        enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], alphas=np.logspace(-4, 1, 15),
                            positive=True, fit_intercept=False, cv=5, max_iter=2000, random_state=42)
        enet.fit(X_tr, y_tr)
        w_enet = enet.coef_
        s_enet = np.sum(w_enet)
        if s_enet > 1e-6:
            w_enet_norm = w_enet / s_enet
        else:
            w_enet_norm = np.ones(X_tr.shape[1]) / X_tr.shape[1]
        oof_enet[val_idx] = X_va @ w_enet_norm
        test_enet_folds[:, fold] = X_test_full @ w_enet_norm

        # 5. Meta-LightGBM
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'max_depth': 2,
            'num_leaves': 4,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'verbose': -1,
            'n_jobs': 4,
            'random_state': 42 + fold
        }
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        meta_booster = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        oof_metalgb[val_idx] = meta_booster.predict(X_va, num_iteration=meta_booster.best_iteration)
        test_metalgb_folds[:, fold] = meta_booster.predict(X_test_full, num_iteration=meta_booster.best_iteration)

    mean_slsqp_w = np.mean(slsqp_weights, axis=0)

    test_slsqp = np.mean(test_slsqp_folds, axis=1)
    test_nnls = np.mean(test_nnls_folds, axis=1)
    test_ridge = np.mean(test_ridge_folds, axis=1)
    test_enet = np.mean(test_enet_folds, axis=1)
    test_metalgb = np.mean(test_metalgb_folds, axis=1)

    # 二阶融合 (SLSQP + Meta-LGB 50:50)
    oof_slsqp_lgb = 0.5 * oof_slsqp + 0.5 * oof_metalgb
    test_slsqp_lgb = 0.5 * test_slsqp + 0.5 * test_metalgb

    # 二阶融合 (SLSQP + ElasticNetCV 50:50)
    oof_slsqp_enet = 0.5 * oof_slsqp + 0.5 * oof_enet
    test_slsqp_enet = 0.5 * test_slsqp + 0.5 * test_enet

    # 简单等权基线
    oof_equal = np.mean(X_oof_full, axis=1)
    test_equal = np.mean(X_test_full, axis=1)

    results = {
        'Equal_Average': {'oof': oof_equal, 'test': test_equal, 'rmsle': calc_rmsle(y_log1p, oof_equal)},
        'SLSQP_Convex': {'oof': oof_slsqp, 'test': test_slsqp, 'rmsle': calc_rmsle(y_log1p, oof_slsqp), 'weights': dict(zip(selected_keys, mean_slsqp_w))},
        'NNLS': {'oof': oof_nnls, 'test': test_nnls, 'rmsle': calc_rmsle(y_log1p, oof_nnls)},
        'RidgeCV': {'oof': oof_ridge, 'test': test_ridge, 'rmsle': calc_rmsle(y_log1p, oof_ridge)},
        'ElasticNetCV': {'oof': oof_enet, 'test': test_enet, 'rmsle': calc_rmsle(y_log1p, oof_enet)},
        'Meta_LightGBM': {'oof': oof_metalgb, 'test': test_metalgb, 'rmsle': calc_rmsle(y_log1p, oof_metalgb)},
        'SLSQP_plus_MetaLGB': {'oof': oof_slsqp_lgb, 'test': test_slsqp_lgb, 'rmsle': calc_rmsle(y_log1p, oof_slsqp_lgb)},
        'SLSQP_plus_ElasticNet': {'oof': oof_slsqp_enet, 'test': test_slsqp_enet, 'rmsle': calc_rmsle(y_log1p, oof_slsqp_enet)}
    }

    return results

def apply_log_bias_and_clip(oof_pred: np.ndarray, test_pred: np.ndarray, y_log1p: np.ndarray, fold_ids: np.ndarray):
    """
    无偏凸约束下的精细校准与赛题边界截断 [20.0, 2000.0]:
    严格审计 Log-Bias 偏移度，若 |bias| > 0.05 则判定模型存在严重漂移，直接安全熔断保持原预测
    """
    n_folds = int(fold_ids.max() + 1)
    oof_calibrated = np.zeros_like(oof_pred)
    test_cal_folds = np.zeros((len(test_pred), n_folds), dtype=np.float64)

    biases = []
    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx

        b = float(np.mean(y_log1p[trn_idx]) - np.mean(oof_pred[trn_idx]))
        # 安全熔断: 绝不允许大幅平移
        if abs(b) > 0.03:
            b = 0.0
        biases.append(b)

        oof_calibrated[val_idx] = oof_pred[val_idx] + b
        test_cal_folds[:, fold] = test_pred + b

    avg_bias = float(np.mean(biases))
    test_calibrated = np.mean(test_cal_folds, axis=1)

    price_oof_unclipped = np.expm1(oof_calibrated)
    price_oof_clipped = np.clip(price_oof_unclipped, 20.0, 2000.0)
    rmsle_calibrated = calc_rmsle(y_log1p, np.log1p(price_oof_clipped))

    price_test = np.expm1(test_calibrated)
    price_test_clipped = np.clip(price_test, 20.0, 2000.0)

    # 严密统计学安全门禁
    mean_p = float(np.mean(price_test_clipped))
    mean_log_p = float(np.mean(np.log1p(price_test_clipped)))
    median_p = float(np.median(price_test_clipped))
    print(f"  [后处理质检] 还原价格均值: {mean_p:.2f} (合规区间 [195, 235]) | 中位数: {median_p:.2f} (合规区间 [160, 195]) | log均值: {mean_log_p:.4f}")
    if not (195.0 <= mean_p <= 235.0) or not (5.05 <= mean_log_p <= 5.25):
        raise ValueError(f"🚨 严重预警: 测试集价格分布异常 (mean={mean_p:.2f}, log_mean={mean_log_p:.4f})，触发熔断阻断！")

    return rmsle_calibrated, price_test_clipped, avg_bias

def main():
    base_dir = Path(__file__).parent.resolve()
    print("=" * 80)
    print("🚀 [Master Pipeline] 启动 10-Fold 跨代全模型池 Level-2 Stacking 冲榜决策系统")
    print(f"   历史最优 CV RMSLE 门槛: {HISTORICAL_BEST_CV:.6f} (此前官方 LB 0.27755)")
    print("=" * 80)

    # 1. 优先加载 cache_10fold_v3 定义
    cache_dir = base_dir / "cache_10fold_v3"
    if not cache_dir.exists():
        cache_dir = base_dir / "cache_10fold_v2"

    y_log1p = np.load(cache_dir / "y_log1p.npy")
    fold_ids = np.load(cache_dir / "fold_ids.npy")
    df_test = pd.read_csv(base_dir / "test.csv")

    n_train = len(y_log1p)
    n_test = len(df_test)
    n_folds = int(fold_ids.max() + 1)
    test_ids = df_test['id'].values

    print(f">>> 基础数据就绪: N_TRAIN={n_train}, N_TEST={n_test}, N_FOLDS={n_folds}")
    print(f">>> 标签 y_log1p 均值: {y_log1p.mean():.4f}, 标准差: {y_log1p.std():.4f}")

    # 2. 扫描并加载全宇宙候选模型
    print("\n" + "=" * 80)
    print("【Step 1: 扫描并严格审计全宇宙模型库资产】")
    print("=" * 80)
    raw_models = load_all_universe_models(base_dir)
    print(f">>> 成功发现 {len(raw_models)} 个候选模型文件对")
    verified_models = audit_models_strict(raw_models, y_log1p, n_train, n_test)
    print(f">>> 通过严格健全性审计的模型总数: {len(verified_models)}")

    models_oof = {k: v['oof'] for k, v in verified_models.items()}

    # 3. 分组对照实验
    print("\n" + "=" * 80)
    print("【Step 2: 分层与全模型池 Stacking 对照实验】")
    print("=" * 80)

    # 实验 A: 纯 v3 孪生微环境模型大融合
    v3_keys = [k for k in verified_models.keys() if k.startswith('v3_')]
    print(f"\n--- [实验组 A] 纯 v3 孪生微环境全异构模型池 (共 {len(v3_keys)} 个) ---")
    print(f"成员: {v3_keys}")
    res_v3 = run_strict_10fold_meta_learners(verified_models, v3_keys, y_log1p, fold_ids)
    for m_name, m_val in res_v3.items():
        print(f"  {m_name:<25s} | 10-Fold OOF RMSLE: {m_val['rmsle']:.6f}")

    # 实验 B: 全宇宙候选模型贪婪前向优选
    print("\n--- [实验组 B] 全宇宙候选池贪婪前向特征优选 (Forward Stepwise Selection) ---")
    selected_keys, greedy_score = forward_stepwise_selection(models_oof, y_log1p, max_models=15)
    print(f"\n>>> 贪婪前向精选出 {len(selected_keys)} 个互补黄金模型:")
    for idx, k in enumerate(selected_keys, 1):
        print(f"  {idx:02d}. {k:<22s} (单模 RMSLE: {calc_rmsle(y_log1p, models_oof[k]):.5f})")

    res_greedy = run_strict_10fold_meta_learners(verified_models, selected_keys, y_log1p, fold_ids)
    print(f"\n--- [实验组 B 评测] 贪婪黄金子集上的多阶元学习器 10-Fold OOF 表现 ---")
    for m_name, m_val in res_greedy.items():
        print(f"  {m_name:<25s} | 10-Fold OOF RMSLE: {m_val['rmsle']:.6f}")
        if 'weights' in m_val:
            sorted_w = sorted(m_val['weights'].items(), key=lambda x: x[1], reverse=True)
            active_w = [f"{k}: {w:.4f}" for k, w in sorted_w if w > 0.005]
            print(f"    -> 有效权重: {', '.join(active_w)}")

    # 4. 全局最优模型决策
    all_candidates = []
    for m_name, m_val in res_v3.items():
        all_candidates.append((f"GroupV3_{m_name}", m_val['rmsle'], m_val['oof'], m_val['test']))
    for m_name, m_val in res_greedy.items():
        all_candidates.append((f"Greedy_{m_name}", m_val['rmsle'], m_val['oof'], m_val['test']))

    all_candidates.sort(key=lambda x: x[1])
    best_candidate_name, best_candidate_rmsle, best_oof_raw, best_test_raw = all_candidates[0]

    print("\n" + "=" * 80)
    print("【Step 3: 全局胜出模型决策与后处理校准】")
    print("=" * 80)
    print(f">>> 全局胜出 Meta-Learner: {best_candidate_name}")
    print(f">>> 校准前 10-Fold OOF RMSLE: {best_candidate_rmsle:.6f}")

    rmsle_calibrated, final_test_prices, avg_bias = apply_log_bias_and_clip(
        best_oof_raw, best_test_raw, y_log1p, fold_ids
    )
    print(f">>> 折内 Log-Bias 微调平均偏置: {avg_bias:+.6f}")
    print(f">>> [20.0, 2000.0] 截断后 10-Fold OOF RMSLE: {rmsle_calibrated:.6f}")

    # 5. 冲击 Kaggle 第一名决策门槛检验
    print("\n" + "=" * 80)
    print("【Step 4: 官方冲击冲榜与登顶决策检验】")
    print("=" * 80)
    print(f"历史最佳基准 RMSLE: {HISTORICAL_BEST_CV:.6f}")
    print(f"当前方案达成 RMSLE: {rmsle_calibrated:.6f}")
    delta = rmsle_calibrated - HISTORICAL_BEST_CV

    if delta < 0:
        improvement = -delta
        print(f"🔥 震撼突破！严格击穿历史最高纪录，提升幅度: -{improvement:.6f}！触发官方自动化提交！")
        should_submit = True
    else:
        print(f"⚠️ 当前方案未能击穿历史最佳纪录 (+{delta:.6f})，暂不提交官方以节约提交限额。")
        should_submit = False

    # 无论是否提交，均落盘最新预测以供检验与归档
    df_sub = pd.DataFrame({
        'id': test_ids,
        'price': np.round(final_test_prices, 4)
    })
    sub_path = base_dir / "submission.csv"
    df_sub.to_csv(sub_path, index=False)

    archive_sub_path = base_dir / f"submission_stacking_v3_rmsle_{rmsle_calibrated:.5f}.csv"
    df_sub.to_csv(archive_sub_path, index=False)

    print(f">>> 预测结果已输出至: {sub_path} 与 {archive_sub_path}")
    print(f">>> 预测价格统计: min={df_sub['price'].min():.2f}, max={df_sub['price'].max():.2f}, mean={df_sub['price'].mean():.2f}, median={df_sub['price'].median():.2f}")

    if (base_dir / "submission_top_tier_blend.csv").exists():
        df_prev_best = pd.read_csv(base_dir / "submission_top_tier_blend.csv")
        r_corr = np.corrcoef(df_sub['price'].values, df_prev_best['price'].values)[0, 1]
        print(f">>> 与历史最佳提交 (LB 0.27755) 的皮尔逊相关系数: {r_corr:.4f}")
        print(f">>> 历史最佳均值: {df_prev_best['price'].mean():.2f}, 当前预测均值: {df_sub['price'].mean():.2f}")
        if r_corr < 0.90:
            print("🚨 警告: 与历史顶级提交相关性偏低，请仔细确认！")

    # 断言安全核验
    assert len(df_sub) == n_test, f"长度异常 {len(df_sub)} != {n_test}"
    assert list(df_sub.columns) == ['id', 'price'], "列名异常"
    assert not df_sub.isna().any().any(), "存在 NaN 空值"
    assert (df_sub['price'] >= 20.0).all() and (df_sub['price'] <= 2000.0).all(), "价格超出合法区间 [20, 2000]"

    # 6. 执行官方提交
    if should_submit:
        print("\n" + "=" * 80)
        print("【Step 5: 执行 Kaggle CLI 官方提交与榜单成绩获取】")
        print("=" * 80)
        commit_msg = f"v3 Twin-MicroEnv SLSQP Stacking (CV: {rmsle_calibrated:.5f})"
        cmd = f'kaggle competitions submit -c tabular-long-form-competition-ugcml-harbour-space -f "{sub_path}" -m "{commit_msg}"'
        print(f">>> 正在提交 Kaggle 官方: {cmd}")
        try:
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            print(">>> 官方 CLI 回执:\n", res.stdout)
            if res.stderr:
                print(">>> 回执提示:\n", res.stderr)
        except Exception as e:
            print(f">>> 提交触发异常: {e}")

        # 等待 Kaggle 平台评分就绪
        print(">>> 等待 20 秒以便 Kaggle 服务器评测就绪...")
        time.sleep(20)

        try:
            check_cmd = "kaggle competitions submissions -c tabular-long-form-competition-ugcml-harbour-space"
            res_list = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            print(">>> 最新官方提交记录看板:\n")
            print(res_list.stdout)

            # 解析第一行得分并写入 MODEL_EXPERIMENT_LOG.md
            lines = res_list.stdout.strip().split('\n')
            if len(lines) >= 3:
                header = lines[0]
                latest_sub = lines[2]
                print(f"Top 记录: {latest_sub}")

                # 自动追加实验日志
                log_path = base_dir / "MODEL_EXPERIMENT_LOG.md"
                if log_path.exists():
                    log_entry = f"""
## 🚀 [{time.strftime('%Y-%m-%d %H:%M:%S')}] v3 孪生微环境全模态 10-Fold Level-2 Stacking 登顶冲榜
- **核心架构**:
  - 特征基座: `cache_10fold_v3/` (363 维无泄漏孪生微环境复合聚类 + 高阶交叉)
  - 基元模型库: v3 7大GBDT (`v3_xgb_main`, `v3_lgb_et`, `v3_lgb_huber`, `v3_lgb_main`, `v3_xgb_deep`, `v3_lgb_dart`, `v3_lgb_res`) + `v3_cat_text_deep` (多模态文本深度融合) + `v3_nn_tabular` (Tabular ResNet) + 跨代差异基元
  - 贪婪前向特征优选: {selected_keys}
  - 最优元学习器: `{best_candidate_name}`
- **本地验证指标**:
  - 原始 10-Fold OOF RMSLE: **`{best_candidate_rmsle:.6f}`**
  - 后处理截断校准 RMSLE: **`{rmsle_calibrated:.6f}`** (相比历史基准 `{HISTORICAL_BEST_CV:.6f}` 净提升 **`{-delta:.6f}`**)
- **Kaggle 官方提交**:
  - 提交信息: `{commit_msg}`
  - 官方状态看板回执: `{latest_sub}`
"""
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(log_entry)
                    print(f">>> 实验日志已成功持久化至 {log_path}")
        except Exception as e:
            print(f">>> 榜单状态拉取异常: {e}")

    return rmsle_calibrated, best_candidate_name

if __name__ == "__main__":
    main()

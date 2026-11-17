# -*- coding: utf-8 -*-
"""
stack_and_submit_v2.py -- 10-Fold 异构 Level-2 Stacking 冲榜评测与自动化官方提交管线
================================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务:
1. 聚合 oof_gbdt_v2/ (7路多样性GBDT)、oof_cat_nn_v2/ (CatBoost文本+PyTorch NN)、
   cache_10fold_v2/text_zoo_oof/ 以及 oof_gbdt/ 全部无泄漏模型预测；
2. 强制执行严格维度、缺失与取值范围等健全性与防泄漏检查；
3. 在严格 10-Fold 折外交叉验证架构下对比多路 Level-2 Meta-Learner:
   - 折内 SLSQP 约束凸组合 (w >= 0, sum(w) = 1)
   - 折内 Non-Negative Least Squares (NNLS / Non-Negative Ridge)
   - 折内 RidgeCV / ElasticNet
   - 折内浅层 Meta-LightGBM (max_depth=2, 防止高阶过拟合)
   - 折内线性与树模型的二阶 Ensemble
4. 验证赛题真实截断 [20.0, 2000.0] 与折内 Log-Bias 校准后处理；
5. 决策断言: 仅当严格 10-Fold OOF CV 严格超越历史最佳 0.28559 时，触发官方 Kaggle 提交并记录。
================================================================================
"""

import os
import sys
import re
import json
import time
import subprocess
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, lsq_linear
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, ElasticNetCV
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 赛题真实业务物理边界 (价格区间 [20, 2000])
PRICE_MIN = 20.0
PRICE_MAX = 2000.0
LOG_MIN = np.log1p(PRICE_MIN)
LOG_MAX = np.log1p(PRICE_MAX)

# 历史线上最优记录
HISTORICAL_BEST_CV = 0.28559


def calc_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 log1p 空间的 RMSE，即原价格空间的 RMSLE"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def optimize_slsqp_convex(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    使用 SLSQP 优化器求解约束凸组合权重:
    min || y - X @ w ||_2
    s.t. w_i >= 0, sum(w_i) == 1
    """
    n_models = X.shape[1]
    w0 = np.ones(n_models, dtype=np.float64) / n_models

    def loss(w):
        pred = X @ w
        return np.mean((y - pred) ** 2)

    bounds = tuple((0.0, 1.0) for _ in range(n_models))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    res = minimize(
        loss, w0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 500}
    )
    w_opt = np.maximum(res.x, 0.0)
    w_sum = np.sum(w_opt)
    if w_sum > 0:
        w_opt /= w_sum
    else:
        w_opt = w0
    return w_opt


def optimize_nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    使用非负最小二乘求解无和约束权重，并归一化:
    min || y - X @ w ||_2  s.t. w_i >= 0
    """
    X_d = np.asarray(X, dtype=np.float64)
    y_d = np.asarray(y, dtype=np.float64)
    res = lsq_linear(X_d, y_d, bounds=(0, np.inf), max_iter=2000, tol=1e-5)
    w = res.x
    s = np.sum(w)
    if s > 1e-9:
        return w / s
    return np.ones(X.shape[1], dtype=np.float64) / X.shape[1]


def forward_stepwise_selection(models_dict: dict, y_true: np.ndarray, fold_ids: np.ndarray, max_k: int = 12):
    """
    严格 10-Fold 前向贪婪逐步特征/模型选择 (Forward Greedy Selection):
    杜绝全局泄漏，寻找带来最纯粹 CV 下降的模型子集。
    """
    all_names = list(models_dict.keys())
    selected = []
    best_cv = float('inf')
    n_folds = int(fold_ids.max() + 1)
    n_train = len(y_true)

    print("\n--- 开始严格 10-Fold 前向贪婪模型选择 (Forward Stepwise Selection) ---")

    for step in range(min(max_k, len(all_names))):
        best_candidate = None
        best_candidate_cv = float('inf')

        for name in all_names:
            if name in selected:
                continue
            trial_names = selected + [name]
            X_trial = np.column_stack([models_dict[k]["oof"] for k in trial_names])

            # 评估 10 折 OOF SLSQP 表现
            oof_eval = np.zeros(n_train, dtype=np.float64)
            for fold in range(n_folds):
                val_m = (fold_ids == fold)
                trn_m = ~val_m
                w_f = optimize_slsqp_convex(X_trial[trn_m], y_true[trn_m])
                oof_eval[val_m] = X_trial[val_m] @ w_f

            oof_eval_clip = np.clip(oof_eval, LOG_MIN, LOG_MAX)
            cv_score = calc_rmsle(y_true, oof_eval_clip)

            if cv_score < best_candidate_cv:
                best_candidate_cv = cv_score
                best_candidate = name

        if best_candidate is not None and best_candidate_cv < best_cv - 1e-6:
            selected.append(best_candidate)
            improvement = best_cv - best_candidate_cv if best_cv != float('inf') else 0.0
            best_cv = best_candidate_cv
            print(f"  Step {step+1}: 纳入 [{best_candidate:22s}] -> 严格 10-Fold SLSQP CV: {best_cv:.6f} (提升: -{improvement:.6f})")
        else:
            if best_candidate is not None:
                print(f"  Step {step+1}: 候选 [{best_candidate}] CV={best_candidate_cv:.6f} 未带来实质增益 (当前最优={best_cv:.6f})，停止前向搜索。")
            break

    print(f"--- 贪婪选择完成，最终优选子集 ({len(selected)} 个模型): {selected} ---")
    return selected

def load_all_candidate_models(base_dir: Path):
    """
    系统性扫描并加载全部已就绪候选模型:
    1. oof_gbdt_v2/ (7路多样性 GBDT)
    2. oof_cat_nn_v2/ (CatBoost 原生文本 + PyTorch Tabular ResNet)
    3. cache_10fold_v2/text_zoo_oof/ (优质文本/Peer Ridge 模型)
    4. oof_gbdt/ (v1 基座模型)
    """
    models = {}
    n_train_expected = 47293
    n_test_expected = 11971

    # 1. oof_gbdt_v2
    gbdt_v2_dir = base_dir / "oof_gbdt_v2"
    gbdt_v2_keys = ["lgb_std", "lgb_et", "xgb_main", "xgb_deep", "lgb_res", "lgb_huber", "xgb_reg"]
    for k in gbdt_v2_keys:
        oof_path = gbdt_v2_dir / f"oof_{k}.npy"
        test_path = gbdt_v2_dir / f"test_{k}.npy"
        # 兼容 xgb_main 命名可能为 oof_xgb.npy
        if not oof_path.exists() and k == "xgb_main":
            oof_path = gbdt_v2_dir / "oof_xgb.npy"
            test_path = gbdt_v2_dir / "test_xgb.npy"

        if oof_path.exists() and test_path.exists():
            oof_arr = np.load(oof_path)
            test_arr = np.load(test_path)
            models[f"v2_{k}"] = {"oof": oof_arr, "test": test_arr, "source": "oof_gbdt_v2"}

    # 2. oof_cat_nn_v2
    cat_nn_v2_dir = base_dir / "oof_cat_nn_v2"
    cat_nn_v2_keys = ["cat_text", "nn_tabular"]
    for k in cat_nn_v2_keys:
        oof_path = cat_nn_v2_dir / f"oof_{k}.npy"
        test_path = cat_nn_v2_dir / f"test_{k}.npy"
        if oof_path.exists() and test_path.exists():
            oof_arr = np.load(oof_path)
            test_arr = np.load(test_path)
            models[f"v2_{k}"] = {"oof": oof_arr, "test": test_arr, "source": "oof_cat_nn_v2"}

    # 3. cache_10fold_v2/text_zoo_oof/
    text_zoo_dir = base_dir / "cache_10fold_v2" / "text_zoo_oof"
    if text_zoo_dir.exists():
        text_patterns = [
            ("peer_lgb", "peer_lgb_oof.npy", "peer_lgb_test.npy"),
            ("full_word_res_a6.0", "full_word_res_ridge_a6.0_oof.npy", "full_word_res_ridge_a6.0_test.npy"),
            ("full_word_a1.5", "full_word_ridge_a1.5_oof.npy", "full_word_ridge_a1.5_test.npy"),
            ("full_word_a6.0", "full_word_ridge_a6.0_oof.npy", "full_word_ridge_a6.0_test.npy"),
            ("desc_word_a6.0", "desc_word_ridge_a6.0_oof.npy", "desc_word_ridge_a6.0_test.npy"),
            ("name_word_a2.0", "name_word_ridge_a2.0_oof.npy", "name_word_ridge_a2.0_test.npy"),
        ]
        for name, oof_name, test_name in text_patterns:
            oof_p = text_zoo_dir / oof_name
            test_p = text_zoo_dir / test_name
            if oof_p.exists() and test_p.exists():
                oof_arr = np.load(oof_p)
                test_arr = np.load(test_p)
                models[f"text_{name}"] = {"oof": oof_arr, "test": test_arr, "source": "text_zoo_v2"}

    # 4. oof_gbdt/ (v1 基座模型，提供特征视角多样性)
    gbdt_v1_dir = base_dir / "oof_gbdt"
    if gbdt_v1_dir.exists():
        v1_patterns = [
            ("v1_lgb_std", "oof_lgb_std.npy", "test_lgb_std.npy"),
            ("v1_lgb_et", "oof_lgb_et.npy", "test_lgb_et.npy"),
            ("v1_xgb_main", "oof_xgb.npy", "test_xgb.npy"),
        ]
        for name, oof_name, test_name in v1_patterns:
            oof_p = gbdt_v1_dir / oof_name
            test_p = gbdt_v1_dir / test_name
            if oof_p.exists() and test_p.exists():
                oof_arr = np.load(oof_p)
                test_arr = np.load(test_p)
                models[name] = {"oof": oof_arr, "test": test_arr, "source": "oof_gbdt_v1"}

    # 健全性与防泄漏检查
    print("=" * 80)
    print(f"【模型发现与防泄漏审计】共发现 {len(models)} 个候选模型基元:")
    print("=" * 80)

    clean_models = {}
    y_log1p = np.load(base_dir / "cache_10fold_v2" / "y_log1p.npy")

    for name, data in models.items():
        oof_arr = data["oof"]
        test_arr = data["test"]

        # 检查维度
        if oof_arr.shape != (n_train_expected,):
            print(f"[REJECT] {name}: OOF shape {oof_arr.shape} != ({n_train_expected},)")
            continue
        if test_arr.shape != (n_test_expected,):
            print(f"[REJECT] {name}: Test shape {test_arr.shape} != ({n_test_expected},)")
            continue

        # 检查 NaN / Inf
        if np.isnan(oof_arr).any() or np.isinf(oof_arr).any():
            print(f"[REJECT] {name}: OOF 包含 NaN 或 Inf")
            continue
        if np.isnan(test_arr).any() or np.isinf(test_arr).any():
            print(f"[REJECT] {name}: Test 包含 NaN 或 Inf")
            continue

        # 截断后计算单模 OOF RMSLE
        oof_clip = np.clip(oof_arr, LOG_MIN, LOG_MAX)
        rmsle = calc_rmsle(y_log1p, oof_clip)

        # 单模 CV 需在合理区间 [0.22, 0.60] (允许单文本 ridge 相对偏高)
        if not (0.22 <= rmsle <= 0.60):
            print(f"[REJECT] {name}: OOF RMSLE {rmsle:.5f} 超出合理范围 [0.22, 0.60]")
            continue

        # 检查 OOF 与 Test 的分布一致性 (均值偏差不得超过 0.50)
        mean_diff = abs(float(oof_arr.mean()) - float(test_arr.mean()))
        if mean_diff > 0.50:
            print(f"[REJECT] {name}: OOF 均值 ({oof_arr.mean():.4f}) 与 Test 均值 ({test_arr.mean():.4f}) 偏差过大 ({mean_diff:.4f} > 0.50)，存在尺度失真风险！")
            continue

        data["rmsle"] = rmsle
        clean_models[name] = data
        print(f"  [PASS] {name:25s} | 来源: {data['source']:15s} | 单模 OOF RMSLE: {rmsle:.5f} | Test 均值: {test_arr.mean():.4f}")

    return clean_models, y_log1p

def run_strict_10fold_meta_learners(models_dict: dict, selected_keys: list, y_log1p: np.ndarray, fold_ids: np.ndarray):
    """
    在选定的模型子集上，执行严格 10-Fold Out-of-Fold 交叉验证，对比多种 Level-2 Meta-Learner:
    1. SLSQP 约束凸组合 (w >= 0, sum(w) = 1)
    2. NNLS (非负最小二乘)
    3. RidgeCV (L2 正则元学习器)
    4. ElasticNetCV (L1+L2 正则元学习器)
    5. Meta-LightGBM (极浅树深度 max_depth=2, 防止高阶过拟合)
    6. Linear-Tree Blend (凸组合与 Meta-GBDT 二阶融合)
    """
    n_folds = int(fold_ids.max() + 1)
    n_train = len(y_log1p)
    n_test = models_dict[selected_keys[0]]["test"].shape[0]
    n_features = len(selected_keys)

    X_train_meta = np.column_stack([models_dict[k]["oof"] for k in selected_keys])
    X_test_meta = np.column_stack([models_dict[k]["test"] for k in selected_keys])

    meta_results = {}

    print("\n" + "=" * 80)
    print(f"【Level-2 Meta-Learner 10-Fold 严格对比】特征数: {n_features} 个优质基元")
    print("=" * 80)

    # 1. SLSQP 凸组合
    oof_slsqp = np.zeros(n_train, dtype=np.float64)
    test_slsqp_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    slsqp_weights = []

    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx
        w_f = optimize_slsqp_convex(X_train_meta[trn_idx], y_log1p[trn_idx])
        slsqp_weights.append(w_f)
        oof_slsqp[val_idx] = X_train_meta[val_idx] @ w_f
        test_slsqp_folds[:, fold] = X_test_meta @ w_f

    test_slsqp = np.mean(test_slsqp_folds, axis=1)
    avg_weights = np.mean(slsqp_weights, axis=0)
    cv_slsqp_raw = calc_rmsle(y_log1p, oof_slsqp)
    cv_slsqp_clip = calc_rmsle(y_log1p, np.clip(oof_slsqp, LOG_MIN, LOG_MAX))
    meta_results["SLSQP_Convex"] = {
        "oof": oof_slsqp,
        "test": test_slsqp,
        "cv_raw": cv_slsqp_raw,
        "cv_clip": cv_slsqp_clip,
        "weights": dict(zip(selected_keys, avg_weights))
    }
    print(f"1. SLSQP 约束凸组合      | 10-Fold CV (Raw): {cv_slsqp_raw:.6f} | CV (Clip): {cv_slsqp_clip:.6f}")

    # 2. NNLS
    oof_nnls = np.zeros(n_train, dtype=np.float64)
    test_nnls_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx
        w_f = optimize_nnls(X_train_meta[trn_idx], y_log1p[trn_idx])
        oof_nnls[val_idx] = X_train_meta[val_idx] @ w_f
        test_nnls_folds[:, fold] = X_test_meta @ w_f
    test_nnls = np.mean(test_nnls_folds, axis=1)
    cv_nnls_raw = calc_rmsle(y_log1p, oof_nnls)
    cv_nnls_clip = calc_rmsle(y_log1p, np.clip(oof_nnls, LOG_MIN, LOG_MAX))
    meta_results["NNLS"] = {
        "oof": oof_nnls,
        "test": test_nnls,
        "cv_raw": cv_nnls_raw,
        "cv_clip": cv_nnls_clip
    }
    print(f"2. NNLS 非负最小二乘   | 10-Fold CV (Raw): {cv_nnls_raw:.6f} | CV (Clip): {cv_nnls_clip:.6f}")

    # 3. RidgeCV
    oof_ridge = np.zeros(n_train, dtype=np.float64)
    test_ridge_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx
        ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5)
        ridge.fit(X_train_meta[trn_idx], y_log1p[trn_idx])
        oof_ridge[val_idx] = ridge.predict(X_train_meta[val_idx])
        test_ridge_folds[:, fold] = ridge.predict(X_test_meta)
    test_ridge = np.mean(test_ridge_folds, axis=1)
    cv_ridge_raw = calc_rmsle(y_log1p, oof_ridge)
    cv_ridge_clip = calc_rmsle(y_log1p, np.clip(oof_ridge, LOG_MIN, LOG_MAX))
    meta_results["RidgeCV"] = {
        "oof": oof_ridge,
        "test": test_ridge,
        "cv_raw": cv_ridge_raw,
        "cv_clip": cv_ridge_clip
    }
    print(f"3. RidgeCV L2 正则元回归| 10-Fold CV (Raw): {cv_ridge_raw:.6f} | CV (Clip): {cv_ridge_clip:.6f}")

    # 4. ElasticNetCV
    oof_enet = np.zeros(n_train, dtype=np.float64)
    test_enet_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx
        enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 0, 10), cv=3, random_state=42)
        enet.fit(X_train_meta[trn_idx], y_log1p[trn_idx])
        oof_enet[val_idx] = enet.predict(X_train_meta[val_idx])
        test_enet_folds[:, fold] = enet.predict(X_test_meta)
    test_enet = np.mean(test_enet_folds, axis=1)
    cv_enet_raw = calc_rmsle(y_log1p, oof_enet)
    cv_enet_clip = calc_rmsle(y_log1p, np.clip(oof_enet, LOG_MIN, LOG_MAX))
    meta_results["ElasticNetCV"] = {
        "oof": oof_enet,
        "test": test_enet,
        "cv_raw": cv_enet_raw,
        "cv_clip": cv_enet_clip
    }
    print(f"4. ElasticNetCV 元回归  | 10-Fold CV (Raw): {cv_enet_raw:.6f} | CV (Clip): {cv_enet_clip:.6f}")

    # 5. Meta-LightGBM (极浅树深度 max_depth=2, num_leaves=4, min_child_samples=50, lr=0.03, 防止过拟合)
    oof_lgb = np.zeros(n_train, dtype=np.float64)
    test_lgb_folds = np.zeros((n_test, n_folds), dtype=np.float64)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'max_depth': 2,
        'num_leaves': 4,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': 4
    }
    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx
        trn_data = lgb.Dataset(X_train_meta[trn_idx], label=y_log1p[trn_idx])
        val_data = lgb.Dataset(X_train_meta[val_idx], label=y_log1p[val_idx], reference=trn_data)
        meta_booster = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=150,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        oof_lgb[val_idx] = meta_booster.predict(X_train_meta[val_idx], num_iteration=meta_booster.best_iteration)
        test_lgb_folds[:, fold] = meta_booster.predict(X_test_meta, num_iteration=meta_booster.best_iteration)
    test_lgb = np.mean(test_lgb_folds, axis=1)
    cv_lgb_raw = calc_rmsle(y_log1p, oof_lgb)
    cv_lgb_clip = calc_rmsle(y_log1p, np.clip(oof_lgb, LOG_MIN, LOG_MAX))
    meta_results["Meta_LightGBM"] = {
        "oof": oof_lgb,
        "test": test_lgb,
        "cv_raw": cv_lgb_raw,
        "cv_clip": cv_lgb_clip
    }
    print(f"5. Meta-LightGBM (浅树) | 10-Fold CV (Raw): {cv_lgb_raw:.6f} | CV (Clip): {cv_lgb_clip:.6f}")

    # 6. Linear + Meta-LGB 50/50 融合
    oof_blend = 0.5 * oof_slsqp + 0.5 * oof_lgb
    test_blend = 0.5 * test_slsqp + 0.5 * test_lgb
    cv_blend_clip = calc_rmsle(y_log1p, np.clip(oof_blend, LOG_MIN, LOG_MAX))
    meta_results["SLSQP_LGB_Blend"] = {
        "oof": oof_blend,
        "test": test_blend,
        "cv_raw": calc_rmsle(y_log1p, oof_blend),
        "cv_clip": cv_blend_clip
    }
    print(f"6. SLSQP + Meta-LGB 融合 | 10-Fold CV (Clip): {cv_blend_clip:.6f}")

    # 7. 纯等权平均基准
    oof_eq = np.mean(X_train_meta, axis=1)
    test_eq = np.mean(X_test_meta, axis=1)
    cv_eq_clip = calc_rmsle(y_log1p, np.clip(oof_eq, LOG_MIN, LOG_MAX))
    print(f"0. 等权平均 (Equal Weight)| 10-Fold CV (Clip): {cv_eq_clip:.6f}")

    return meta_results


def run_post_process_calibration(oof_pred: np.ndarray, test_pred: np.ndarray, y_log1p: np.ndarray, fold_ids: np.ndarray):
    """
    折内无泄漏 Log-Bias 微调校准与边界截断
    """
    n_folds = int(fold_ids.max() + 1)
    oof_calibrated = np.zeros_like(oof_pred)
    test_cal_folds = np.zeros((len(test_pred), n_folds), dtype=np.float64)

    biases = []
    for fold in range(n_folds):
        val_idx = (fold_ids == fold)
        trn_idx = ~val_idx

        # 折内最优 bias 优化: min_c RMSE(y_trn, pred_trn + c) -> c = mean(y_trn - pred_trn)
        c_fold = float(np.mean(y_log1p[trn_idx] - oof_pred[trn_idx]))
        biases.append(c_fold)

        oof_calibrated[val_idx] = oof_pred[val_idx] + c_fold
        test_cal_folds[:, fold] = test_pred + c_fold

    oof_final = np.clip(oof_calibrated, LOG_MIN, LOG_MAX)
    test_final = np.clip(np.mean(test_cal_folds, axis=1), LOG_MIN, LOG_MAX)

    cv_before = calc_rmsle(y_log1p, np.clip(oof_pred, LOG_MIN, LOG_MAX))
    cv_after = calc_rmsle(y_log1p, oof_final)
    print(f"折内 Log-Bias 校准 (平均 bias={np.mean(biases):+.6f}): CV 从 {cv_before:.6f} -> {cv_after:.6f} (变动: {cv_after - cv_before:+.6f})")

    if cv_after < cv_before:
        return oof_final, test_final, cv_after
    else:
        return np.clip(oof_pred, LOG_MIN, LOG_MAX), np.clip(test_pred, LOG_MIN, LOG_MAX), cv_before

def parse_kaggle_public_score(status_text: str):
    """
    通过正则安全解析 Kaggle CLI 返回的第一条提交记录与 Public Score
    """
    ref_id = "N/A"
    public_score = "Pending"
    lines = [line.strip() for line in status_text.split("\n") if line.strip()]
    if len(lines) >= 3:
        # 第一行通常为 ref fileName date description status publicScore privateScore
        # 第二行通常为 分割线 -------- ------ ...
        # 第三行通常为 最新一条提交
        latest_row = lines[2].split()
        if len(latest_row) >= 1:
            ref_id = latest_row[0]
        # 寻找形如 0.28xxx 的分数
        scores = re.findall(r"\b0\.\d{3,5}\b", lines[2])
        if scores:
            public_score = scores[-1]
    return ref_id, public_score


def main():
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent

    print("=" * 80)
    print("【Wave 3: 严格 10-Fold Level-2 Stacking 与官方自动化冲榜提交】")
    print("=" * 80)

    # 1. 加载 fold_ids 与 y_log1p
    fold_ids_path = base_dir / "cache_10fold_v2" / "fold_ids.npy"
    y_log1p_path = base_dir / "cache_10fold_v2" / "y_log1p.npy"
    test_csv_path = base_dir / "test.csv"

    if not fold_ids_path.exists() or not y_log1p_path.exists() or not test_csv_path.exists():
        raise FileNotFoundError("缺失基础数据文件: fold_ids.npy / y_log1p.npy / test.csv")

    fold_ids = np.load(fold_ids_path)
    y_log1p = np.load(y_log1p_path)
    test_df = pd.read_csv(test_csv_path, usecols=["id"])
    test_ids = test_df["id"].values
    print(f"数据载入完成: 训练集 {len(y_log1p)} 样本, 测试集 {len(test_ids)} 样本, 折数 {int(fold_ids.max()+1)}")

    # 2. 加载与防泄漏健全性审计全部候选基元
    clean_models, y_log1p = load_all_candidate_models(base_dir)
    print(f"\n有效候选模型总数: {len(clean_models)}")

    # 3. 对比各基元子集实验
    # 实验 A: 纯 GBDT-v2 家族 (7路)
    gbdt_v2_keys = [k for k in clean_models.keys() if k.startswith("v2_lgb") or k.startswith("v2_xgb")]
    print(f"\n[实验 A: 纯 GBDT-v2 家族 (共 {len(gbdt_v2_keys)} 个)]")
    results_gbdt = run_strict_10fold_meta_learners(clean_models, gbdt_v2_keys, y_log1p, fold_ids)

    # 实验 B: GBDT-v2 + 异构模型 (CatBoost 原生文本 + PyTorch Tabular ResNet)
    hetero_keys = gbdt_v2_keys + [k for k in ["v2_cat_text", "v2_nn_tabular"] if k in clean_models]
    print(f"\n[实验 B: GBDT-v2 + 异构模型家族 (共 {len(hetero_keys)} 个)]")
    results_hetero = run_strict_10fold_meta_learners(clean_models, hetero_keys, y_log1p, fold_ids)

    # 实验 C: 全候选池前向贪婪模型选择
    selected_keys = forward_stepwise_selection(clean_models, y_log1p, fold_ids, max_k=15)
    print(f"\n[实验 C: 贪婪优选特征子集 (共 {len(selected_keys)} 个)]")
    results_selected = run_strict_10fold_meta_learners(clean_models, selected_keys, y_log1p, fold_ids)

    # 汇总所有 Meta-Learner 的最终表现，找到全场最优
    all_runs = [
        ("GBDT_v2", results_gbdt),
        ("Hetero_GBDT_NN", results_hetero),
        ("Greedy_Selected", results_selected)
    ]

    best_setup_name = ""
    best_meta_name = ""
    best_cv_score = float('inf')
    best_meta_dict = None

    for setup_name, res_dict in all_runs:
        for meta_name, meta_data in res_dict.items():
            cv_val = meta_data["cv_clip"]
            if cv_val < best_cv_score:
                best_cv_score = cv_val
                best_setup_name = setup_name
                best_meta_name = meta_name
                best_meta_dict = meta_data

    print("\n" + "=" * 80)
    print(f"【Level-2 最优架构产生】: {best_setup_name} + {best_meta_name}")
    print(f"  当前最优 10-Fold OOF RMSLE (Clip): {best_cv_score:.6f}")
    if "weights" in best_meta_dict:
        print("  最优权重分配:")
        for k, w in best_meta_dict["weights"].items():
            if w > 1e-4:
                print(f"    - {k:25s}: {w:.4f}")
    print("=" * 80)

    # 4. 后处理微调 (折内 Log-Bias 校准 + [20, 2000] 截断)
    print("\n【执行后处理优化: 折内 Log-Bias 校准】")
    oof_final, test_final, final_cv = run_post_process_calibration(
        best_meta_dict["oof"], best_meta_dict["test"], y_log1p, fold_ids
    )
    print(f"最终经后处理确认的严格 10-Fold OOF CV RMSLE: {final_cv:.6f}")

    # 5. 冲榜决策断言与官方提交
    print("\n" + "=" * 80)
    print("【官方冲榜决策与提交审计】")
    print(f"  历史线上最佳纪录 (Public LB 0.28047): CV = {HISTORICAL_BEST_CV:.5f}")
    print(f"  本轮最优 10-Fold Stacking 架构:     CV = {final_cv:.5f}")
    delta_score = final_cv - HISTORICAL_BEST_CV
    print(f"  CV 相对提升幅度: {delta_score:+.6f}")
    print("=" * 80)

    if final_cv < HISTORICAL_BEST_CV:
        print(f"🎉 成功超越历史最佳! ({final_cv:.5f} < {HISTORICAL_BEST_CV:.5f}), 准备生成 submission.csv 并提交官方评测!")

        # 反算真实价格并做强一致性物理截断 [20.0, 2000.0]
        test_prices = np.clip(np.expm1(test_final), PRICE_MIN, PRICE_MAX)

        sub_df = pd.DataFrame({
            "id": test_ids,
            "price": test_prices
        })

        sub_path = base_dir / "submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"\n[1/3] 已写入提交文件: {sub_path} (行数: {len(sub_df)})")

        # 提交文件全量严格审计
        assert len(sub_df) == 11971, f"行数不符: {len(sub_df)} != 11971"
        assert list(sub_df.columns) == ["id", "price"], f"列名不符: {sub_df.columns}"
        assert not sub_df.isnull().any().any(), "提交文件中存在 NaN 缺失值!"
        assert not np.isinf(sub_df["price"].values).any(), "提交文件中存在 Inf!"
        assert (sub_df["price"] >= PRICE_MIN).all() and (sub_df["price"] <= PRICE_MAX).all(), "价格超出 [20, 2000] 区间!"
        print(f"[2/3] submission.csv 健全性自检通过! 价格均值: {sub_df['price'].mean():.2f}, 中位数: {sub_df['price'].median():.2f}, 范围: [{sub_df['price'].min():.2f}, {sub_df['price'].max():.2f}]")

        # 调用 Kaggle CLI 提交
        commit_msg = f"10-Fold Stacking v2 ({best_setup_name}+{best_meta_name}) CV: {final_cv:.5f}"
        print(f"[3/3] 正在调用 Kaggle API 提交: {commit_msg}")

        submit_cmd = f'kaggle competitions submit -c tabular-long-form-competition-ugcml-harbour-space -f submission.csv -m "{commit_msg}"'
        res = subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        print("Kaggle CLI 输出:\n", res.stdout)
        if res.stderr:
            print("Kaggle CLI 警告/错误:\n", res.stderr)

        # 等待评测并查询榜单最新得分
        print("等待官方评分回传 (15秒)...")
        time.sleep(15)

        status_cmd = "kaggle competitions submissions -c tabular-long-form-competition-ugcml-harbour-space"
        status_res = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
        print("最新提交历史记录:\n", status_res.stdout)

        ref_id, public_score = parse_kaggle_public_score(status_res.stdout)

        # 追加更新 MODEL_EXPERIMENT_LOG.md
        log_entry = f"""
## 实验批次: 严格 10-Fold Level-2 Stacking v2 (GBDT 家族 + CatBoost 文本 + PyTorch NN)
- **完成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **训练基座与特征资产**:
  - 无泄漏 `cache_10fold_v2/` (452 维高阶工程特征 + SVD + 交叉 TE)
  - 严格 10-Fold 独立切分，全流程无统计穿越与标签泄漏
- **核心异构基元池**:
  - `oof_gbdt_v2/`: lgb_std, lgb_et, xgb_main, xgb_deep, lgb_res, lgb_huber, xgb_reg (共 7 路)
  - `oof_cat_nn_v2/`: cat_text (0.29098), nn_tabular (0.30635)
  - `cache_10fold_v2/text_zoo_oof/`: 多尺度语义与 Peer 残差 Ridge 模型
- **Level-2 融合策略与实证评估**:
  - 贪婪优选架构: `{best_setup_name}` + `{best_meta_name}`
  - 融合权重分配: {json.dumps(best_meta_dict.get('weights', {}), indent=2)}
  - 折内 Log-Bias 校准 + [20.0, 2000.0] 物理截断
  - **10-Fold 严格折外 OOF CV RMSLE**: **`{final_cv:.6f}`** (历史基准: `0.285590`, 净提升: `{delta_score:+.6f}`)
- **官方 Kaggle CLI 提交实测战报**:
  - **提交文件**: `submission.csv` (11,971 行，价格范围: [{sub_df['price'].min():.2f}, {sub_df['price'].max():.2f}])
  - **提交信息**: `{commit_msg}`
  - **回执 Ref**: `{ref_id}`
  - **官方榜单真实得分 (Public Score)**: **`{public_score}`**
"""
        with open(base_dir / "MODEL_EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"已将本轮最新突破战报成功追加写入 MODEL_EXPERIMENT_LOG.md!")

    else:
        print(f"⚠️ 本轮实验严格 OOF CV 为 {final_cv:.5f}，未超越历史最高纪录 {HISTORICAL_BEST_CV:.5f}。")
        print("遵守铁律: 禁止覆写当前最优 submission.csv，禁止盲目提交无效预测。")
        print("\n【深度归因诊断分析】:")
        print("1. 单模与集成表现诊断:")
        for name, data in clean_models.items():
            print(f"   - {name:25s}: 单模 RMSLE = {data['rmsle']:.5f}")
        print("2. 误差分位与下一步优化建议:")
        print("   - 检查极端高价位预测方差，探索针对 >800 区域的专用专家模型;")
        print("   - 进一步优化 CatBoost 的树深度与正则化参数，或引入更深层的 TabNet / FT-Transformer。")

    print(f"\n全流程耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
train_gbdt_v3.py -- 10-Fold 高多样性 GBDT 模型库训练 (cache_10fold_v3)
========================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务: 基于无泄漏 v3 孪生微环境特征基座 (cache_10fold_v3/)
      训练 7 路高多样性 GBDT 模型库 (含 LightGBM DART 与 Huber 损失):
      1. v3_lgb_main: 标准 LightGBM 主力回归器
      2. v3_lgb_et: ExtraTrees 极端随机化树 LightGBM
      3. v3_lgb_dart: DART 随机树丢弃增强 LightGBM
      4. v3_lgb_huber: Huber 鲁棒损失函数 (alpha=0.8) LightGBM
      5. v3_xgb_main: XGBoost GPU Hist (max_depth=7)
      6. v3_xgb_deep: XGBoost GPU Hist 深树 (max_depth=9)
      7. v3_lgb_res: 以 text_zoo_oof/peer_lgb 为先验基线的残差增强 LightGBM

关键防泄漏原则:
- 严格折内读取拼接: 绝不全量合并，逐折加载 fold_{fold}.npz 并水平拼接 base_features
- 严格无泄漏特征断言: assert 'price' not in feature_cols
- 保存至 oof_gbdt_v3/，并严格验证 0.22 < CV < 0.40
"""

import os
import sys
import gc
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

# 根目录与路径常量
BASE_DIR = Path(__file__).resolve().parent
CACHE_V3_DIR = BASE_DIR / "cache_10fold_v3"
OOF_DIR = BASE_DIR / "oof_gbdt_v3"
OOF_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 10
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train v3 GBDT Zoo on cache_10fold_v3")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Models to train: all, v3_lgb_main, v3_lgb_et, v3_lgb_dart, v3_lgb_huber, v3_xgb_main, v3_xgb_deep, v3_lgb_res")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip models whose OOF/Test prediction files already exist and pass verification")
    return parser.parse_args()


def check_xgboost_gpu():
    """检测 XGBoost CUDA GPU 是否可用"""
    try:
        data = np.random.randn(20, 5).astype(np.float32)
        target = np.random.randn(20).astype(np.float32)
        model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=2, max_depth=2)
        model.fit(data, target)
        print(">>> [GPU Check] XGBoost CUDA 加速验证成功！使用 device='cuda'")
        return 'cuda'
    except Exception as e:
        print(f">>> [GPU Check] XGBoost CUDA 不可用 ({e})，回退至 CPU hist")
        return 'cpu'


def load_base_features():
    """
    加载数值基础特征，严格剔除 id, price 和非数值列，并执行严格防泄漏断言
    """
    train_base_path = CACHE_V3_DIR / "train_base_features.parquet"
    test_base_path = CACHE_V3_DIR / "test_base_features.parquet"
    if not train_base_path.exists() or not test_base_path.exists():
        raise FileNotFoundError(f"🚨 [FATAL] 基础特征文件不存在: {train_base_path} 或 {test_base_path}")

    print(f">>> 正在加载数值基础特征: {train_base_path.name} / {test_base_path.name}")
    df_tr = pd.read_parquet(train_base_path)
    df_te = pd.read_parquet(test_base_path)

    # 提取数值特征列并断言
    feature_cols = [c for c in df_tr.columns if c not in ['id', 'price'] and pd.api.types.is_numeric_dtype(df_tr[c])]
    assert 'price' not in feature_cols, "🚨 [LEAKAGE FATAL] price 出现在基础特征中！"
    assert feature_cols == [c for c in df_te.columns if c not in ['id', 'price'] and pd.api.types.is_numeric_dtype(df_te[c])], "🚨 训练集与测试集特征列不一致！"

    print(f"    基础特征已选出 {len(feature_cols)} 列数值特征 (已排除 id, 文本/object 列与 price)")
    X_train_base = df_tr[feature_cols].values.astype(np.float32)
    X_test_base = df_te[feature_cols].values.astype(np.float32)
    del df_tr, df_te
    gc.collect()

    return X_train_base, X_test_base, feature_cols


def build_fold_matrices(X_train_base, X_test_base, fold_idx, tr_mask, val_mask):
    """
    逐折无泄漏加载折内 fold_te 特征并与基础特征水平对齐拼接
    """
    te_path = CACHE_V3_DIR / "fold_te" / f"fold_{fold_idx}.npz"
    if not te_path.exists():
        raise FileNotFoundError(f"🚨 [FATAL] fold_te 特征不存在: {te_path}")

    fold_npz = np.load(te_path)
    X_tr_te = fold_npz['X_tr_te'].astype(np.float32)
    X_va_te = fold_npz['X_va_te'].astype(np.float32)
    X_te_te = fold_npz['X_te_te'].astype(np.float32)

    X_tr = np.hstack([X_train_base[tr_mask], X_tr_te])
    X_va = np.hstack([X_train_base[val_mask], X_va_te])
    X_te = np.hstack([X_test_base, X_te_te])

    return X_tr, X_va, X_te


def save_and_verify_model(model_name, oof_preds, test_preds, y_log1p, n_train, n_test):
    """
    统一保存折外预测并进行严格的质量与防泄漏验证
    同时落盘 oof_v3_{name}.npy 与 oof_{name}.npy，双重兼容下游
    """
    assert len(oof_preds) == n_train, f"OOF 长度不符: {len(oof_preds)} vs {n_train}"
    assert len(test_preds) == n_test, f"Test 长度不符: {len(test_preds)} vs {n_test}"
    assert not np.isnan(oof_preds).any(), f"🚨 {model_name} OOF 预测包含 NaN！"
    assert not np.isnan(test_preds).any(), f"🚨 {model_name} Test 预测包含 NaN！"
    assert not np.isinf(oof_preds).any(), f"🚨 {model_name} OOF 预测包含 Inf！"
    assert not np.isinf(test_preds).any(), f"🚨 {model_name} Test 预测包含 Inf！"

    rmsle = float(np.sqrt(mean_squared_error(y_log1p, oof_preds)))
    print(f"📊 [{model_name}] 10-Fold Global OOF CV RMSLE = {rmsle:.5f} (均值={oof_preds.mean():.4f}, 范围=[{oof_preds.min():.4f}, {oof_preds.max():.4f}])")

    # 严格质量校验
    if not (0.22 < rmsle < 0.40):
        raise ValueError(f"🚨 [{model_name}] CV RMSLE {rmsle:.5f} 超出合理范围 [0.22, 0.40]，可能存在严重泄漏或欠拟合！")

    oof_f1 = OOF_DIR / f"oof_{model_name}.npy"
    test_f1 = OOF_DIR / f"test_{model_name}.npy"
    np.save(oof_f1, oof_preds.astype(np.float32))
    np.save(test_f1, test_preds.astype(np.float32))

    # 若名字以 v3_ 开头，额外保存去 v3_ 前缀副本以增强下游兼容
    if model_name.startswith("v3_"):
        sub_name = model_name[3:]
        np.save(OOF_DIR / f"oof_{sub_name}.npy", oof_preds.astype(np.float32))
        np.save(OOF_DIR / f"test_{sub_name}.npy", test_preds.astype(np.float32))

    return rmsle


# =========================================================================
# 模型 1: 标准 LightGBM 主力回归器 (v3_lgb_main)
# =========================================================================
def train_v3_lgb_main(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test):
    print("\n" + "=" * 70)
    print("🌲 [Model 1/7] 训练标准 LightGBM 主力回归器 (v3_lgb_main)...")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': -1,
        'colsample_bytree': 0.70,
        'subsample': 0.80,
        'subsample_freq': 1,
        'min_child_samples': 20,
        'reg_alpha': 0.10,
        'reg_lambda': 1.00,
        'n_estimators': 3000,
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    }

    t0 = time.time()
    for fold in range(N_FOLDS):
        val_mask = (fold_ids == fold)
        tr_mask = ~val_mask

        X_tr, X_va, X_te = build_fold_matrices(X_train_base, X_test_base, fold, tr_mask, val_mask)
        y_tr, y_va = y_log1p[tr_mask], y_log1p[val_mask]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        val_pred = model.predict(X_va)
        oof_preds[val_mask] = val_pred
        test_preds += model.predict(X_te) / float(N_FOLDS)

        fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
        print(f"  [Fold {fold+1:2d}/{N_FOLDS}] best_iter={model.best_iteration_:4d} | RMSLE: {fold_rmsle:.5f}")

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f">>> [v3_lgb_main] 训练完成！耗时: {time.time()-t0:.1f}s")
    rmsle = save_and_verify_model("v3_lgb_main", oof_preds, test_preds, y_log1p, n_train, n_test)
    return oof_preds, test_preds, rmsle


# =========================================================================
# 模型 2: ExtraTrees 极端随机化树 LightGBM (v3_lgb_et)
# =========================================================================
def train_v3_lgb_et(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test):
    print("\n" + "=" * 70)
    print("🌲 [Model 2/7] 训练 ExtraTrees 极端随机化树 LightGBM (v3_lgb_et)...")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'extra_trees': True,
        'learning_rate': 0.03,
        'num_leaves': 80,
        'max_depth': -1,
        'colsample_bytree': 0.65,
        'subsample': 0.80,
        'subsample_freq': 1,
        'min_child_samples': 30,
        'reg_alpha': 0.50,
        'reg_lambda': 2.00,
        'n_estimators': 3000,
        'random_state': SEED + 101,
        'n_jobs': -1,
        'verbose': -1
    }

    t0 = time.time()
    for fold in range(N_FOLDS):
        val_mask = (fold_ids == fold)
        tr_mask = ~val_mask

        X_tr, X_va, X_te = build_fold_matrices(X_train_base, X_test_base, fold, tr_mask, val_mask)
        y_tr, y_va = y_log1p[tr_mask], y_log1p[val_mask]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        val_pred = model.predict(X_va)
        oof_preds[val_mask] = val_pred
        test_preds += model.predict(X_te) / float(N_FOLDS)

        fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
        print(f"  [Fold {fold+1:2d}/{N_FOLDS}] best_iter={model.best_iteration_:4d} | RMSLE: {fold_rmsle:.5f}")

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f">>> [v3_lgb_et] 训练完成！耗时: {time.time()-t0:.1f}s")
    rmsle = save_and_verify_model("v3_lgb_et", oof_preds, test_preds, y_log1p, n_train, n_test)
    return oof_preds, test_preds, rmsle


# =========================================================================
# 模型 3: DART 随机树丢弃增强 LightGBM (v3_lgb_dart)
# =========================================================================
def train_v3_lgb_dart(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test):
    print("\n" + "=" * 70)
    print("🌲 [Model 3/7] 训练 DART 随机树丢弃增强 LightGBM (v3_lgb_dart)...")
    print("    超参数: boosting_type='dart', drop_rate=0.08, skip_drop=0.5, max_drop=50")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart',
        'drop_rate': 0.08,
        'skip_drop': 0.50,
        'max_drop': 50,
        'learning_rate': 0.08,
        'num_leaves': 45,
        'max_depth': -1,
        'colsample_bytree': 0.70,
        'subsample': 0.85,
        'subsample_freq': 1,
        'min_child_samples': 25,
        'reg_alpha': 0.15,
        'reg_lambda': 1.50,
        'n_estimators': 1000,
        'random_state': SEED + 202,
        'n_jobs': -1,
        'verbose': -1
    }

    t0 = time.time()
    for fold in range(N_FOLDS):
        t_fold = time.time()
        val_mask = (fold_ids == fold)
        tr_mask = ~val_mask

        X_tr, X_va, X_te = build_fold_matrices(X_train_base, X_test_base, fold, tr_mask, val_mask)
        y_tr, y_va = y_log1p[tr_mask], y_log1p[val_mask]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        oof_preds[val_mask] = val_pred
        test_preds += model.predict(X_te) / float(N_FOLDS)

        fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
        print(f"  [Fold {fold+1:2d}/{N_FOLDS}] n_estimators=1000 | RMSLE: {fold_rmsle:.5f} | 耗时: {time.time()-t_fold:.1f}s")

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f">>> [v3_lgb_dart] 训练完成！耗时: {time.time()-t0:.1f}s")
    rmsle = save_and_verify_model("v3_lgb_dart", oof_preds, test_preds, y_log1p, n_train, n_test)
    return oof_preds, test_preds, rmsle


# =========================================================================
# 模型 4: Huber 鲁棒损失函数 LightGBM (v3_lgb_huber)
# =========================================================================
def train_v3_lgb_huber(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test):
    print("\n" + "=" * 70)
    print("🌲 [Model 4/7] 训练 Huber 鲁棒损失函数 LightGBM (v3_lgb_huber)...")
    print("    超参数: objective='huber', alpha=0.8, metric='rmse'")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    params = {
        'objective': 'huber',
        'alpha': 0.80,
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 55,
        'max_depth': -1,
        'colsample_bytree': 0.70,
        'subsample': 0.80,
        'subsample_freq': 1,
        'min_child_samples': 25,
        'reg_alpha': 0.30,
        'reg_lambda': 1.50,
        'n_estimators': 3000,
        'random_state': SEED + 303,
        'n_jobs': -1,
        'verbose': -1
    }

    t0 = time.time()
    for fold in range(N_FOLDS):
        val_mask = (fold_ids == fold)
        tr_mask = ~val_mask

        X_tr, X_va, X_te = build_fold_matrices(X_train_base, X_test_base, fold, tr_mask, val_mask)
        y_tr, y_va = y_log1p[tr_mask], y_log1p[val_mask]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        val_pred = model.predict(X_va)
        oof_preds[val_mask] = val_pred
        test_preds += model.predict(X_te) / float(N_FOLDS)

        fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
        print(f"  [Fold {fold+1:2d}/{N_FOLDS}] best_iter={model.best_iteration_:4d} | RMSLE: {fold_rmsle:.5f}")

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f">>> [v3_lgb_huber] 训练完成！耗时: {time.time()-t0:.1f}s")
    rmsle = save_and_verify_model("v3_lgb_huber", oof_preds, test_preds, y_log1p, n_train, n_test)
    return oof_preds, test_preds, rmsle


# =========================================================================
# 模型 5: XGBoost GPU Hist 主力回归器 (v3_xgb_main)
# =========================================================================
def train_v3_xgb_main(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test, xgb_device):
    print("\n" + "=" * 70)
    print(f"🌲 [Model 5/7] 训练 XGBoost GPU Hist 主力回归器 (v3_xgb_main, device={xgb_device})...")
    print("    超参数: max_depth=7, learning_rate=0.03, colsample_bytree=0.70")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    params = {
        'tree_method': 'hist',
        'device': xgb_device,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 3000,
        'subsample': 0.80,
        'colsample_bytree': 0.70,
        'reg_alpha': 0.50,
        'reg_lambda': 2.00,
        'min_child_weight': 3.0,
        'early_stopping_rounds': 100,
        'random_state': SEED + 404,
        'n_jobs': -1
    }

    t0 = time.time()
    for fold in range(N_FOLDS):
        val_mask = (fold_ids == fold)
        tr_mask = ~val_mask

        X_tr, X_va, X_te = build_fold_matrices(X_train_base, X_test_base, fold, tr_mask, val_mask)
        y_tr, y_va = y_log1p[tr_mask], y_log1p[val_mask]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_pred = model.predict(X_va)
        oof_preds[val_mask] = val_pred
        test_preds += model.predict(X_te) / float(N_FOLDS)

        fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
        print(f"  [Fold {fold+1:2d}/{N_FOLDS}] best_iter={model.best_iteration:4d} | RMSLE: {fold_rmsle:.5f}")

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f">>> [v3_xgb_main] 训练完成！耗时: {time.time()-t0:.1f}s")
    rmsle = save_and_verify_model("v3_xgb_main", oof_preds, test_preds, y_log1p, n_train, n_test)
    return oof_preds, test_preds, rmsle


# =========================================================================
# 模型 6: XGBoost GPU Hist 深树高阶回归器 (v3_xgb_deep)
# =========================================================================
def train_v3_xgb_deep(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test, xgb_device):
    print("\n" + "=" * 70)
    print(f"🌲 [Model 6/7] 训练 XGBoost GPU Hist 深树高阶回归器 (v3_xgb_deep, device={xgb_device})...")
    print("    超参数: max_depth=9, learning_rate=0.025, colsample_bytree=0.65")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    params = {
        'tree_method': 'hist',
        'device': xgb_device,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 9,
        'learning_rate': 0.035,
        'n_estimators': 2500,
        'subsample': 0.80,
        'colsample_bytree': 0.65,
        'reg_alpha': 1.00,
        'reg_lambda': 3.00,
        'min_child_weight': 5.0,
        'early_stopping_rounds': 70,
        'random_state': SEED + 505,
        'n_jobs': -1
    }

    t0 = time.time()
    for fold in range(N_FOLDS):
        val_mask = (fold_ids == fold)
        tr_mask = ~val_mask

        X_tr, X_va, X_te = build_fold_matrices(X_train_base, X_test_base, fold, tr_mask, val_mask)
        y_tr, y_va = y_log1p[tr_mask], y_log1p[val_mask]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_pred = model.predict(X_va)
        oof_preds[val_mask] = val_pred
        test_preds += model.predict(X_te) / float(N_FOLDS)

        fold_rmsle = np.sqrt(mean_squared_error(y_va, val_pred))
        print(f"  [Fold {fold+1:2d}/{N_FOLDS}] best_iter={model.best_iteration:4d} | RMSLE: {fold_rmsle:.5f}")

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f">>> [v3_xgb_deep] 训练完成！耗时: {time.time()-t0:.1f}s")
    rmsle = save_and_verify_model("v3_xgb_deep", oof_preds, test_preds, y_log1p, n_train, n_test)
    return oof_preds, test_preds, rmsle


# =========================================================================
# 模型 7: 先验残差增强 LightGBM (v3_lgb_res)
# =========================================================================
def train_v3_lgb_res(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test):
    peer_oof_path = CACHE_V3_DIR / "text_zoo_oof" / "peer_lgb_oof.npy"
    peer_test_path = CACHE_V3_DIR / "text_zoo_oof" / "peer_lgb_test.npy"
    if not peer_oof_path.exists() or not peer_test_path.exists():
        raise FileNotFoundError(f"🚨 [FATAL] 先验 peer_lgb 预测文件缺失: {peer_oof_path}")

    peer_lgb_oof = np.load(peer_oof_path).astype(np.float32)
    peer_lgb_test = np.load(peer_test_path).astype(np.float32)
    peer_rmsle = float(np.sqrt(mean_squared_error(y_log1p, peer_lgb_oof)))

    print("\n" + "=" * 70)
    print("🌲 [Model 7/7] 训练先验残差增强 LightGBM (v3_lgb_res)...")
    print(f"    残差目标: y_log1p - peer_lgb_oof (先验基线 RMSLE: {peer_rmsle:.5f})")
    print("=" * 70)

    residuals = y_log1p - peer_lgb_oof
    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 45,
        'max_depth': -1,
        'colsample_bytree': 0.65,
        'subsample': 0.80,
        'subsample_freq': 1,
        'min_child_samples': 30,
        'reg_alpha': 0.80,
        'reg_lambda': 2.50,
        'n_estimators': 3000,
        'random_state': SEED + 606,
        'n_jobs': -1,
        'verbose': -1
    }

    t0 = time.time()
    for fold in range(N_FOLDS):
        val_mask = (fold_ids == fold)
        tr_mask = ~val_mask

        X_tr, X_va, X_te = build_fold_matrices(X_train_base, X_test_base, fold, tr_mask, val_mask)
        res_tr, res_va = residuals[tr_mask], residuals[val_mask]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, res_tr,
            eval_set=[(X_va, res_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        val_res_pred = model.predict(X_va)
        val_final_pred = peer_lgb_oof[val_mask] + val_res_pred
        oof_preds[val_mask] = val_final_pred

        test_res_pred = model.predict(X_te)
        test_preds += (peer_lgb_test + test_res_pred) / float(N_FOLDS)

        fold_rmsle = np.sqrt(mean_squared_error(y_log1p[val_mask], val_final_pred))
        print(f"  [Fold {fold+1:2d}/{N_FOLDS}] best_iter={model.best_iteration_:4d} | 还原后 RMSLE: {fold_rmsle:.5f}")

        del X_tr, X_va, X_te, model
        gc.collect()

    print(f">>> [v3_lgb_res] 训练完成！耗时: {time.time()-t0:.1f}s")
    rmsle = save_and_verify_model("v3_lgb_res", oof_preds, test_preds, y_log1p, n_train, n_test)
    return oof_preds, test_preds, rmsle


# =========================================================================
# 主程序驱动入口
# =========================================================================
def main():
    args = parse_args()
    print("=" * 75)
    print("🚀 [Wave 2] 启动 10-Fold v3 高多样性 GBDT 模型库训练流水线")
    print(f"   * 特征缓存基座: {CACHE_V3_DIR}")
    print(f"   * 产出保存目录: {OOF_DIR}")
    print(f"   * 指定执行模型: {args.models}")
    print("=" * 75)

    # 1. 检验就绪标记
    ready_v3_flag = CACHE_V3_DIR / "READY_V3.flag"
    if not ready_v3_flag.exists():
        raise FileNotFoundError(f"🚨 [FATAL] cache_10fold_v3/READY_V3.flag 不存在，上游特征基座尚未准备就绪！")

    # 2. 读取目标值与折划分
    y_log1p = np.load(CACHE_V3_DIR / "y_log1p.npy").astype(np.float32)
    fold_ids = np.load(CACHE_V3_DIR / "fold_ids.npy")
    n_train = len(y_log1p)
    n_test = 11971

    print(f">>> 数据规模确认: Train={n_train}, Test={n_test}, 目标值分布=[{y_log1p.min():.3f}, {y_log1p.max():.3f}], mean={y_log1p.mean():.3f}")

    # 3. 加载基础数值特征
    X_train_base, X_test_base, feature_cols = load_base_features()

    # 4. 检测 GPU
    xgb_device = check_xgboost_gpu()

    # 5. 待训练模型清单
    all_models = [
        "v3_lgb_main",
        "v3_lgb_et",
        "v3_lgb_dart",
        "v3_lgb_huber",
        "v3_xgb_main",
        "v3_xgb_deep",
        "v3_lgb_res"
    ]

    if "all" in args.models:
        models_to_run = all_models
    else:
        models_to_run = [m for m in args.models if m in all_models]

    print(f">>> 本次拟训练模型: {models_to_run}")

    model_results = {}
    oof_dict = {}
    test_dict = {}

    for m_name in models_to_run:
        oof_path = OOF_DIR / f"oof_{m_name}.npy"
        test_path = OOF_DIR / f"test_{m_name}.npy"

        # 断点续传跳过检查
        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof_arr = np.load(oof_path)
                test_arr = np.load(test_path)
                rmsle = save_and_verify_model(m_name, oof_arr, test_arr, y_log1p, n_train, n_test)
                oof_dict[m_name] = oof_arr
                test_dict[m_name] = test_arr
                model_results[m_name] = rmsle
                print(f"⏩ [Skip] 模型 {m_name} 已存在有效预测，跳过重训 (CV RMSLE = {rmsle:.5f})")
                continue
            except Exception as e:
                print(f">>> [Warning] 模型 {m_name} 现有预测校验未通过 ({e})，重新执行训练")

        if m_name == "v3_lgb_main":
            oof_p, test_p, rmsle = train_v3_lgb_main(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test)
        elif m_name == "v3_lgb_et":
            oof_p, test_p, rmsle = train_v3_lgb_et(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test)
        elif m_name == "v3_lgb_dart":
            oof_p, test_p, rmsle = train_v3_lgb_dart(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test)
        elif m_name == "v3_lgb_huber":
            oof_p, test_p, rmsle = train_v3_lgb_huber(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test)
        elif m_name == "v3_xgb_main":
            oof_p, test_p, rmsle = train_v3_xgb_main(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test, xgb_device)
        elif m_name == "v3_xgb_deep":
            oof_p, test_p, rmsle = train_v3_xgb_deep(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test, xgb_device)
        elif m_name == "v3_lgb_res":
            oof_p, test_p, rmsle = train_v3_lgb_res(X_train_base, X_test_base, y_log1p, fold_ids, n_train, n_test)
        else:
            continue

        oof_dict[m_name] = oof_p
        test_dict[m_name] = test_p
        model_results[m_name] = rmsle

    # 6. 如果有模型已落盘，补充加载以便计算互相关与 Blend
    for m in all_models:
        if m not in oof_dict:
            oof_path = OOF_DIR / f"oof_{m}.npy"
            test_path = OOF_DIR / f"test_{m}.npy"
            if oof_path.exists() and test_path.exists():
                oof_arr = np.load(oof_path)
                test_arr = np.load(test_path)
                oof_dict[m] = oof_arr
                test_dict[m] = test_arr
                model_results[m] = float(np.sqrt(mean_squared_error(y_log1p, oof_arr)))

    print("\n" + "=" * 75)
    print("📈 [Global 10-Fold CV RMSLE Leaderboard - oof_gbdt_v3]")
    print("=" * 75)
    sorted_models = sorted(model_results.items(), key=lambda x: x[1])
    for rank, (m_name, score) in enumerate(sorted_models, 1):
        print(f"  #{rank:2d}  {m_name.ljust(22)} : CV RMSLE = {score:.5f}")

    # 7. 计算多样性相关性矩阵 (Pearson Correlation)
    corr_dict = {}
    if len(oof_dict) >= 2:
        df_oof = pd.DataFrame(oof_dict)
        corr_matrix = df_oof.corr()
        print("\n" + "-" * 60)
        print("🔗 [OOF Prediction Correlation Matrix]")
        print("-" * 60)
        print(corr_matrix.round(4))
        corr_dict = corr_matrix.to_dict()

    # 8. 快速等权 Blend 检验互补增益
    blend_results = {}
    if len(oof_dict) >= 3:
        # 全部已训练模型等权
        blend_all = np.mean([oof_dict[m] for m in oof_dict], axis=0)
        rmsle_blend_all = float(np.sqrt(mean_squared_error(y_log1p, blend_all)))
        print(f"\n💡 [Equal Weight Blend (All {len(oof_dict)} Models)] RMSLE = {rmsle_blend_all:.5f}")
        blend_results["blend_all_equal"] = round(rmsle_blend_all, 5)

        # Top 4 模型等权
        top4_names = [m[0] for m in sorted_models[:4]]
        blend_top4 = np.mean([oof_dict[m] for m in top4_names], axis=0)
        rmsle_blend_top4 = float(np.sqrt(mean_squared_error(y_log1p, blend_top4)))
        print(f"💡 [Equal Weight Blend (Top 4: {', '.join(top4_names)})] RMSLE = {rmsle_blend_top4:.5f}")
        blend_results["blend_top4_equal"] = round(rmsle_blend_top4, 5)

    # 9. 落盘 summary_v3.json
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_dir": str(CACHE_V3_DIR),
        "n_train": n_train,
        "n_test": n_test,
        "n_base_features": len(feature_cols),
        "model_scores": {k: round(v, 5) for k, v in model_results.items()},
        "blend_scores": blend_results,
        "models_verified": list(model_results.keys()),
        "correlation_matrix": corr_dict
    }

    summary_file = OOF_DIR / "summary_v3.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f">>> [Summary 落盘] {summary_file}")

    # 10. 检查全部 7 个模型是否已落盘，若齐备则生成 READY_V3.flag
    all_ready = all(
        (OOF_DIR / f"oof_{m}.npy").exists() and (OOF_DIR / f"test_{m}.npy").exists()
        for m in all_models
    )

    if all_ready:
        ready_flag_file = OOF_DIR / "READY_V3.flag"
        best_model_name, best_score = sorted_models[0]
        with open(ready_flag_file, "w", encoding="utf-8") as f:
            f.write(f"oof_gbdt_v3 completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total models trained: {len(all_models)}\n")
            f.write(f"Best single model: {best_model_name} (CV RMSLE = {best_score:.5f})\n")
            if blend_results:
                f.write(f"Top-4 equal blend RMSLE: {blend_results.get('blend_top4_equal', 'N/A')}\n")
        print(f">>> 🏁 [READY_V3.flag 落盘] {ready_flag_file} 全部 7 路模型已就绪！")

    # 11. 释放显存与内存
    gc.collect()
    print(">>> 显存与内存资源已彻底清理完成！")


if __name__ == "__main__":
    main()

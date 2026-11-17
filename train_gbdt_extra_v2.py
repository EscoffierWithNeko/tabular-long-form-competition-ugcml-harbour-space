# -*- coding: utf-8 -*-
"""
train_gbdt_extra_v2.py -- 10-Fold 差异化高阶 GBDT 扩展训练与残差模型落盘
===================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务: 基于无泄漏 v2 特征基座 (cache_10fold_v2/) 训练差异化 GBDT 扩展基元，
      包含：
      1. lgb_res: 以 peer_lgb 为先验基线的残差增强 LightGBM
      2. lgb_huber: Huber 鲁棒损失函数 (huber_delta=0.8) 抗长尾噪声 LightGBM
      3. xgb_reg: 浅树高正则化 (max_depth=5, reg_alpha=3.0, reg_lambda=8.0) XGBoost (CUDA)
      并将 10-Fold OOF 与 Test 预测严格落盘至 oof_gbdt_v2/，
      同时与原有 4 路模型 (lgb_std, lgb_et, xgb_main, xgb_deep) 联合评估，
      更新 summary_v2.json 与 READY_V2.flag。
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 目录常量
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / 'cache_10fold_v2'
OOF_DIR = BASE_DIR / 'oof_gbdt_v2'

# 赛题真实目标值硬边界 (MemPalace 验证: min=20.0, max=2000.0)
PRICE_MIN = 20.0
PRICE_MAX = 2000.0
LOG1P_MIN = float(np.log1p(PRICE_MIN))
LOG1P_MAX = float(np.log1p(PRICE_MAX))


def check_xgboost_gpu():
    """检测 XGBoost CUDA 加速支持"""
    try:
        x_dummy = np.random.randn(10, 5).astype(np.float32)
        y_dummy = np.random.randn(10).astype(np.float32)
        test_xgb = xgb.XGBRegressor(n_estimators=1, tree_method='hist', device='cuda')
        test_xgb.fit(x_dummy, y_dummy)
        return 'cuda'
    except Exception as e:
        print(f"[WARN] XGBoost CUDA 加速不可用 ({e})，降级为 CPU hist")
        return 'cpu'


def verify_and_report_file(file_path, expected_len):
    """校验落盘 npy 文件长度及无 NaN/Inf"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    arr = np.load(file_path)
    assert len(arr) == expected_len, f"长度不符: 期望 {expected_len}, 实际 {len(arr)}"
    assert not np.isnan(arr).any(), f"文件中含有 NaN: {file_path}"
    assert not np.isinf(arr).any(), f"文件中含有 Inf: {file_path}"
    return arr


def parse_args():
    parser = argparse.ArgumentParser(description="10-Fold 差异化 GBDT 扩展训练管道")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "lgb_res", "lgb_huber", "xgb_reg"],
                        help="指定训练的模型: all, lgb_res, lgb_huber, xgb_reg")
    parser.add_argument("--skip-existing", action="store_true",
                        help="若预测文件已存在且校验通过则跳过训练")
    return parser.parse_args()


def train_lgb_res(X_train_all, y_log1p, peer_lgb_oof, X_test_all, peer_lgb_test, fold_ids, n_train, n_test):
    """
    模型 1: lgb_res -- 对 peer_lgb 的残差进行 10-Fold 拟合
    """
    oof_path = OOF_DIR / 'oof_lgb_res.npy'
    test_path = OOF_DIR / 'test_lgb_res.npy'

    print("\n" + "=" * 70)
    print("🌲 [Extra 1/3] 开始训练 LightGBM 残差增强模型 (lgb_res)...")
    print(f"    残差目标: y_log1p - peer_lgb_oof (先验基准 RMSLE: {np.sqrt(mean_squared_error(y_log1p, peer_lgb_oof)):.5f})")
    print("=" * 70)

    # 目标残差
    residuals = y_log1p - peer_lgb_oof
    print(f"    残差均值: {np.mean(residuals):.5f}, 标准差: {np.std(residuals):.5f}")

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)
    fold_scores = []

    for fold in range(10):
        t0 = time.time()
        va_mask = (fold_ids == fold)
        tr_mask = ~va_mask

        X_tr, y_res_tr = X_train_all[tr_mask], residuals[tr_mask]
        X_va, y_res_va = X_train_all[va_mask], residuals[va_mask]
        y_orig_va = y_log1p[va_mask]
        peer_va = peer_lgb_oof[va_mask]

        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 45,
            'max_depth': -1,
            'colsample_bytree': 0.65,
            'subsample': 0.8,
            'subsample_freq': 1,
            'min_child_samples': 30,
            'reg_alpha': 0.8,
            'reg_lambda': 2.5,
            'n_estimators': 3000,
            'random_state': 42 + fold,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_res_tr,
            eval_set=[(X_va, y_res_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        best_iter = model.best_iteration_
        res_va_pred = model.predict(X_va)
        fold_final_pred = peer_va + res_va_pred
        oof_preds[va_mask] = fold_final_pred

        res_test_pred = model.predict(X_test_all)
        test_preds += (peer_lgb_test + res_test_pred) / 10.0

        fold_rmsle = np.sqrt(mean_squared_error(y_orig_va, fold_final_pred))
        fold_scores.append(fold_rmsle)
        dur = time.time() - t0
        print(f"  [Fold {fold:02d}/10] best_iter: {best_iter:4d} | va RMSLE: {fold_rmsle:.5f} | 耗时: {dur:.1f}s")

    total_rmsle = float(np.sqrt(mean_squared_error(y_log1p, oof_preds)))
    print(f">> [lgb_res] 10-Fold CV RMSLE: {total_rmsle:.5f} (折间均值: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    np.save(oof_path, oof_preds)
    np.save(test_path, test_preds)
    print(f"   [Saved] {oof_path.name} & {test_path.name}")
    return oof_preds, test_preds, total_rmsle


def train_lgb_huber(X_train_all, y_log1p, X_test_all, fold_ids, n_train, n_test):
    """
    模型 2: lgb_huber -- 基于 Huber 损失 (delta=0.8) 抗长尾噪声的 LightGBM
    """
    oof_path = OOF_DIR / 'oof_lgb_huber.npy'
    test_path = OOF_DIR / 'test_lgb_huber.npy'

    print("\n" + "=" * 70)
    print("🌲 [Extra 2/3] 开始训练 LightGBM Huber 损失模型 (lgb_huber)...")
    print("    超参数: objective='huber', huber_delta=0.8, metric='rmse', num_leaves=55")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)
    fold_scores = []

    for fold in range(10):
        t0 = time.time()
        va_mask = (fold_ids == fold)
        tr_mask = ~va_mask

        X_tr, y_tr = X_train_all[tr_mask], y_log1p[tr_mask]
        X_va, y_va = X_train_all[va_mask], y_log1p[va_mask]

        lgb_params = {
            'objective': 'huber',
            'huber_delta': 0.8,
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 55,
            'max_depth': -1,
            'colsample_bytree': 0.70,
            'subsample': 0.8,
            'subsample_freq': 1,
            'min_child_samples': 25,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
            'n_estimators': 3000,
            'random_state': 142 + fold,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        best_iter = model.best_iteration_
        fold_pred = model.predict(X_va)
        oof_preds[va_mask] = fold_pred
        test_preds += model.predict(X_test_all) / 10.0

        fold_rmsle = np.sqrt(mean_squared_error(y_va, fold_pred))
        fold_scores.append(fold_rmsle)
        dur = time.time() - t0
        print(f"  [Fold {fold:02d}/10] best_iter: {best_iter:4d} | va RMSLE: {fold_rmsle:.5f} | 耗时: {dur:.1f}s")

    total_rmsle = float(np.sqrt(mean_squared_error(y_log1p, oof_preds)))
    print(f">> [lgb_huber] 10-Fold CV RMSLE: {total_rmsle:.5f} (折间均值: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    np.save(oof_path, oof_preds)
    np.save(test_path, test_preds)
    print(f"   [Saved] {oof_path.name} & {test_path.name}")
    return oof_preds, test_preds, total_rmsle


def train_xgb_reg(X_train_all, y_log1p, X_test_all, fold_ids, n_train, n_test, xgb_device):
    """
    模型 3: xgb_reg -- 浅树强正则 (max_depth=5, reg_alpha=3.0, reg_lambda=8.0) XGBoost
    """
    oof_path = OOF_DIR / 'oof_xgb_reg.npy'
    test_path = OOF_DIR / 'test_xgb_reg.npy'

    print("\n" + "=" * 70)
    print(f"🚀 [Extra 3/3] 开始训练浅树强正则 XGBoost (xgb_reg, device={xgb_device})...")
    print("    超参数: max_depth=5, lr=0.03, colsample=0.6, reg_alpha=3.0, reg_lambda=8.0")
    print("=" * 70)

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)
    fold_scores = []

    for fold in range(10):
        t0 = time.time()
        va_mask = (fold_ids == fold)
        tr_mask = ~va_mask

        X_tr, y_tr = X_train_all[tr_mask], y_log1p[tr_mask]
        X_va, y_va = X_train_all[va_mask], y_log1p[va_mask]

        xgb_params = {
            'n_estimators': 3000,
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_alpha': 3.0,
            'reg_lambda': 8.0,
            'random_state': 242 + fold,
            'tree_method': 'hist',
            'device': xgb_device,
            'eval_metric': 'rmse',
            'early_stopping_rounds': 100
        }

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )

        best_iter = model.best_iteration
        fold_pred = model.predict(X_va)
        oof_preds[va_mask] = fold_pred
        test_preds += model.predict(X_test_all) / 10.0

        fold_rmsle = np.sqrt(mean_squared_error(y_va, fold_pred))
        fold_scores.append(fold_rmsle)
        dur = time.time() - t0
        print(f"  [Fold {fold:02d}/10] best_iter: {best_iter:4d} | va RMSLE: {fold_rmsle:.5f} | 耗时: {dur:.1f}s")

    total_rmsle = float(np.sqrt(mean_squared_error(y_log1p, oof_preds)))
    print(f">> [xgb_reg] 10-Fold CV RMSLE: {total_rmsle:.5f} (折间均值: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    np.save(oof_path, oof_preds)
    np.save(test_path, test_preds)
    print(f"   [Saved] {oof_path.name} & {test_path.name}")
    return oof_preds, test_preds, total_rmsle


def main():
    args = parse_args()
    t_start = time.time()
    print("=" * 80)
    print("🌟 启动 GBDT 家族动物园差异化扩展训练管道 (v2)")
    print(f"   执行模型: {args.model} | skip_existing: {args.skip_existing}")
    print("=" * 80)

    # 1. 检查基座与依赖
    ready_flag = CACHE_DIR / 'READY_V2.flag'
    if not ready_flag.exists():
        raise FileNotFoundError(f"前置缓存标志不存在: {ready_flag}")

    print(f">>> 正在从 {CACHE_DIR.name} 加载统一 10-Fold 零泄漏特征基座...")
    t0 = time.time()
    train_feat = pd.read_parquet(CACHE_DIR / 'train_features.parquet')
    test_feat = pd.read_parquet(CACHE_DIR / 'test_features.parquet')
    fold_ids = np.load(CACHE_DIR / 'fold_ids.npy')
    y_log1p = np.load(CACHE_DIR / 'y_log1p.npy')

    # 加载 peer_lgb 先验基准
    peer_lgb_oof = np.load(CACHE_DIR / 'text_zoo_oof' / 'peer_lgb_oof.npy')
    peer_lgb_test = np.load(CACHE_DIR / 'text_zoo_oof' / 'peer_lgb_test.npy')

    n_train = len(train_feat)
    n_test = len(test_feat)
    assert len(y_log1p) == n_train
    assert len(fold_ids) == n_train
    assert len(peer_lgb_oof) == n_train
    assert len(peer_lgb_test) == n_test

    feature_cols = [c for c in train_feat.columns if c not in ['id', 'price']]
    print(f">>> 数值特征维度: {len(feature_cols)} 列 (Train={n_train}, Test={n_test})")

    X_train_all = train_feat[feature_cols].values.astype(np.float32)
    X_test_all = test_feat[feature_cols].values.astype(np.float32)

    xgb_device = check_xgboost_gpu()

    # 存储所有新旧模型的 OOF 和 Test 字典
    all_oofs = {}
    all_tests = {}
    cv_scores = {}

    # 读取旧有的 4 路模型 (lgb_std, lgb_et, xgb_main, xgb_deep)
    base_models = ['lgb_std', 'lgb_et', 'xgb_main', 'xgb_deep']
    for m in base_models:
        oof_f = OOF_DIR / f"oof_{m}.npy"
        test_f = OOF_DIR / f"test_{m}.npy"
        if oof_f.exists() and test_f.exists():
            arr_oof = verify_and_report_file(oof_f, n_train)
            arr_test = verify_and_report_file(test_f, n_test)
            all_oofs[m] = arr_oof
            all_tests[m] = arr_test
            score = float(np.sqrt(mean_squared_error(y_log1p, arr_oof)))
            cv_scores[m] = score
            print(f"  [Existing] 已加载基座模型 {m:10s} | CV RMSLE: {score:.5f}")

    # 训练/加载 Extra 模型
    # Model 1: lgb_res
    if args.model in ["all", "lgb_res"]:
        oof_path = OOF_DIR / 'oof_lgb_res.npy'
        test_path = OOF_DIR / 'test_lgb_res.npy'
        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof = verify_and_report_file(oof_path, n_train)
                test = verify_and_report_file(test_path, n_test)
                score = float(np.sqrt(mean_squared_error(y_log1p, oof)))
                print(f"  [Skip] lgb_res 已存在且校验通过，CV RMSLE: {score:.5f}")
                all_oofs['lgb_res'] = oof
                all_tests['lgb_res'] = test
                cv_scores['lgb_res'] = score
            except Exception as e:
                print(f"  [Rebuild] lgb_res 校验失败 ({e})，重新训练...")
                oof, test, score = train_lgb_res(X_train_all, y_log1p, peer_lgb_oof, X_test_all, peer_lgb_test, fold_ids, n_train, n_test)
                all_oofs['lgb_res'] = oof
                all_tests['lgb_res'] = test
                cv_scores['lgb_res'] = score
        else:
            oof, test, score = train_lgb_res(X_train_all, y_log1p, peer_lgb_oof, X_test_all, peer_lgb_test, fold_ids, n_train, n_test)
            all_oofs['lgb_res'] = oof
            all_tests['lgb_res'] = test
            cv_scores['lgb_res'] = score

    # Model 2: lgb_huber
    if args.model in ["all", "lgb_huber"]:
        oof_path = OOF_DIR / 'oof_lgb_huber.npy'
        test_path = OOF_DIR / 'test_lgb_huber.npy'
        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof = verify_and_report_file(oof_path, n_train)
                test = verify_and_report_file(test_path, n_test)
                score = float(np.sqrt(mean_squared_error(y_log1p, oof)))
                print(f"  [Skip] lgb_huber 已存在且校验通过，CV RMSLE: {score:.5f}")
                all_oofs['lgb_huber'] = oof
                all_tests['lgb_huber'] = test
                cv_scores['lgb_huber'] = score
            except Exception as e:
                print(f"  [Rebuild] lgb_huber 校验失败 ({e})，重新训练...")
                oof, test, score = train_lgb_huber(X_train_all, y_log1p, X_test_all, fold_ids, n_train, n_test)
                all_oofs['lgb_huber'] = oof
                all_tests['lgb_huber'] = test
                cv_scores['lgb_huber'] = score
        else:
            oof, test, score = train_lgb_huber(X_train_all, y_log1p, X_test_all, fold_ids, n_train, n_test)
            all_oofs['lgb_huber'] = oof
            all_tests['lgb_huber'] = test
            cv_scores['lgb_huber'] = score

    # Model 3: xgb_reg
    if args.model in ["all", "xgb_reg"]:
        oof_path = OOF_DIR / 'oof_xgb_reg.npy'
        test_path = OOF_DIR / 'test_xgb_reg.npy'
        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof = verify_and_report_file(oof_path, n_train)
                test = verify_and_report_file(test_path, n_test)
                score = float(np.sqrt(mean_squared_error(y_log1p, oof)))
                print(f"  [Skip] xgb_reg 已存在且校验通过，CV RMSLE: {score:.5f}")
                all_oofs['xgb_reg'] = oof
                all_tests['xgb_reg'] = test
                cv_scores['xgb_reg'] = score
            except Exception as e:
                print(f"  [Rebuild] xgb_reg 校验失败 ({e})，重新训练...")
                oof, test, score = train_xgb_reg(X_train_all, y_log1p, X_test_all, fold_ids, n_train, n_test, xgb_device)
                all_oofs['xgb_reg'] = oof
                all_tests['xgb_reg'] = test
                cv_scores['xgb_reg'] = score
        else:
            oof, test, score = train_xgb_reg(X_train_all, y_log1p, X_test_all, fold_ids, n_train, n_test, xgb_device)
            all_oofs['xgb_reg'] = oof
            all_tests['xgb_reg'] = test
            cv_scores['xgb_reg'] = score

    # ======================================================================
    # 综合评估、多样性分析与融合提升校验
    # ======================================================================
    print("\n" + "=" * 80)
    print("📊 [综合评估] GBDT 家族动物园多样性矩阵与融合验证")
    print("=" * 80)

    model_names = list(all_oofs.keys())
    print(f"可用模型总数: {len(model_names)} -> {model_names}")

    # 1. 严格质量区间检验: 0.22 < CV < 0.45
    for name in model_names:
        score = cv_scores[name]
        assert 0.22 < score < 0.45, f"模型 {name} CV RMSLE {score:.5f} 越界 (0.22, 0.45)!"
        print(f"  - {name:12s} CV RMSLE: {score:.5f} [PASS: 0.22 < {score:.5f} < 0.45]")

    # 2. 相关系数多样性矩阵
    df_oof = pd.DataFrame(all_oofs)
    corr_mat = df_oof.corr()
    print("\n[模型预测 Pearson 相关系数矩阵]:")
    print(corr_mat.round(4).to_string())

    # 3. 融合评估对比
    print("\n[集成融合对比实验]:")
    # A. 基础 4-GBDT (lgb_std, lgb_et, xgb_main, xgb_deep)
    if all(m in all_oofs for m in base_models):
        blend_4 = np.mean([all_oofs[m] for m in base_models], axis=0)
        score_4 = np.sqrt(mean_squared_error(y_log1p, blend_4))
        print(f"  - 4-GBDT Baseline 等权融合 RMSLE: {score_4:.5f}")

    # B. 全量 GBDT (6 或 7 模型) 等权融合
    all_blend = np.mean([all_oofs[m] for m in model_names], axis=0)
    score_all_raw = np.sqrt(mean_squared_error(y_log1p, all_blend))
    print(f"  - 全量 {len(model_names)}-GBDT 等权融合 RMSLE: {score_all_raw:.5f}")

    # C. 带真实硬边界截断 [LOG1P_MIN, LOG1P_MAX]
    all_blend_clipped = np.clip(all_blend, LOG1P_MIN, LOG1P_MAX)
    score_all_clipped = np.sqrt(mean_squared_error(y_log1p, all_blend_clipped))
    print(f"  - 全量 {len(model_names)}-GBDT + 硬边界截断 RMSLE: {score_all_clipped:.5f}")

    # D. 贪心加权优化 (Ridge / Non-negative Least Squares 简版优化)
    from scipy.optimize import minimize
    def obj_func(weights):
        weights = weights / np.sum(weights)
        pred = sum(w * all_oofs[m] for w, m in zip(weights, model_names))
        return np.sqrt(mean_squared_error(y_log1p, pred))

    init_w = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]
    res_opt = minimize(obj_func, init_w, bounds=bounds, method='SLSQP')
    opt_w = res_opt.x / np.sum(res_opt.x)
    opt_pred = sum(w * all_oofs[m] for w, m in zip(opt_w, model_names))
    score_opt_raw = np.sqrt(mean_squared_error(y_log1p, opt_pred))
    opt_pred_clipped = np.clip(opt_pred, LOG1P_MIN, LOG1P_MAX)
    score_opt_clipped = np.sqrt(mean_squared_error(y_log1p, opt_pred_clipped))

    print(f"\n[加权集成最优结果]:")
    for name, w in zip(model_names, opt_w):
        print(f"    权重 {name:12s}: {w:6.4f}")
    print(f"  - 最佳加权融合 RMSLE: {score_opt_raw:.5f}")
    print(f"  - 最佳加权融合 + 硬边界截断 RMSLE: {score_opt_clipped:.5f}")

    # 4. 汇总元数据至 summary_v2.json
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": len(model_names),
        "model_list": model_names,
        "cv_scores": cv_scores,
        "blend_equal_rmsle": float(score_all_raw),
        "blend_equal_clipped_rmsle": float(score_all_clipped),
        "blend_optimal_weights": {m: float(w) for m, w in zip(model_names, opt_w)},
        "blend_optimal_rmsle": float(score_opt_raw),
        "blend_optimal_clipped_rmsle": float(score_opt_clipped),
        "n_train": n_train,
        "n_test": n_test,
        "n_features": len(feature_cols),
        "total_elapsed_seconds": round(time.time() - t_start, 2)
    }

    summary_file = OOF_DIR / 'summary_v2.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n>>> 元数据已成功刷新至 {summary_file}")

    # 5. 刷新 READY_V2.flag
    ready_v2_flag = OOF_DIR / 'READY_V2.flag'
    with open(ready_v2_flag, 'w', encoding='utf-8') as f:
        f.write(f"ALL_GBDT_READY_V2\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models: {len(model_names)}\n")
        f.write(f"Best Blended RMSLE: {score_opt_clipped:.5f}\n")
    print(f">>> 标志文件已刷新: {ready_v2_flag}")

    print("\n" + "=" * 80)
    print(f"✅ GBDT 动物园扩展全部完成！耗时: {time.time() - t_start:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()

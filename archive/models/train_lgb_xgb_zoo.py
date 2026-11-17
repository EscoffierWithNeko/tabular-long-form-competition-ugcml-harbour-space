# -*- coding: utf-8 -*-
"""
train_lgb_xgb_zoo.py -- 10-Fold GBDT 家族动物园训练与折外元特征落盘
===================================================================
任务定位:
1. 读取统一 10-Fold 基座 (cache_10fold/train_features.parquet, test_features.parquet, fold_ids.npy, y_log1p.npy)
2. 训练 3 路高多样性 GBDT 模型:
   - lgb_std: LightGBM 标准配置 (num_leaves=63, lr=0.03, colsample_bytree=0.7, n_estimators=2500, early_stopping=100)
   - lgb_et: LightGBM Extra-Trees 随机切分 (extra_trees=True, colsample_bytree=0.6, subsample=0.8)
   - xgb_main: XGBoost GPU/Hist 精调版 (max_depth=7, lr=0.03, colsample_bytree=0.7, n_estimators=2500, early_stopping=100)
3. 严格计算每折与全体外折 OOF RMSLE，即时落盘至 oof_gbdt/ 目录:
   - oof_lgb_std.npy, test_lgb_std.npy
   - oof_lgb_et.npy, test_lgb_et.npy
   - oof_xgb.npy, test_xgb.npy
4. 校验所有 npy 长度（47293 与 11971）无 NaN/Inf，生成 summary.json 并创建 oof_gbdt/READY.flag。
"""

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

WORKSPACE_DIR = Path(__file__).resolve().parent
CACHE_DIR = WORKSPACE_DIR / 'cache_10fold'
OOF_DIR = WORKSPACE_DIR / 'oof_gbdt'
OOF_DIR.mkdir(parents=True, exist_ok=True)


def check_xgboost_gpu():
    """检测 XGBoost 的 CUDA 支持情况，支持则返回 'cuda'，否则降级 'cpu'"""
    try:
        dummy_x = np.random.randn(50, 5).astype(np.float32)
        dummy_y = np.random.randn(50).astype(np.float32)
        model = xgb.XGBRegressor(
            n_estimators=5,
            tree_method='hist',
            device='cuda',
            random_state=42,
            verbosity=0
        )
        model.fit(dummy_x, dummy_y)
        _ = model.predict(dummy_x)
        print(">>> [SUCCESS] XGBoost CUDA GPU 加速就绪！将使用 device='cuda'")
        return 'cuda'
    except Exception as e:
        print(f">>> [WARNING] XGBoost CUDA 不可用 ({e})，降级使用 device='cpu'。")
        return 'cpu'


def verify_and_report_file(file_path, expected_len):
    """校验 numpy 预测文件的存在性、长度以及无 NaN/Inf"""
    assert file_path.exists(), f"文件不存在: {file_path}"
    arr = np.load(file_path)
    assert len(arr) == expected_len, f"文件 {file_path.name} 长度异常: 期望 {expected_len}, 实际 {len(arr)}"
    assert not np.isnan(arr).any(), f"文件 {file_path.name} 包含 NaN!"
    assert not np.isinf(arr).any(), f"文件 {file_path.name} 包含 Inf!"
    print(f"  [VERIFY OK] {file_path.name:18s} | 长度: {len(arr):5d} | 均值: {arr.mean():.4f} | std: {arr.std():.4f} | min: {arr.min():.4f} | max: {arr.max():.4f}")
    return arr


def parse_args():
    parser = argparse.ArgumentParser(description="Train LGBM & XGBoost Zoo on 10-Fold")
    parser.add_argument("--model", type=str, default="all", choices=["all", "lgb_std", "lgb_et", "xgb_main"],
                        help="指定要训练的模型 (默认: all)")
    parser.add_argument("--skip-existing", action="store_true", help="若预测文件已存在且校验通过则跳过训练")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    print("=" * 70)
    print("🚀 [Step 2] 启动 GBDT 家族动物园训练与 10-Fold OOF 落盘管道")
    print(f"   目标模型: {args.model} | skip_existing: {args.skip_existing}")
    print("=" * 70)

    # 1. 检查基座缓存
    clean_flag = CACHE_DIR / 'READY_CLEAN.flag'
    ready_flag = CACHE_DIR / 'READY.flag'
    if not clean_flag.exists() and not ready_flag.exists():
        raise FileNotFoundError(f"前置缓存标志不存在: {clean_flag} 或 {ready_flag}，请先运行上游特征工程！")

    print(">>> 正在从 cache_10fold 加载统一 10-Fold 零泄漏特征基座...")
    t0 = time.time()
    train_feat = pd.read_parquet(CACHE_DIR / 'train_base_features.parquet')
    test_feat = pd.read_parquet(CACHE_DIR / 'test_base_features.parquet')
    fold_ids = np.load(CACHE_DIR / 'fold_ids.npy')
    y_log1p = np.load(CACHE_DIR / 'y_log1p.npy')
    print(f">>> 基座加载完成 (耗时: {time.time() - t0:.2f}s)")
    print(f"    Train 基座特征矩阵: {train_feat.shape}")
    print(f"    Test 基座特征矩阵 : {test_feat.shape}")
    print(f"    Fold IDs 形状     : {fold_ids.shape} (唯一折数: {len(np.unique(fold_ids))})")
    print(f"    Target 形状       : {y_log1p.shape} (min: {y_log1p.min():.4f}, max: {y_log1p.max():.4f})")

    # 准备纯数值基础特征矩阵，并进行严格防泄漏审查
    excl_cols = {'id', 'price', 'log_price', 'target', 'y_log1p'}
    base_num_cols = [c for c in train_feat.columns if c not in excl_cols and np.issubdtype(train_feat[c].dtype, np.number)]
    assert 'price' not in base_num_cols, "FATAL: 'price' found in base_num_cols!"
    print(f">>> 入模纯数值基础特征维度: {len(base_num_cols)} 列 (已显式剔除 price 及非数值列)")

    # 防泄漏守卫: 检查每列与 y_log1p 的相关系数绝对值 < 0.98
    for c in base_num_cols:
        vals = np.nan_to_num(train_feat[c].values.astype(np.float64))
        corr = abs(np.corrcoef(vals, y_log1p)[0, 1])
        assert corr < 0.98, f"FATAL: Column '{c}' has suspiciously high correlation with target: {corr:.4f}"
    print(">>> [防泄漏守卫] 全部基础特征与 y_log1p 相关系数校验通过 (全部 |corr| < 0.98)")

    X_base_train = train_feat[base_num_cols].values.astype(np.float32)
    X_base_test = test_feat[base_num_cols].values.astype(np.float32)

    # 检查并填充可能存在的极小残留 NaN
    if np.isnan(X_base_train).any():
        print(">>> 警告: X_base_train 存在极少 NaN，执行快速均值补齐...")
        col_means = np.nanmean(X_base_train, axis=0)
        inds = np.where(np.isnan(X_base_train))
        X_base_train[inds] = np.take(col_means, inds[1])
    if np.isnan(X_base_test).any():
        print(">>> 警告: X_base_test 存在极少 NaN，执行快速均值补齐...")
        col_means_te = np.nanmean(X_base_train, axis=0)
        inds = np.where(np.isnan(X_base_test))
        X_base_test[inds] = np.take(col_means_te, inds[1])

    n_train = len(X_base_train)
    n_test = len(X_base_test)
    assert n_train == 47293, f"训练集长度不匹配: {n_train}"
    assert n_test == 11971, f"测试集长度不匹配: {n_test}"

    # 预加载 10 折折内高阶无泄漏特征矩阵
    print(">>> 正在预加载 10-Fold 折内高阶无泄漏特征矩阵 (cache_10fold/fold_te/)...")
    fold_te_data = {}
    for fold in range(10):
        npz_path = CACHE_DIR / 'fold_te' / f'fold_{fold}.npz'
        if not npz_path.exists():
            raise FileNotFoundError(f"Fold 特征矩阵不存在: {npz_path}")
        data = np.load(npz_path)
        fold_te_data[fold] = {
            'X_tr_te': data['X_tr_te'].astype(np.float32),
            'X_va_te': data['X_va_te'].astype(np.float32),
            'X_te_te': data['X_te_te'].astype(np.float32),
        }
    te_feat_dim = fold_te_data[0]['X_tr_te'].shape[1]
    total_feat_dim = X_base_train.shape[1] + te_feat_dim
    print(f">>> 10 折高阶折内特征预加载完成: 每折高阶特征 {te_feat_dim} 列 | 总入模特征 {total_feat_dim} 列")

    # 路径定义
    lgb_std_oof_path = OOF_DIR / 'oof_lgb_std.npy'
    lgb_std_test_path = OOF_DIR / 'test_lgb_std.npy'
    lgb_et_oof_path = OOF_DIR / 'oof_lgb_et.npy'
    lgb_et_test_path = OOF_DIR / 'test_lgb_et.npy'
    xgb_oof_path = OOF_DIR / 'oof_xgb.npy'
    xgb_test_path = OOF_DIR / 'test_xgb.npy'

    # =========================================================================
    # 模型 1: LightGBM Standard (lgb_std)
    # =========================================================================
    if args.model in ['all', 'lgb_std']:
        need_train = True
        if args.skip_existing and lgb_std_oof_path.exists() and lgb_std_test_path.exists():
            try:
                verify_and_report_file(lgb_std_oof_path, 47293)
                verify_and_report_file(lgb_std_test_path, 11971)
                print(">>> [lgb_std] 检测到已有完整预测文件且校验通过，跳过重新训练。")
                need_train = False
            except Exception as e:
                print(f">>> [lgb_std] 已有文件校验失败 ({e})，重新执行训练。")

        if need_train:
            print("\n" + "=" * 70)
            print("🌲 [1/3] 开始训练 LightGBM Standard (lgb_std)...")
            print("=" * 70)
            lgb_std_oof = np.zeros(n_train, dtype=np.float32)
            lgb_std_test = np.zeros(n_test, dtype=np.float32)
            lgb_std_fold_scores = []

            lgb_std_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.03,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': 8,
                'verbose': -1
            }

            t_m1 = time.time()
            for fold in range(10):
                t_f = time.time()
                tr_mask = (fold_ids != fold)
                va_mask = (fold_ids == fold)

                te_dict = fold_te_data[fold]
                X_tr = np.hstack([X_base_train[tr_mask], te_dict['X_tr_te']])
                X_va = np.hstack([X_base_train[va_mask], te_dict['X_va_te']])
                X_te = np.hstack([X_base_test, te_dict['X_te_te']])
                y_tr = y_log1p[tr_mask]
                y_va = y_log1p[va_mask]

                trn_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
                val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, free_raw_data=False)

                model = lgb.train(
                    lgb_std_params,
                    trn_data,
                    num_boost_round=2500,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )

                va_pred = model.predict(X_va, num_iteration=model.best_iteration)
                te_pred = model.predict(X_te, num_iteration=model.best_iteration)

                lgb_std_oof[va_mask] = va_pred
                lgb_std_test += te_pred / 10.0

                score = float(np.sqrt(mean_squared_error(y_va, va_pred)))
                lgb_std_fold_scores.append(score)
                print(f"  [Fold {fold+1}/10] best_iter: {model.best_iteration:4d} | RMSLE: {score:.5f} | 耗时: {time.time() - t_f:.2f}s")

            overall_score = float(np.sqrt(mean_squared_error(y_log1p, lgb_std_oof)))
            print(f">>> [lgb_std 完成] 10-Fold Overall OOF RMSLE: {overall_score:.5f} (总耗时: {time.time() - t_m1:.2f}s)")
            np.save(lgb_std_oof_path, lgb_std_oof)
            np.save(lgb_std_test_path, lgb_std_test)
            verify_and_report_file(lgb_std_oof_path, 47293)
            verify_and_report_file(lgb_std_test_path, 11971)

    # =========================================================================
    # 模型 2: LightGBM Extra-Trees (lgb_et)
    # =========================================================================
    if args.model in ['all', 'lgb_et']:
        need_train = True
        if args.skip_existing and lgb_et_oof_path.exists() and lgb_et_test_path.exists():
            try:
                verify_and_report_file(lgb_et_oof_path, 47293)
                verify_and_report_file(lgb_et_test_path, 11971)
                print(">>> [lgb_et] 检测到已有完整预测文件且校验通过，跳过重新训练。")
                need_train = False
            except Exception as e:
                print(f">>> [lgb_et] 已有文件校验失败 ({e})，重新执行训练。")

        if need_train:
            print("\n" + "=" * 70)
            print("🌲 [2/3] 开始训练 LightGBM Extra-Trees (lgb_et)...")
            print("=" * 70)
            lgb_et_oof = np.zeros(n_train, dtype=np.float32)
            lgb_et_test = np.zeros(n_test, dtype=np.float32)
            lgb_et_fold_scores = []

            lgb_et_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'extra_trees': True,
                'num_leaves': 48,
                'learning_rate': 0.045,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 43,
                'n_jobs': 6,
                'verbose': -1
            }

            t_m2 = time.time()
            for fold in range(10):
                t_f = time.time()
                tr_mask = (fold_ids != fold)
                va_mask = (fold_ids == fold)

                te_dict = fold_te_data[fold]
                X_tr = np.hstack([X_base_train[tr_mask], te_dict['X_tr_te']])
                X_va = np.hstack([X_base_train[va_mask], te_dict['X_va_te']])
                X_te = np.hstack([X_base_test, te_dict['X_te_te']])
                y_tr = y_log1p[tr_mask]
                y_va = y_log1p[va_mask]

                trn_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
                val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, free_raw_data=False)

                model = lgb.train(
                    lgb_et_params,
                    trn_data,
                    num_boost_round=2000,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=80, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )

                va_pred = model.predict(X_va, num_iteration=model.best_iteration)
                te_pred = model.predict(X_te, num_iteration=model.best_iteration)

                lgb_et_oof[va_mask] = va_pred
                lgb_et_test += te_pred / 10.0

                score = float(np.sqrt(mean_squared_error(y_va, va_pred)))
                lgb_et_fold_scores.append(score)
                print(f"  [Fold {fold+1}/10] best_iter: {model.best_iteration:4d} | RMSLE: {score:.5f} | 耗时: {time.time() - t_f:.2f}s")

            overall_score = float(np.sqrt(mean_squared_error(y_log1p, lgb_et_oof)))
            print(f">>> [lgb_et 完成] 10-Fold Overall OOF RMSLE: {overall_score:.5f} (总耗时: {time.time() - t_m2:.2f}s)")
            np.save(lgb_et_oof_path, lgb_et_oof)
            np.save(lgb_et_test_path, lgb_et_test)
            verify_and_report_file(lgb_et_oof_path, 47293)
            verify_and_report_file(lgb_et_test_path, 11971)

    # =========================================================================
    # 模型 3: XGBoost Hist (xgb_main)
    # =========================================================================
    if args.model in ['all', 'xgb_main']:
        need_train = True
        if args.skip_existing and xgb_oof_path.exists() and xgb_test_path.exists():
            try:
                verify_and_report_file(xgb_oof_path, 47293)
                verify_and_report_file(xgb_test_path, 11971)
                print(">>> [xgb_main] 检测到已有完整预测文件且校验通过，跳过重新训练。")
                need_train = False
            except Exception as e:
                print(f">>> [xgb_main] 已有文件校验失败 ({e})，重新执行训练。")

        if need_train:
            xgb_device = check_xgboost_gpu()
            print("\n" + "=" * 70)
            print(f"⚡ [3/3] 开始训练 XGBoost Hist (xgb_main, device={xgb_device})...")
            print("=" * 70)
            xgb_oof = np.zeros(n_train, dtype=np.float32)
            xgb_test = np.zeros(n_test, dtype=np.float32)
            xgb_fold_scores = []

            t_m3 = time.time()
            for fold in range(10):
                t_f = time.time()
                tr_mask = (fold_ids != fold)
                va_mask = (fold_ids == fold)

                te_dict = fold_te_data[fold]
                X_tr = np.hstack([X_base_train[tr_mask], te_dict['X_tr_te']])
                X_va = np.hstack([X_base_train[va_mask], te_dict['X_va_te']])
                X_te = np.hstack([X_base_test, te_dict['X_te_te']])
                y_tr = y_log1p[tr_mask]
                y_va = y_log1p[va_mask]

                xgb_model = xgb.XGBRegressor(
                    n_estimators=2500,
                    learning_rate=0.03,
                    max_depth=7,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    min_child_weight=5,
                    tree_method='hist',
                    device=xgb_device,
                    early_stopping_rounds=100,
                    eval_metric='rmse',
                    random_state=42 + fold,
                    verbosity=0
                )

                xgb_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False
                )

                best_iter = getattr(xgb_model, 'best_iteration', -1)
                va_pred = xgb_model.predict(X_va)
                te_pred = xgb_model.predict(X_te)

                xgb_oof[va_mask] = va_pred
                xgb_test += te_pred / 10.0

                score = float(np.sqrt(mean_squared_error(y_va, va_pred)))
                xgb_fold_scores.append(score)
                print(f"  [Fold {fold+1}/10] best_iter: {best_iter:4d} | RMSLE: {score:.5f} | 耗时: {time.time() - t_f:.2f}s")

            overall_score = float(np.sqrt(mean_squared_error(y_log1p, xgb_oof)))
            print(f">>> [xgb_main 完成] 10-Fold Overall OOF RMSLE: {overall_score:.5f} (总耗时: {time.time() - t_m3:.2f}s)")
            np.save(xgb_oof_path, xgb_oof)
            np.save(xgb_test_path, xgb_test)
            verify_and_report_file(xgb_oof_path, 47293)
            verify_and_report_file(xgb_test_path, 11971)

    # =========================================================================
    # 汇总检查：如果 3 个模型的预测文件均已就绪，执行完整校验与就绪标志
    # =========================================================================
    all_files_exist = all(p.exists() for p in [
        lgb_std_oof_path, lgb_std_test_path,
        lgb_et_oof_path, lgb_et_test_path,
        xgb_oof_path, xgb_test_path
    ])

    if all_files_exist:
        print("\n" + "=" * 70)
        print("📊 [全部就绪] 开始进行 GBDT 家族动物园完整性验证与集成评估...")
        print("=" * 70)
        oof_std = verify_and_report_file(lgb_std_oof_path, 47293)
        test_std = verify_and_report_file(lgb_std_test_path, 11971)
        oof_et = verify_and_report_file(lgb_et_oof_path, 47293)
        test_et = verify_and_report_file(lgb_et_test_path, 11971)
        oof_xgb_arr = verify_and_report_file(xgb_oof_path, 47293)
        test_xgb_arr = verify_and_report_file(xgb_test_path, 11971)

        rmsle_std = float(np.sqrt(mean_squared_error(y_log1p, oof_std)))
        rmsle_et = float(np.sqrt(mean_squared_error(y_log1p, oof_et)))
        rmsle_xgb = float(np.sqrt(mean_squared_error(y_log1p, oof_xgb_arr)))

        # 多样性分析
        corr_std_et = float(np.corrcoef(oof_std, oof_et)[0, 1])
        corr_std_xgb = float(np.corrcoef(oof_std, oof_xgb_arr)[0, 1])
        corr_et_xgb = float(np.corrcoef(oof_et, oof_xgb_arr)[0, 1])

        print(f"  -> 单模 10-Fold OOF: lgb_std  = {rmsle_std:.5f}")
        print(f"  -> 单模 10-Fold OOF: lgb_et   = {rmsle_et:.5f}")
        print(f"  -> 单模 10-Fold OOF: xgb_main = {rmsle_xgb:.5f}")
        print(f"  -> 相关性矩阵: std vs et: {corr_std_et:.5f} | std vs xgb: {corr_std_xgb:.5f} | et vs xgb: {corr_et_xgb:.5f}")

        # 融合评估
        equal_blend = (oof_std + oof_et + oof_xgb_arr) / 3.0
        rmsle_equal = float(np.sqrt(mean_squared_error(y_log1p, equal_blend)))
        print(f"  -> 3 路 GBDT 等权融合 OOF RMSLE: {rmsle_equal:.5f} (较最佳单模提升: {min(rmsle_std, rmsle_et, rmsle_xgb) - rmsle_equal:+.5f})")

        # 严格校验 CV 分数是否在合理区间 (约 0.28~0.32，绝非泄漏假分)
        for m_name, rmsle_val in [('lgb_std', rmsle_std), ('lgb_et', rmsle_et), ('xgb_main', rmsle_xgb)]:
            assert 0.27 <= rmsle_val <= 0.35, f"异常 RMSLE: {m_name} 分数 {rmsle_val:.5f} 超出合理区间 [0.27, 0.35]！"
        print(f"  [OK] 全模型 CV RMSLE 严格合理性校验通过 (0.27 ~ 0.35 范围内)")

        # 写入 summary.json
        summary = {
            'status': 'success',
            'n_train': 47293,
            'n_test': 11971,
            'models': {
                'lgb_std': {'oof_rmsle': rmsle_std, 'file_oof': 'oof_lgb_std.npy', 'file_test': 'test_lgb_std.npy'},
                'lgb_et': {'oof_rmsle': rmsle_et, 'file_oof': 'oof_lgb_et.npy', 'file_test': 'test_lgb_et.npy'},
                'xgb_main': {'oof_rmsle': rmsle_xgb, 'file_oof': 'oof_xgb.npy', 'file_test': 'test_xgb.npy'},
            },
            'diversity_correlations': {
                'corr_lgb_std_lgb_et': corr_std_et,
                'corr_lgb_std_xgb_main': corr_std_xgb,
                'corr_lgb_et_xgb_main': corr_et_xgb
            },
            'ensemble_oof_rmsle': {
                'equal_blend_3gbdt': rmsle_equal
            },
            'total_time_seconds': round(time.time() - t_start, 2)
        }
        with open(OOF_DIR / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  [OK] 汇总指标已保存至: {OOF_DIR / 'summary.json'}")

        # 写入 READY_CLEAN.flag 与 READY.flag
        flag_clean = OOF_DIR / 'READY_CLEAN.flag'
        flag_clean.write_text("READY_CLEAN\n", encoding='utf-8')
        flag_legacy = OOF_DIR / 'READY.flag'
        flag_legacy.write_text("READY\n", encoding='utf-8')
        print(f"  [OK] 队伍交接信号已写入: {flag_clean} 与 {flag_legacy}")
    else:
        print("\n>>> 部分模型仍待训练或落盘，未写入 READY_CLEAN.flag。请继续运行未完成的模型。")

    print("\n" + "=" * 70)
    print(f"🎉 任务阶段运行结束！总耗时: {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()

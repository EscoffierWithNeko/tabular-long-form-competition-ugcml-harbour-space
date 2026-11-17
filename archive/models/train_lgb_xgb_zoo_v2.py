# -*- coding: utf-8 -*-
"""
train_lgb_xgb_zoo_v2.py -- 10-Fold GBDT 家族动物园 v2 训练与折外元特征落盘
===================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务: 基于无泄漏 v2 特征基座 (cache_10fold_v2/) 训练 4 路高多样性 GBDT 模型
      并产出严格对齐的 10-Fold OOF 预测与 Test 预测至 oof_gbdt_v2/

模型配置:
1. lgb_std: LightGBM 标准配置 (num_leaves=63, lr=0.03, colsample=0.7, subsample=0.8, early_stopping=100)
2. lgb_et: LightGBM Extra-Trees 随机切分 (extra_trees=True, num_leaves=48, lr=0.045, early_stopping=80)
3. xgb_main: XGBoost GPU/Hist 主力回归器 (max_depth=7, lr=0.03, colsample=0.7, subsample=0.8, early_stopping=100)
4. xgb_deep: XGBoost 深树高正则模型 (max_depth=9, lr=0.025, colsample=0.6, subsample=0.7, reg_alpha=1.5, reg_lambda=5.0, early_stopping=100)

严格防泄漏与数据校验原则:
- 读取 cache_10fold_v2/train_features.parquet, test_features.parquet, fold_ids.npy, y_log1p.npy
- 严格排除 'id' 与 'price' 列
- 严查特征与 y_log1p 相关性 (断言 max_corr < 0.95，杜绝假特征泄漏)
- 单模防假分数守卫 (0.22 < CV_RMSLE < 0.40, 无 NaN/Inf)
- 产出 oof_*.npy, test_*.npy, summary_v2.json 与 oof_gbdt_v2/READY_V2.flag
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / 'cache_10fold_v2'
OOF_DIR = ROOT_DIR / 'oof_gbdt_v2'


def check_xgboost_gpu():
    """检测 XGBoost CUDA 环境可用性"""
    try:
        clf = xgb.XGBRegressor(n_estimators=1, tree_method='hist', device='cuda')
        clf.fit(np.zeros((10, 2), dtype=np.float32), np.zeros(10, dtype=np.float32))
        return 'cuda'
    except Exception as e:
        print(f"XGBoost CUDA 不可用 ({e})，回退到 CPU")
        return 'cpu'


def audit_features_leakage(train_feat, test_feat, y_log1p, feature_cols):
    """严格审计特征防泄漏，验证排除 id/price 并核查最大相关系数"""
    print("\n>>> [安全审计] 执行无泄漏与标签隔离严格断言...")
    assert 'price' not in feature_cols, "Fatal: 'price' found in feature_cols!"
    assert 'id' not in feature_cols, "Fatal: 'id' found in feature_cols!"
    assert 'price' not in train_feat.columns, "Fatal: 'price' in train_feat!"
    assert 'price' not in test_feat.columns, "Fatal: 'price' in test_feat!"
    
    # 抽样计算数值列与 y_log1p 的 Pearson 相关系数
    print(">>> 正在检验所有特征与 target (y_log1p) 的相关系数...")
    max_corr = 0.0
    max_col = None
    y_std = np.std(y_log1p)
    assert y_std > 1e-6, "Fatal: y_log1p 方差异常接近 0!"
    
    for c in feature_cols:
        col_arr = train_feat[c].values
        col_std = np.std(col_arr)
        if col_std > 1e-6:
            r = abs(np.corrcoef(col_arr, y_log1p)[0, 1])
            if not np.isnan(r) and r > max_corr:
                max_corr = r
                max_col = c
                
    print(f"    最高特征相关度: {max_corr:.4f} (对应特征: {max_col})")
    assert max_corr < 0.95, f"Fatal: 疑似标签直接泄漏！特征 {max_col} 与 target 相关系数达 {max_corr:.4f} >= 0.95!"
    print("    [LEAKAGE AUDIT PASS] 所有特征相关系数均在安全阈值 (< 0.95) 内，无直接标签泄漏。")


def verify_and_report_file(file_path, expected_len):
    """校验落盘 npy 文件长度及无 NaN/Inf"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    arr = np.load(file_path)
    assert len(arr) == expected_len, f"文件 {file_path.name} 长度不匹配: 期望 {expected_len}, 实际 {len(arr)}"
    assert not np.isnan(arr).any(), f"文件 {file_path.name} 包含 NaN!"
    assert not np.isinf(arr).any(), f"文件 {file_path.name} 包含 Inf!"
    print(f"  [VERIFY OK] {file_path.name:20s} | 长度: {len(arr):5d} | 均值: {arr.mean():.4f} | std: {arr.std():.4f} | min: {arr.min():.4f} | max: {arr.max():.4f}")
    return arr


def parse_args():
    parser = argparse.ArgumentParser(description="Train GBDT Zoo v2 on 10-Fold")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "lgb_std", "lgb_et", "xgb_main", "xgb_deep"],
                        help="指定要训练的模型 (默认: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="若预测文件已存在且校验通过则跳过训练")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    print("=" * 80)
    print("🚀 [Step 2] 启动 GBDT 家族动物园 v2 训练与 10-Fold OOF 落盘管道")
    print(f"   目标模型: {args.model} | skip_existing: {args.skip_existing}")
    print("=" * 80)

    # 1. 检查基座缓存标志
    ready_flag = CACHE_DIR / 'READY_V2.flag'
    if not ready_flag.exists():
        raise FileNotFoundError(f"前置缓存标志不存在: {ready_flag}，请等待上游 v2 特征工程落盘完毕！")

    print(f">>> 正在从 {CACHE_DIR.name} 加载统一 10-Fold 零泄漏特征基座...")
    t0 = time.time()
    train_feat = pd.read_parquet(CACHE_DIR / 'train_features.parquet')
    test_feat = pd.read_parquet(CACHE_DIR / 'test_features.parquet')
    fold_ids = np.load(CACHE_DIR / 'fold_ids.npy')
    y_log1p = np.load(CACHE_DIR / 'y_log1p.npy')
    print(f">>> 基座加载完成 (耗时: {time.time() - t0:.2f}s)")
    print(f"    Train 特征矩阵形状: {train_feat.shape}, Test 形状: {test_feat.shape}")
    print(f"    fold_ids 形状: {fold_ids.shape}, y_log1p 形状: {y_log1p.shape}")

    n_train = len(train_feat)
    n_test = len(test_feat)
    assert len(y_log1p) == n_train, "y_log1p 长度与 train 不匹配!"
    assert len(fold_ids) == n_train, "fold_ids 长度与 train 不匹配!"

    # 2. 提取特征列（严格排除 id 与 price）
    feature_cols = [c for c in train_feat.columns if c not in ['id', 'price']]
    print(f">>> 纯数值特征总维度: {len(feature_cols)} 列")

    # 3. 执行严格防泄漏审计
    audit_features_leakage(train_feat, test_feat, y_log1p, feature_cols)

    # 提取纯数值 float32 矩阵以供模型高速拟合
    print(">>> 正在准备数值特征矩阵...")
    X_train_all = train_feat[feature_cols].values.astype(np.float32)
    X_test_all = test_feat[feature_cols].values.astype(np.float32)
    assert not np.isnan(X_train_all).any(), "X_train_all 含有 NaN!"
    assert not np.isinf(X_train_all).any(), "X_train_all 含有 Inf!"
    assert not np.isnan(X_test_all).any(), "X_test_all 含有 NaN!"
    assert not np.isinf(X_test_all).any(), "X_test_all 含有 Inf!"
    print(f"    X_train_all: {X_train_all.shape}, dtype: {X_train_all.dtype}")
    print(f"    X_test_all:  {X_test_all.shape}, dtype: {X_test_all.dtype}")

    # 创建输出目录
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    # 检查 XGBoost 加速设备
    xgb_device = check_xgboost_gpu()
    print(f">>> XGBoost 使用硬件设备: {xgb_device.upper()}")

    # 收集各模型评估结果
    cv_scores = {}
    oof_predictions = {}
    test_predictions = {}

    models_to_run = ["lgb_std", "lgb_et", "xgb_main", "xgb_deep"] if args.model == "all" else [args.model]

    # =========================================================================
    # 模型 1: LightGBM 标准 GBDT (lgb_std)
    # =========================================================================
    if "lgb_std" in models_to_run:
        oof_path = OOF_DIR / 'oof_lgb_std.npy'
        test_path = OOF_DIR / 'test_lgb_std.npy'
        need_train = True

        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof_lgb_std = verify_and_report_file(oof_path, n_train)
                test_lgb_std = verify_and_report_file(test_path, n_test)
                score = np.sqrt(mean_squared_error(y_log1p, oof_lgb_std))
                print(f">>> [lgb_std] 检测到已有有效产物，直接复用！OOF RMSLE: {score:.5f}")
                cv_scores['lgb_std'] = float(score)
                oof_predictions['lgb_std'] = oof_lgb_std
                test_predictions['lgb_std'] = test_lgb_std
                need_train = False
            except Exception as e:
                print(f">>> [lgb_std] 已有文件校验失败 ({e})，重新训练。")

        if need_train:
            print("\n" + "=" * 70)
            print("🌲 [1/4] 开始训练 LightGBM 标准 GBDT (lgb_std)...")
            print("=" * 70)

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

            oof_preds = np.zeros(n_train, dtype=np.float32)
            test_preds = np.zeros(n_test, dtype=np.float32)

            for fold in range(10):
                t_f = time.time()
                tr_mask = (fold_ids != fold)
                va_mask = (fold_ids == fold)

                X_tr, y_tr = X_train_all[tr_mask], y_log1p[tr_mask]
                X_va, y_va = X_train_all[va_mask], y_log1p[va_mask]

                trn_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
                val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, free_raw_data=False)

                model = lgb.train(
                    lgb_std_params,
                    trn_data,
                    num_boost_round=2500,
                    valid_sets=[trn_data, val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )

                fold_pred = model.predict(X_va, num_iteration=model.best_iteration)
                oof_preds[va_mask] = fold_pred
                test_preds += model.predict(X_test_all, num_iteration=model.best_iteration) / 10.0

                fold_score = np.sqrt(mean_squared_error(y_va, fold_pred))
                print(f"  [Fold {fold:02d}/10] best_iter: {model.best_iteration:4d} | va RMSLE: {fold_score:.5f} | 耗时: {time.time() - t_f:.1f}s")

            overall_score = np.sqrt(mean_squared_error(y_log1p, oof_preds))
            print(f">>> [lgb_std] 10-Fold CV RMSLE: {overall_score:.5f}")
            assert 0.22 < overall_score < 0.40, f"lgb_std 分数异常: {overall_score}"

            np.save(oof_path, oof_preds)
            np.save(test_path, test_preds)
            verify_and_report_file(oof_path, n_train)
            verify_and_report_file(test_path, n_test)

            cv_scores['lgb_std'] = float(overall_score)
            oof_predictions['lgb_std'] = oof_preds
            test_predictions['lgb_std'] = test_preds

    # =========================================================================
    # 模型 2: LightGBM Extra-Trees (lgb_et)
    # =========================================================================
    if "lgb_et" in models_to_run:
        oof_path = OOF_DIR / 'oof_lgb_et.npy'
        test_path = OOF_DIR / 'test_lgb_et.npy'
        need_train = True

        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof_lgb_et = verify_and_report_file(oof_path, n_train)
                test_lgb_et = verify_and_report_file(test_path, n_test)
                score = np.sqrt(mean_squared_error(y_log1p, oof_lgb_et))
                print(f">>> [lgb_et] 检测到已有有效产物，直接复用！OOF RMSLE: {score:.5f}")
                cv_scores['lgb_et'] = float(score)
                oof_predictions['lgb_et'] = oof_lgb_et
                test_predictions['lgb_et'] = test_lgb_et
                need_train = False
            except Exception as e:
                print(f">>> [lgb_et] 已有文件校验失败 ({e})，重新训练。")

        if need_train:
            print("\n" + "=" * 70)
            print("🌲 [2/4] 开始训练 LightGBM Extra-Trees (lgb_et)...")
            print("=" * 70)

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
                'n_jobs': 8,
                'verbose': -1
            }

            oof_preds = np.zeros(n_train, dtype=np.float32)
            test_preds = np.zeros(n_test, dtype=np.float32)

            for fold in range(10):
                t_f = time.time()
                tr_mask = (fold_ids != fold)
                va_mask = (fold_ids == fold)

                X_tr, y_tr = X_train_all[tr_mask], y_log1p[tr_mask]
                X_va, y_va = X_train_all[va_mask], y_log1p[va_mask]

                trn_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
                val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, free_raw_data=False)

                model = lgb.train(
                    lgb_et_params,
                    trn_data,
                    num_boost_round=2000,
                    valid_sets=[trn_data, val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=80, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )

                fold_pred = model.predict(X_va, num_iteration=model.best_iteration)
                oof_preds[va_mask] = fold_pred
                test_preds += model.predict(X_test_all, num_iteration=model.best_iteration) / 10.0

                fold_score = np.sqrt(mean_squared_error(y_va, fold_pred))
                print(f"  [Fold {fold:02d}/10] best_iter: {model.best_iteration:4d} | va RMSLE: {fold_score:.5f} | 耗时: {time.time() - t_f:.1f}s")

            overall_score = np.sqrt(mean_squared_error(y_log1p, oof_preds))
            print(f">>> [lgb_et] 10-Fold CV RMSLE: {overall_score:.5f}")
            assert 0.22 < overall_score < 0.40, f"lgb_et 分数异常: {overall_score}"

            np.save(oof_path, oof_preds)
            np.save(test_path, test_preds)
            verify_and_report_file(oof_path, n_train)
            verify_and_report_file(test_path, n_test)

            cv_scores['lgb_et'] = float(overall_score)
            oof_predictions['lgb_et'] = oof_preds
            test_predictions['lgb_et'] = test_preds

    # =========================================================================
    # 模型 3: XGBoost GPU/Hist 主力回归器 (xgb_main)
    # =========================================================================
    if "xgb_main" in models_to_run:
        oof_path = OOF_DIR / 'oof_xgb_main.npy'
        test_path = OOF_DIR / 'test_xgb_main.npy'
        need_train = True

        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof_xgb_main = verify_and_report_file(oof_path, n_train)
                test_xgb_main = verify_and_report_file(test_path, n_test)
                score = np.sqrt(mean_squared_error(y_log1p, oof_xgb_main))
                print(f">>> [xgb_main] 检测到已有有效产物，直接复用！OOF RMSLE: {score:.5f}")
                cv_scores['xgb_main'] = float(score)
                oof_predictions['xgb_main'] = oof_xgb_main
                test_predictions['xgb_main'] = test_xgb_main
                need_train = False
            except Exception as e:
                print(f">>> [xgb_main] 已有文件校验失败 ({e})，重新训练。")

        if need_train:
            print("\n" + "=" * 70)
            print("🚀 [3/4] 开始训练 XGBoost 主力回归器 (xgb_main)...")
            print("=" * 70)

            oof_preds = np.zeros(n_train, dtype=np.float32)
            test_preds = np.zeros(n_test, dtype=np.float32)

            for fold in range(10):
                t_f = time.time()
                tr_mask = (fold_ids != fold)
                va_mask = (fold_ids == fold)

                X_tr, y_tr = X_train_all[tr_mask], y_log1p[tr_mask]
                X_va, y_va = X_train_all[va_mask], y_log1p[va_mask]

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

                best_iter = xgb_model.best_iteration
                fold_pred = xgb_model.predict(X_va)
                oof_preds[va_mask] = fold_pred
                test_preds += xgb_model.predict(X_test_all) / 10.0

                fold_score = np.sqrt(mean_squared_error(y_va, fold_pred))
                print(f"  [Fold {fold:02d}/10] best_iter: {best_iter:4d} | va RMSLE: {fold_score:.5f} | 耗时: {time.time() - t_f:.1f}s")

            overall_score = np.sqrt(mean_squared_error(y_log1p, oof_preds))
            print(f">>> [xgb_main] 10-Fold CV RMSLE: {overall_score:.5f}")
            assert 0.22 < overall_score < 0.40, f"xgb_main 分数异常: {overall_score}"

            np.save(oof_path, oof_preds)
            np.save(test_path, test_preds)
            # 兼容性保存 oof_xgb.npy / test_xgb.npy
            np.save(OOF_DIR / 'oof_xgb.npy', oof_preds)
            np.save(OOF_DIR / 'test_xgb.npy', test_preds)

            verify_and_report_file(oof_path, n_train)
            verify_and_report_file(test_path, n_test)

            cv_scores['xgb_main'] = float(overall_score)
            oof_predictions['xgb_main'] = oof_preds
            test_predictions['xgb_main'] = test_preds

    # =========================================================================
    # 模型 4: XGBoost 深树高正则模型 (xgb_deep)
    # =========================================================================
    if "xgb_deep" in models_to_run:
        oof_path = OOF_DIR / 'oof_xgb_deep.npy'
        test_path = OOF_DIR / 'test_xgb_deep.npy'
        need_train = True

        if args.skip_existing and oof_path.exists() and test_path.exists():
            try:
                oof_xgb_deep = verify_and_report_file(oof_path, n_train)
                test_xgb_deep = verify_and_report_file(test_path, n_test)
                score = np.sqrt(mean_squared_error(y_log1p, oof_xgb_deep))
                print(f">>> [xgb_deep] 检测到已有有效产物，直接复用！OOF RMSLE: {score:.5f}")
                cv_scores['xgb_deep'] = float(score)
                oof_predictions['xgb_deep'] = oof_xgb_deep
                test_predictions['xgb_deep'] = test_xgb_deep
                need_train = False
            except Exception as e:
                print(f">>> [xgb_deep] 已有文件校验失败 ({e})，重新训练。")

        if need_train:
            print("\n" + "=" * 70)
            print("🌲 [4/4] 开始训练 XGBoost 深树高正则模型 (xgb_deep)...")
            print("=" * 70)

            oof_preds = np.zeros(n_train, dtype=np.float32)
            test_preds = np.zeros(n_test, dtype=np.float32)

            for fold in range(10):
                t_f = time.time()
                tr_mask = (fold_ids != fold)
                va_mask = (fold_ids == fold)

                X_tr, y_tr = X_train_all[tr_mask], y_log1p[tr_mask]
                X_va, y_va = X_train_all[va_mask], y_log1p[va_mask]

                xgb_deep_model = xgb.XGBRegressor(
                    n_estimators=2500,
                    learning_rate=0.025,
                    max_depth=9,
                    subsample=0.7,
                    colsample_bytree=0.6,
                    reg_alpha=1.5,
                    reg_lambda=5.0,
                    min_child_weight=8,
                    tree_method='hist',
                    device=xgb_device,
                    early_stopping_rounds=100,
                    eval_metric='rmse',
                    random_state=2024 + fold,
                    verbosity=0
                )

                xgb_deep_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False
                )

                best_iter = xgb_deep_model.best_iteration
                fold_pred = xgb_deep_model.predict(X_va)
                oof_preds[va_mask] = fold_pred
                test_preds += xgb_deep_model.predict(X_test_all) / 10.0

                fold_score = np.sqrt(mean_squared_error(y_va, fold_pred))
                print(f"  [Fold {fold:02d}/10] best_iter: {best_iter:4d} | va RMSLE: {fold_score:.5f} | 耗时: {time.time() - t_f:.1f}s")

            overall_score = np.sqrt(mean_squared_error(y_log1p, oof_preds))
            print(f">>> [xgb_deep] 10-Fold CV RMSLE: {overall_score:.5f}")
            assert 0.22 < overall_score < 0.40, f"xgb_deep 分数异常: {overall_score}"

            np.save(oof_path, oof_preds)
            np.save(test_path, test_preds)

            verify_and_report_file(oof_path, n_train)
            verify_and_report_file(test_path, n_test)

            cv_scores['xgb_deep'] = float(overall_score)
            oof_predictions['xgb_deep'] = oof_preds
            test_predictions['xgb_deep'] = test_preds

    # =========================================================================
    # 多模型综合评估与融合验证
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 [GBDT Zoo v2] 评估总结与简单融合指标")
    print("=" * 80)

    summary_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": round(time.time() - t_start, 2),
        "cv_scores": cv_scores,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": len(feature_cols),
        "device": xgb_device
    }

    for name, score in cv_scores.items():
        print(f"  * {name:10s} 10-Fold CV RMSLE: {score:.5f}")

    # 若跑了全部模型，计算模型两两相关系数矩阵与等权融合分
    available_models = [m for m in ["lgb_std", "lgb_et", "xgb_main", "xgb_deep"] if m in oof_predictions]
    if len(available_models) >= 2:
        print("\n>>> 模型预测 Pearson 相关系数矩阵 (Diversity Matrix):")
        matrix_header = f"{'':12s}" + "".join([f"{m:>12s}" for m in available_models])
        print(matrix_header)
        corr_matrix = {}
        for m1 in available_models:
            row_str = f"{m1:12s}"
            corr_matrix[m1] = {}
            for m2 in available_models:
                r = float(np.corrcoef(oof_predictions[m1], oof_predictions[m2])[0, 1])
                corr_matrix[m1][m2] = round(r, 5)
                row_str += f"{r:>12.4f}"
            print(row_str)
        summary_results["correlation_matrix"] = corr_matrix

        # 1. 经典 3-GBDT 等权平均 (lgb_std + lgb_et + xgb_main)
        trio = [m for m in ["lgb_std", "lgb_et", "xgb_main"] if m in oof_predictions]
        if len(trio) == 3:
            oof_trio = np.mean([oof_predictions[m] for m in trio], axis=0)
            score_trio = float(np.sqrt(mean_squared_error(y_log1p, oof_trio)))
            print(f"\n>>> [Equal Blend 3-GBDT] (lgb_std + lgb_et + xgb_main) RMSLE: {score_trio:.5f}")
            summary_results["blend_3gbdt_rmsle"] = score_trio

        # 2. 全量 4-GBDT 等权平均
        if len(available_models) == 4:
            oof_all = np.mean([oof_predictions[m] for m in available_models], axis=0)
            score_all = float(np.sqrt(mean_squared_error(y_log1p, oof_all)))
            print(f">>> [Equal Blend 4-GBDT] (All 4 models) RMSLE: {score_all:.5f}")
            summary_results["blend_4gbdt_rmsle"] = score_all

    # 落盘 summary_v2.json
    summary_path = OOF_DIR / 'summary_v2.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"\n>>> 实验元数据已落盘: {summary_path}")

    # 落盘 READY_V2.flag
    flag_path = OOF_DIR / 'READY_V2.flag'
    with open(flag_path, 'w', encoding='utf-8') as f:
        f.write(f"GBDT Zoo v2 Training Completed.\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k, v in cv_scores.items():
            f.write(f"{k}: {v:.5f}\n")
    print(f">>> 成功落盘就绪标志: {flag_path}")
    print(f">>> 全部任务圆满完成，总耗时: {time.time() - t_start:.2f}s")


if __name__ == '__main__':
    main()

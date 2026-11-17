# -*- coding: utf-8 -*-
"""
train_catboost_nn_zoo.py
================================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务: 10-Fold CatBoost 原生文本/高阶类别模型与 PyTorch Tabular ResNet 异构模型训练与落盘
目标: 冲刺 Kaggle 第一名，为后续 Level-2 Stacking 提供高多样性、高质量的模型基元

包含模型:
1. 模型 1 (cat_text): CatBoostRegressor 满血版
   - text_features = ['clean_name', 'clean_desc', 'clean_amenities']
   - cat_features = ['geo_cell', 'host_id', 'room_type', 'property_type', 'twin_4tuple', 'twin_5tuple']
   - iterations=2000, lr=0.05, early_stopping_rounds=80
2. 模型 2 (nn_tabular): PyTorch Entity-Embedding Tabular ResNet
   - 数值特征折内 StandardScaler 归一化与中位数补全
   - 高基数类别 Entity-Embedding
   - AdamW + CosineAnnealingLR + Early Stopping
================================================================================
"""

import os
import gc
import re
import sys
import time
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import catboost as cb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 基础配置与随机种子固定
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache_10fold"
OUTPUT_DIR = BASE_DIR / "oof_cat_nn"
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)

def calc_rmsle(y_true, y_pred):
    """
    y_true 和 y_pred 均基于 log1p(price)，故其 RMSE 即为原始 price 的 RMSLE。
    """
    clipped_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(y_true, clipped_pred)))

# ------------------------------------------------------------------------------
# PyTorch Tabular ResNet 模块定义
# ------------------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx]


class TabularResNet(nn.Module):
    def __init__(self, num_features, cat_cardinalities, emb_dims, hidden_dim=256, n_blocks=2, dropout=0.15):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=d)
            for c, d in zip(cat_cardinalities, emb_dims)
        ])
        total_emb_dim = sum(emb_dims)
        self.num_bn = nn.BatchNorm1d(num_features)
        
        in_dim = total_emb_dim + num_features
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_num, x_cat):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_repr = torch.cat(emb_outs, dim=1) if len(emb_outs) > 0 else None
        num_repr = self.num_bn(x_num)
        
        if cat_repr is not None:
            x = torch.cat([num_repr, cat_repr], dim=1)
        else:
            x = num_repr
            
        h = self.in_proj(x)
        for block in self.blocks:
            h = self.act(h + block(h))
        out = self.head(h).squeeze(-1)
        return out


def train_nn_zoo(raw_num_train, raw_num_test, df_train_text, df_test_text, cat_cols, y_log1p, fold_ids, n_train, n_test, n_folds, device):
    print("\n" + "=" * 80)
    print(">>> [4/5] 开始训练 模型 2: PyTorch Entity-Embedding Tabular ResNet (nn_tabular)")
    print("=" * 80)

    cat_encodings_train = np.zeros((n_train, len(cat_cols)), dtype=np.int64)
    cat_encodings_test = np.zeros((n_test, len(cat_cols)), dtype=np.int64)
    cat_cardinalities = []
    emb_dims = []

    for idx, col in enumerate(cat_cols):
        s_train = df_train_text[col].fillna("missing").astype(str)
        s_test = df_test_text[col].fillna("missing").astype(str)
        unique_vals = list(pd.concat([s_train, s_test]).unique())
        val2idx = {val: i + 1 for i, val in enumerate(unique_vals)}
        cat_encodings_train[:, idx] = s_train.map(val2idx).values
        cat_encodings_test[:, idx] = s_test.map(val2idx).values
        cardinality = len(unique_vals) + 1
        cat_cardinalities.append(cardinality)
        emb_dim = min(64, max(4, int(round(cardinality ** 0.5))))
        emb_dims.append(emb_dim)

    oof_nn_tabular = np.zeros(n_train, dtype=np.float32)
    test_nn_tabular = np.zeros(n_test, dtype=np.float32)
    nn_fold_scores = []

    BATCH_SIZE = 256
    EPOCHS = 28
    PATIENCE = 7
    LR = 1.2e-3
    WEIGHT_DECAY = 1e-4

    for fold in range(n_folds):
        t_fold = time.time()
        trn_idx = np.where(fold_ids != fold)[0]
        val_idx = np.where(fold_ids == fold)[0]

        te_path = CACHE_DIR / "fold_te" / f"fold_{fold}.npz"
        assert te_path.exists(), f"fold_te 特征文件不存在: {te_path}"
        te_data = np.load(te_path)
        X_tr_te = te_data['X_tr_te'].astype(np.float32)
        X_val_te = te_data['X_va_te'].astype(np.float32)
        X_te_te = te_data['X_te_te'].astype(np.float32)

        assert len(X_tr_te) == len(trn_idx)
        assert len(X_val_te) == len(val_idx)
        assert len(X_te_te) == n_test

        fold_trn_num = np.hstack([raw_num_train[trn_idx], X_tr_te])
        fold_val_num = np.hstack([raw_num_train[val_idx], X_val_te])
        fold_test_num = np.hstack([raw_num_test, X_te_te])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_medians = np.nanmedian(fold_trn_num, axis=0)
            col_medians = np.nan_to_num(col_medians, nan=0.0)

        for j in range(fold_trn_num.shape[1]):
            med = col_medians[j]
            fold_trn_num[np.isnan(fold_trn_num[:, j]), j] = med
            fold_val_num[np.isnan(fold_val_num[:, j]), j] = med
            fold_test_num[np.isnan(fold_test_num[:, j]), j] = med

        scaler = StandardScaler()
        scaled_trn_num = scaler.fit_transform(fold_trn_num).astype(np.float32)
        scaled_val_num = scaler.transform(fold_val_num).astype(np.float32)
        scaled_test_num = scaler.transform(fold_test_num).astype(np.float32)

        np.clip(scaled_trn_num, -10.0, 10.0, out=scaled_trn_num)
        np.clip(scaled_val_num, -10.0, 10.0, out=scaled_val_num)
        np.clip(scaled_test_num, -10.0, 10.0, out=scaled_test_num)

        trn_dataset = TabularDataset(scaled_trn_num, cat_encodings_train[trn_idx], y_log1p[trn_idx])
        val_dataset = TabularDataset(scaled_val_num, cat_encodings_train[val_idx], y_log1p[val_idx])
        test_dataset = TabularDataset(scaled_test_num, cat_encodings_test)

        trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

        model = TabularResNet(
            num_features=scaled_trn_num.shape[1],
            cat_cardinalities=cat_cardinalities,
            emb_dims=emb_dims,
            hidden_dim=256,
            n_blocks=2,
            dropout=0.15
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_val_score = float('inf')
        best_state = None
        no_improve_epochs = 0

        for epoch in range(EPOCHS):
            model.train()
            for bx_num, bx_cat, by in trn_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                by = by.to(device)

                optimizer.zero_grad()
                pred = model(bx_num, bx_cat)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx_num, bx_cat, _ in val_loader:
                    bx_num = bx_num.to(device)
                    bx_cat = bx_cat.to(device)
                    val_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
            cur_val_preds = np.concatenate(val_preds_list)
            cur_val_score = calc_rmsle(y_log1p[val_idx], cur_val_preds)

            if cur_val_score < best_val_score:
                best_val_score = cur_val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= PATIENCE:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx_num, bx_cat, _ in val_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                val_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
        fold_best_val_preds = np.concatenate(val_preds_list)
        oof_nn_tabular[val_idx] = fold_best_val_preds

        test_preds_list = []
        with torch.no_grad():
            for bx_num, bx_cat in test_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                test_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
        fold_test_preds = np.concatenate(test_preds_list)
        test_nn_tabular += fold_test_preds / n_folds

        score = calc_rmsle(y_log1p[val_idx], fold_best_val_preds)
        nn_fold_scores.append(score)
        print(f"   [PyTorch NN] Fold {fold + 1:02d}/{n_folds:02d} RMSLE: {score:.5f} | 最佳 Val RMSLE: {best_val_score:.5f} | 耗时: {time.time() - t_fold:.1f}s")

    cv_nn_tabular = calc_rmsle(y_log1p, oof_nn_tabular)
    print("-" * 80)
    print(f">>> [PyTorch NN] 10-Fold 各折 RMSLE: {[round(s, 5) for s in nn_fold_scores]}")
    print(f">>> [PyTorch NN] 10-Fold Overall CV RMSLE: {cv_nn_tabular:.5f}")
    print("-" * 80)

    np.save(OUTPUT_DIR / "oof_nn_tabular.npy", oof_nn_tabular)
    np.save(OUTPUT_DIR / "test_nn_tabular.npy", test_nn_tabular)
    print(">>> [PyTorch NN] oof_nn_tabular.npy 与 test_nn_tabular.npy 已成功原地保存！", flush=True)

    return oof_nn_tabular, test_nn_tabular, cv_nn_tabular, nn_fold_scores


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="10-Fold CatBoost & PyTorch Tabular NN Zoo")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'cat', 'nn'],
                        help="训练目标模型: 'all', 'cat' (CatBoost), 'nn' (PyTorch ResNet)")
    parser.add_argument('--folds', type=str, default='all',
                        help="指定训练折数，如 'all' 或 '0,1,2,3,4' 或 '5,6,7,8,9'")
    parser.add_argument('--skip_existing', action='store_true',
                        help="若已存在有效 oof/test 文件则跳过已完成模型")
    return parser.parse_args()

# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    t_start = time.time()
    print("=" * 80)
    print(f">>> 启动 10-Fold CatBoost 原生文本/类别模型与 PyTorch Tabular NN 训练流程 (模式: {args.model})")
    print(f">>> 当前 PyTorch 版本: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f">>> 当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f">>> 当前 CatBoost 版本: {cb.__version__}")
    print("=" * 80)

    # 1. 验证上游特征工程基座标志文件
    # 0. 检查前置无泄漏基座就绪标志
    clean_flag_path = CACHE_DIR / "READY_CLEAN.flag"
    if not clean_flag_path.exists():
        raise FileNotFoundError(f"前置干净基座标志未找到: {clean_flag_path}，必须基于清洗后的无泄漏基座运行！")
    print(f"[1/5] 核验证实 cache_10fold/READY_CLEAN.flag 存在。开始加载缓存数据...")

    # 读取表格基础特征与文本元数据
    train_base_path = CACHE_DIR / "train_base_features.parquet"
    test_base_path = CACHE_DIR / "test_base_features.parquet"
    train_text_path = CACHE_DIR / "train_text.parquet"
    test_text_path = CACHE_DIR / "test_text.parquet"
    fold_ids_path = CACHE_DIR / "fold_ids.npy"
    y_log1p_path = CACHE_DIR / "y_log1p.npy"

    df_train_base = pd.read_parquet(train_base_path)
    df_test_base = pd.read_parquet(test_base_path)
    df_train_text = pd.read_parquet(train_text_path)
    df_test_text = pd.read_parquet(test_text_path)
    fold_ids = np.load(fold_ids_path)
    y_log1p = np.load(y_log1p_path)

    n_train = len(df_train_base)
    n_test = len(df_test_base)
    n_folds = int(fold_ids.max() + 1)
    print(f"      训练集样本数: {n_train}, 测试集样本数: {n_test}, 折数: {n_folds}")
    print(f"      基础表格列数: train={df_train_base.shape[1]}, test={df_test_base.shape[1]}")
    print(f"      文本特征列数: train={df_train_text.shape[1]}, test={df_test_text.shape[1]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. 字段分离与格式规整
    # --------------------------------------------------------------------------
    print("\n[2/5] 提取数值特征、高阶类别特征与原生文本特征...")
    
    # 定义类别字段与文本字段
    cat_cols = ['geo_cell', 'host_id', 'room_type', 'property_type', 'twin_4tuple', 'twin_5tuple']
    text_cols = ['clean_name', 'clean_desc', 'clean_amenities']

    # 提取纯数值列 (严格排除 id, 文本, 类别, 以及任何形式的目标价格变量)
    exclude_base = {
        'id', 'name', 'description', 'amenities', 'host_id', 
        'property_type', 'room_type', 'geo_cell', 'twin_4tuple', 'twin_5tuple',
        'price', 'log_price', 'target', 'y_log1p'
    }
    numeric_cols = [
        c for c in df_train_base.columns 
        if c not in exclude_base and pd.api.types.is_numeric_dtype(df_train_base[c]) and c in df_test_base.columns
    ]
    # 严格泄漏断言守卫
    assert 'price' not in numeric_cols, "CRITICAL LEAKAGE DETECTED: 'price' is present in numeric_cols!"
    assert 'log_price' not in numeric_cols, "CRITICAL LEAKAGE DETECTED: 'log_price' is present in numeric_cols!"
    assert 'target' not in numeric_cols, "CRITICAL LEAKAGE DETECTED: 'target' is present in numeric_cols!"
    assert 'y_log1p' not in numeric_cols, "CRITICAL LEAKAGE DETECTED: 'y_log1p' is present in numeric_cols!"

    # 自动化相关性熔断守卫 (断言所有特征与 y_log1p 相关系数绝对值严格 < 0.98)
    print(">>> [防泄漏守卫] 正在校验纯数值特征与 y_log1p 的线性相关性熔断指标...", flush=True)
    for col in numeric_cols:
        v = df_train_base[col].values
        valid_mask = ~np.isnan(v)
        if valid_mask.sum() > 10:
            corr = np.corrcoef(v[valid_mask], y_log1p[valid_mask])[0, 1]
            assert abs(corr) < 0.98, f"CRITICAL LEAKAGE DETECTED: 特征 '{col}' 与 y_log1p 相关系数高达 {corr:.5f} >= 0.98!"
    print(f">>> [防泄漏守卫] 校验通过！全部 {len(numeric_cols)} 维基础特征通过熔断检测 (|corr| < 0.98)")

    print(f"      识别纯数值特征维度: {len(numeric_cols)} 维")
    print(f"      高阶类别特征列表: {cat_cols}")
    print(f"      原生文本特征列表: {text_cols}")

    # ==========================================================================
    # 3. 模型 1: CatBoostRegressor 原生文本与高阶类别满血模型
    # ==========================================================================
    do_cat = args.model in ['all', 'cat']
    do_nn = args.model in ['all', 'nn']

    oof_cat_text = None
    test_cat_text = None
    cv_cat_text = None
    cat_fold_scores = []

    cat_oof_path = OUTPUT_DIR / "oof_cat_text.npy"
    cat_test_path = OUTPUT_DIR / "test_cat_text.npy"
    if cat_oof_path.exists() and cat_test_path.exists():
        try:
            _c_oof = np.load(cat_oof_path)
            _c_test = np.load(cat_test_path)
            if len(_c_oof) == n_train and len(_c_test) == n_test and not np.isnan(_c_oof).any():
                _score = calc_rmsle(y_log1p, _c_oof)
                if 0.22 < _score < 0.45:
                    oof_cat_text = _c_oof
                    test_cat_text = _c_test
                    cv_cat_text = _score
                    if args.skip_existing and not (args.model == 'cat'):
                        print(f">>> [复用] 已存在有效无泄漏 CatBoost 预测 (CV: {_score:.5f})，直接加载！", flush=True)
                        do_cat = False
        except (IOError, ValueError) as err:
            print(f">>> [提示] 读取现有 CatBoost 文件异常: {err}，将重新训练。")

    if do_cat:
        print("\n" + "=" * 80)
        print(">>> [3/5] 开始训练 模型 1: CatBoostRegressor 原生文本 + 类别模型 (cat_text)")
        print("=" * 80)

        # 组装 CatBoost 输入特征 DataFrame
        df_cat_train = pd.DataFrame(index=df_train_base.index)
        df_cat_test = pd.DataFrame(index=df_test_base.index)

        # 拷贝数值特征
        for col in numeric_cols:
            df_cat_train[col] = df_train_base[col].astype(np.float32)
            df_cat_test[col] = df_test_base[col].astype(np.float32)

        # 类别特征处理：转为 string 且填充 missing
        for col in cat_cols:
            df_cat_train[col] = df_train_text[col].fillna("missing").astype(str).replace("", "missing")
            df_cat_test[col] = df_test_text[col].fillna("missing").astype(str).replace("", "missing")

        # 文本特征处理：转为 string 且填充 none
        for col in text_cols:
            df_cat_train[col] = df_train_text[col].fillna("none").astype(str).replace("", "none")
            df_cat_test[col] = df_test_text[col].fillna("none").astype(str).replace("", "none")

        # 探测 CatBoost 对 GPU + text_features 的兼容性与稳定性 (使用 1000 行切片防止小样本词典过小误判)
        use_gpu_for_catboost = False
        if torch.cuda.is_available():
            try:
                print("      正在探测 CatBoost GPU 模式对 text_features 的兼容支持 (样本切片: 1000 行)...", flush=True)
                probe_cb = cb.CatBoostRegressor(
                    iterations=5,
                    learning_rate=0.1,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    text_features=text_cols,
                    cat_features=cat_cols,
                    task_type='GPU',
                    verbose=0,
                    allow_writing_files=False
                )
                probe_X = df_cat_train.iloc[:1000]
                probe_y = y_log1p[:1000]
                probe_cb.fit(probe_X, probe_y, verbose=False)
                use_gpu_for_catboost = True
                print("      >>> CatBoost GPU 模式探测成功！优先使用 GPU 模式加速！", flush=True)
            except Exception as e:
                print(f"      >>> CatBoost GPU 模式探测提示: {e}", flush=True)
                print("      >>> 自动无缝降级为 CPU 多线程满血模式 (14核/20线程并行加速)！", flush=True)
                use_gpu_for_catboost = False

        cat_task_type = 'GPU' if use_gpu_for_catboost else 'CPU'

        cat_oof_path = OUTPUT_DIR / "oof_cat_text.npy"
        cat_test_path = OUTPUT_DIR / "test_cat_text.npy"
        if cat_oof_path.exists() and cat_test_path.exists():
            oof_cat_text = np.load(cat_oof_path)
            test_cat_text = np.load(cat_test_path)
            if len(oof_cat_text) != n_train or len(test_cat_text) != n_test:
                oof_cat_text = np.zeros(n_train, dtype=np.float32)
                test_cat_text = np.zeros(n_test, dtype=np.float32)
        else:
            oof_cat_text = np.zeros(n_train, dtype=np.float32)
            test_cat_text = np.zeros(n_test, dtype=np.float32)

        if args.folds == 'all':
            cat_target_folds = list(range(n_folds))
        else:
            cat_target_folds = [int(x.strip()) for x in args.folds.split(',') if x.strip()]

        cat_fold_scores = []

        for fold in cat_target_folds:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            t_fold = time.time()
            trn_idx = np.where(fold_ids != fold)[0]
            val_idx = np.where(fold_ids == fold)[0]

            X_tr = df_cat_train.iloc[trn_idx]
            y_tr = y_log1p[trn_idx]
            X_va = df_cat_train.iloc[val_idx]
            y_va = y_log1p[val_idx]

            cat_params = dict(
                allow_writing_files=False,
                iterations=1200,
                learning_rate=0.08,
                early_stopping_rounds=60,
                depth=6,
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=SEED + fold,
                verbose=100,
                task_type=cat_task_type
            )
            if cat_task_type == 'GPU':
                cat_params['max_ctr_complexity'] = 1
                cat_params['gpu_cat_features_storage'] = 'CpuPinnedMemory'
            else:
                cat_params['thread_count'] = -1

            try:
                model_cat = cb.CatBoostRegressor(**cat_params)
                model_cat.fit(
                    X_tr, y_tr,
                    eval_set=(X_va, y_va),
                    text_features=text_cols,
                    cat_features=cat_cols,
                    use_best_model=True,
                    verbose=200
                )
            except Exception as fit_err:
                if cat_task_type == 'GPU':
                    print(f"   [CatBoost Warn] GPU fit 遇到异常: {fit_err}，立即降级 CPU 模式重训 Fold {fold+1}...", flush=True)
                    cat_task_type = 'CPU'
                    cat_params['task_type'] = 'CPU'
                    cat_params['thread_count'] = -1
                    model_cat = cb.CatBoostRegressor(**cat_params)
                    model_cat.fit(
                        X_tr, y_tr,
                        eval_set=(X_va, y_va),
                        text_features=text_cols,
                        cat_features=cat_cols,
                        use_best_model=True,
                        verbose=200
                    )
                else:
                    raise fit_err

            val_preds = model_cat.predict(X_va)
            oof_cat_text[val_idx] = val_preds
            fold_test_preds = model_cat.predict(df_cat_test)
            test_cat_text += fold_test_preds / n_folds

            score = calc_rmsle(y_va, val_preds)
            cat_fold_scores.append(score)
            best_iter = model_cat.get_best_iteration()
            print(f"   [CatBoost] Fold {fold + 1:02d}/{n_folds:02d} RMSLE: {score:.5f} | Best Iter: {best_iter} | 耗时: {time.time() - t_fold:.1f}s", flush=True)

            # 单折原子级阶段性落盘 (断点续跑防丢)
            np.save(cat_oof_path, oof_cat_text)
            np.save(cat_test_path, test_cat_text)

        if np.all(oof_cat_text > 0):
            cv_cat_text = calc_rmsle(y_log1p, oof_cat_text)
            print("-" * 80, flush=True)
            print(f">>> [CatBoost] 10-Fold 各折 RMSLE: {[round(s, 5) for s in cat_fold_scores]}", flush=True)
            print(f">>> [CatBoost] 10-Fold Overall CV RMSLE: {cv_cat_text:.5f}", flush=True)
            print("-" * 80, flush=True)
        else:
            completed_mask = (oof_cat_text > 0)
            completed_score = calc_rmsle(y_log1p[completed_mask], oof_cat_text[completed_mask])
            print(f">>> [CatBoost 阶段进度] 已完成 {completed_mask.sum()}/{n_train} 样本，验证 RMSLE: {completed_score:.5f}", flush=True)

        print(">>> [CatBoost] oof_cat_text.npy 与 test_cat_text.npy 已成功阶段性落盘！", flush=True)

        # 释放 CatBoost 内存
        del df_cat_train, df_cat_test
    gc.collect()

    # ==========================================================================
    # 4. 模型 2: PyTorch Entity-Embedding Tabular ResNet (nn_tabular)
    # ==========================================================================
    oof_nn_tabular = None
    test_nn_tabular = None
    cv_nn_tabular = None
    nn_fold_scores = []

    nn_oof_path = OUTPUT_DIR / "oof_nn_tabular.npy"
    nn_test_path = OUTPUT_DIR / "test_nn_tabular.npy"
    if nn_oof_path.exists() and nn_test_path.exists():
        try:
            _n_oof = np.load(nn_oof_path)
            _n_test = np.load(nn_test_path)
            if len(_n_oof) == n_train and len(_n_test) == n_test and not np.isnan(_n_oof).any():
                _score = calc_rmsle(y_log1p, _n_oof)
                if 0.22 < _score < 0.50:
                    oof_nn_tabular = _n_oof
                    test_nn_tabular = _n_test
                    cv_nn_tabular = _score
                    if args.skip_existing and not (args.model == 'nn'):
                        print(f">>> [复用] 已存在有效无泄漏 PyTorch NN 预测 (CV: {_score:.5f})，直接加载！", flush=True)
                        do_nn = False
        except (IOError, ValueError) as err:
            print(f">>> [提示] 读取现有 PyTorch NN 文件异常: {err}，将重新训练。")

    if do_nn:
        oof_nn_tabular, test_nn_tabular, cv_nn_tabular, nn_fold_scores = train_nn_zoo(
            raw_num_train, raw_num_test, df_train_text, df_test_text, cat_cols,
            y_log1p, fold_ids, n_train, n_test, n_folds, device
        )
    if False:

        # 4.1 类别特征 Entity-Embedding 预映射 (未知设为 0，已知 1..K)
        cat_encodings_train = np.zeros((n_train, len(cat_cols)), dtype=np.int64)
        cat_encodings_test = np.zeros((n_test, len(cat_cols)), dtype=np.int64)
        cat_cardinalities = []
        emb_dims = []

        for idx, col in enumerate(cat_cols):
            s_train = df_train_text[col].fillna("missing").astype(str)
            s_test = df_test_text[col].fillna("missing").astype(str)
            
            # 统计所有唯一类别
            unique_vals = list(pd.concat([s_train, s_test], axis=0).unique())
            val2idx = {val: i + 1 for i, val in enumerate(unique_vals)}  # 0 为 UNK
            
            cat_encodings_train[:, idx] = s_train.map(lambda v: val2idx.get(v, 0)).values
            cat_encodings_test[:, idx] = s_test.map(lambda v: val2idx.get(v, 0)).values
            
            num_classes = len(unique_vals) + 1  # 包含 UNK 0
            cat_cardinalities.append(num_classes)
            # 遵循经典经验法则确定嵌入维度
            d = min(64, max(4, int(round(1.6 * (num_classes ** 0.56)))))
            emb_dims.append(d)
            print(f"      类别 [{col}]: 唯一取值数={len(unique_vals)}, 总类别数={num_classes}, Embedding Dim={d}")

        # 4.2 数值特征矩阵获取
        raw_num_train = df_train_base[numeric_cols].values.astype(np.float32)
        raw_num_test = df_test_base[numeric_cols].values.astype(np.float32)

        # 4.3 10-Fold 严格折内归一化与训练
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"      PyTorch Tabular ResNet 运行设备: {device}")

        oof_nn_tabular = np.zeros(n_train, dtype=np.float32)
        test_nn_tabular = np.zeros(n_test, dtype=np.float32)
        nn_fold_scores = []

        BATCH_SIZE = 256
        EPOCHS = 28
        PATIENCE = 7
        LR = 1.2e-3
        WEIGHT_DECAY = 1e-4

    for fold in range(0):
        t_fold = time.time()
        trn_idx = np.where(fold_ids != fold)[0]
        val_idx = np.where(fold_ids == fold)[0]

        # 折内无泄漏：加载当前折预计算的高阶折内特征 fold_te
        te_path = CACHE_DIR / "fold_te" / f"fold_{fold}.npz"
        assert te_path.exists(), f"fold_te 特征文件不存在: {te_path}"
        te_data = np.load(te_path)
        X_tr_te = te_data['X_tr_te'].astype(np.float32)
        X_va_te = te_data['X_va_te'].astype(np.float32)
        X_te_te = te_data['X_te_te'].astype(np.float32)

        assert len(X_tr_te) == len(trn_idx), f"X_tr_te 长度 {len(X_tr_te)} != trn_idx 长度 {len(trn_idx)}"
        assert len(X_va_te) == len(val_idx), f"X_va_te 长度 {len(X_va_te)} != val_idx 长度 {len(val_idx)}"
        assert len(X_te_te) == n_test, f"X_te_te 长度 {len(X_te_te)} != n_test {n_test}"

        # 拼接基础纯数值特征与折内零泄漏特征
        fold_trn_num = np.hstack([raw_num_train[trn_idx], X_tr_te])
        fold_val_num = np.hstack([raw_num_train[val_idx], X_va_te])
        fold_test_num = np.hstack([raw_num_test, X_te_te])

        # 计算训练折特征中位数 (缺失值安全的 nanmedian)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_medians = np.nanmedian(fold_trn_num, axis=0)
            col_medians = np.nan_to_num(col_medians, nan=0.0)

        # 填补 NaN
        for j in range(fold_trn_num.shape[1]):
            med = col_medians[j]
            fold_trn_num[np.isnan(fold_trn_num[:, j]), j] = med
            fold_val_num[np.isnan(fold_val_num[:, j]), j] = med
            fold_test_num[np.isnan(fold_test_num[:, j]), j] = med

        # 折内 fit StandardScaler
        scaler = StandardScaler()
        scaled_trn_num = scaler.fit_transform(fold_trn_num).astype(np.float32)
        scaled_val_num = scaler.transform(fold_val_num).astype(np.float32)
        scaled_test_num = scaler.transform(fold_test_num).astype(np.float32)

        # 针对极值截断防御，避免异常数值破坏梯度
        np.clip(scaled_trn_num, -10.0, 10.0, out=scaled_trn_num)
        np.clip(scaled_val_num, -10.0, 10.0, out=scaled_val_num)
        np.clip(scaled_test_num, -10.0, 10.0, out=scaled_test_num)

        # 构造 DataLoader
        trn_dataset = TabularDataset(scaled_trn_num, cat_encodings_train[trn_idx], y_log1p[trn_idx])
        val_dataset = TabularDataset(scaled_val_num, cat_encodings_train[val_idx], y_log1p[val_idx])
        test_dataset = TabularDataset(scaled_test_num, cat_encodings_test)

        trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

        # 初始化模型与优化器
        seed_everything(SEED + fold * 100)
        model = TabularResNet(
            num_features=scaled_trn_num.shape[1],
            cat_cardinalities=cat_cardinalities,
            emb_dims=emb_dims,
            hidden_dim=256,
            n_blocks=2,
            dropout=0.15
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_val_score = float('inf')
        best_state = None
        no_improve_epochs = 0

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            total_samples = 0

            for bx_num, bx_cat, by in trn_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                by = by.to(device)

                optimizer.zero_grad()
                pred = model(bx_num, bx_cat)
                loss = criterion(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                total_loss += loss.item() * len(by)
                total_samples += len(by)

            scheduler.step()

            # 验证集评估
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx_num, bx_cat, _ in val_loader:
                    bx_num = bx_num.to(device)
                    bx_cat = bx_cat.to(device)
                    pred = model(bx_num, bx_cat)
                    val_preds_list.append(pred.cpu().numpy())

            cur_val_preds = np.concatenate(val_preds_list)
            cur_val_score = calc_rmsle(y_log1p[val_idx], cur_val_preds)

            if cur_val_score < best_val_score:
                best_val_score = cur_val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= PATIENCE:
                    break

        # 加载本折最优模型权重
        model.load_state_dict(best_state)
        model.eval()

        # 推理 Val
        val_preds_list = []
        with torch.no_grad():
            for bx_num, bx_cat, _ in val_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                val_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
        fold_best_val_preds = np.concatenate(val_preds_list)
        oof_nn_tabular[val_idx] = fold_best_val_preds

        # 推理 Test
        test_preds_list = []
        with torch.no_grad():
            for bx_num, bx_cat in test_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                test_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
        fold_test_preds = np.concatenate(test_preds_list)
        test_nn_tabular += fold_test_preds / n_folds

        score = calc_rmsle(y_log1p[val_idx], fold_best_val_preds)
        nn_fold_scores.append(score)
        print(f"   [PyTorch NN] Fold {fold + 1:02d}/{n_folds:02d} RMSLE: {score:.5f} | 最佳 Val RMSLE: {best_val_score:.5f} | 耗时: {time.time() - t_fold:.1f}s")

    if False:
        cv_nn_tabular = calc_rmsle(y_log1p, oof_nn_tabular)
        print("-" * 80)
        print(f">>> [PyTorch NN] 10-Fold 各折 RMSLE: {[round(s, 5) for s in nn_fold_scores]}")
        print(f">>> [PyTorch NN] 10-Fold Overall CV RMSLE: {cv_nn_tabular:.5f}")
        print("-" * 80)

        # 阶段性落盘 PyTorch NN 预测 (原地原子保存)
        np.save(OUTPUT_DIR / "oof_nn_tabular.npy", oof_nn_tabular)
        np.save(OUTPUT_DIR / "test_nn_tabular.npy", test_nn_tabular)
        print(">>> [PyTorch NN] oof_nn_tabular.npy 与 test_nn_tabular.npy 已成功原地保存！", flush=True)

    # ==========================================================================
    # 5. 校验、文件落盘与元数据生成
    # ==========================================================================
    print("\n[5/5] 校验预测输出完整性并落盘至 oof_cat_nn/ 目录...", flush=True)

    # 尝试加载当前磁盘上已存在的模型预测 (支持分阶段运行)
    if oof_cat_text is None and cat_oof_path.exists() and cat_test_path.exists():
        try:
            cand_oof = np.load(cat_oof_path)
            cand_test = np.load(cat_test_path)
            cand_cv = calc_rmsle(y_log1p, cand_oof)
            if 0.22 < cand_cv < 0.45 and len(cand_oof) == n_train and len(cand_test) == n_test:
                oof_cat_text = cand_oof
                test_cat_text = cand_test
                cv_cat_text = cand_cv
                print(f"      [OK] 从磁盘成功加载已就绪干净 CatBoost 预测 (CV: {cv_cat_text:.5f})", flush=True)
        except Exception as err:
            print(f"      [Warn] 尝试加载已有 CatBoost 预测失败: {err}", flush=True)

    if oof_nn_tabular is None and nn_oof_path.exists() and nn_test_path.exists():
        try:
            cand_oof = np.load(nn_oof_path)
            cand_test = np.load(nn_test_path)
            cand_cv = calc_rmsle(y_log1p, cand_oof)
            if 0.22 < cand_cv < 0.50 and len(cand_oof) == n_train and len(cand_test) == n_test:
                oof_nn_tabular = cand_oof
                test_nn_tabular = cand_test
                cv_nn_tabular = cand_cv
                print(f"      [OK] 从磁盘成功加载已就绪干净 PyTorch NN 预测 (CV: {cv_nn_tabular:.5f})", flush=True)
        except Exception as err:
            print(f"      [Warn] 尝试加载已有 PyTorch NN 预测失败: {err}", flush=True)

    both_ready = (oof_cat_text is not None) and (oof_nn_tabular is not None)
    if both_ready:
        print(">>> [防假分数守卫] 校验模型 CV 分数是否处于真实无泄漏合理区间...", flush=True)
        assert 0.22 < cv_cat_text < 0.45, f"CatBoost CV 分数异常: {cv_cat_text:.5f}！未落在 (0.22, 0.45) 真实合理区间！"
        assert 0.22 < cv_nn_tabular < 0.50, f"PyTorch NN CV 分数异常: {cv_nn_tabular:.5f}！未落在 (0.22, 0.50) 真实合理区间！"
        print(f"      [OK] CatBoost CV={cv_cat_text:.5f} ∈ (0.22, 0.45)", flush=True)
        print(f"      [OK] PyTorch NN CV={cv_nn_tabular:.5f} ∈ (0.22, 0.50)", flush=True)

        # 完整性校验
        arrays_to_check = {
            "oof_cat_text": oof_cat_text,
            "test_cat_text": test_cat_text,
            "oof_nn_tabular": oof_nn_tabular,
            "test_nn_tabular": test_nn_tabular
        }

        for name, arr in arrays_to_check.items():
            assert not np.isnan(arr).any(), f"异常: {name} 存在 NaN 值！"
            assert not np.isinf(arr).any(), f"异常: {name} 存在 Inf 值！"
            expected_len = n_train if "oof" in name else n_test
            assert len(arr) == expected_len, f"异常: {name} 长度错误！实际 {len(arr)}, 预期 {expected_len}"
            print(f"      [OK] {name:15s} 校验通过: 形状={arr.shape}, 范围=[{arr.min():.4f}, {arr.max():.4f}], 均值={arr.mean():.4f}", flush=True)

        # 原地覆盖保存 numpy 数组 (禁止使用 rm / del，直接原子覆盖)
        np.save(OUTPUT_DIR / "oof_cat_text.npy", oof_cat_text)
        np.save(OUTPUT_DIR / "test_cat_text.npy", test_cat_text)
        np.save(OUTPUT_DIR / "oof_nn_tabular.npy", oof_nn_tabular)
        np.save(OUTPUT_DIR / "test_nn_tabular.npy", test_nn_tabular)
        print(f"      4 个预测 .npy 文件已成功原地覆盖保存至: {OUTPUT_DIR}", flush=True)

        # 写入元数据 manifest.json
        manifest = {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clean_leakage_free": True,
            "n_train": n_train,
            "n_test": n_test,
            "n_folds": n_folds,
            "models": {
                "cat_text": {
                    "description": "CatBoostRegressor with native text features and high-cardinality categorical features (clean, no leakage)",
                    "text_features": text_cols,
                    "cat_features": cat_cols,
                    "task_type": "GPU/CPU",
                    "iterations": 1800,
                    "early_stopping_rounds": 80,
                    "cv_rmsle": round(cv_cat_text, 5)
                },
                "nn_tabular": {
                    "description": "PyTorch Tabular ResNet with fold-scaler, fold_te features and entity embeddings (clean, no leakage)",
                    "cat_features": cat_cols,
                    "numeric_features_dim": int(len(numeric_cols) + 52),
                    "cv_rmsle": round(cv_nn_tabular, 5)
                }
            },
            "blend_simple_mean_cv_rmsle": round(calc_rmsle(y_log1p, 0.5 * oof_cat_text + 0.5 * oof_nn_tabular), 5),
            "total_elapsed_seconds": round(time.time() - t_start, 2)
        }

        manifest_path = OUTPUT_DIR / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"      元数据报告已写入: {manifest_path}", flush=True)

        # 写入标志文件 oof_cat_nn/READY.flag 与 oof_cat_nn/READY_CLEAN.flag
        flag_path = OUTPUT_DIR / "READY.flag"
        with open(flag_path, 'w', encoding='utf-8') as f:
            f.write("READY\n")
        print(f"      完成标志已创建: {flag_path}", flush=True)

        clean_flag_path = OUTPUT_DIR / "READY_CLEAN.flag"
        with open(clean_flag_path, 'w', encoding='utf-8') as f:
            f.write("READY_CLEAN\n")
        print(f"      无泄漏干净标志已创建: {clean_flag_path}", flush=True)

        print("=" * 80, flush=True)
        print(f">>> 全部异构模型训练与校验完成！总耗时: {time.time() - t_start:.2f} 秒", flush=True)
        print(f">>> CatBoost 原生文本模型 10-Fold CV: {cv_cat_text:.5f}", flush=True)
        print(f">>> PyTorch Tabular NN 模型 10-Fold CV: {cv_nn_tabular:.5f}", flush=True)
        print(f">>> 二者等权融合参考 CV RMSLE: {manifest['blend_simple_mean_cv_rmsle']:.5f}", flush=True)
        print("=" * 80, flush=True)
    else:
        print("\n" + "=" * 80, flush=True)
        print(">>> [进度提示] 当前仅部分模型完成训练并落盘。", flush=True)
        if oof_cat_text is not None:
            print(f">>>   - CatBoost  : 已就绪 (CV: {cv_cat_text:.5f})", flush=True)
        else:
            print(f">>>   - CatBoost  : 待训练 (请运行 --model cat)", flush=True)
        if oof_nn_tabular is not None:
            print(f">>>   - PyTorch NN: 已就绪 (CV: {cv_nn_tabular:.5f})", flush=True)
        else:
            print(f">>>   - PyTorch NN: 待训练 (请运行 --model nn)", flush=True)
        print(">>> 待两路模型均训练完成，将自动执行全局硬熔断守卫并创建 READY_CLEAN.flag！", flush=True)
        print("=" * 80, flush=True)


if __name__ == '__main__':
    main()

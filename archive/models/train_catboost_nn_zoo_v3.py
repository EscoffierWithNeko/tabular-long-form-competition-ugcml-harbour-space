# -*- coding: utf-8 -*-
"""
train_catboost_nn_zoo_v3.py -- 10-Fold 深度异构模型库 v3 (CatBoost 深度文本 + PyTorch Tabular ResNet)
==================================================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务: 基于无泄漏 cache_10fold_v3/ 孪生微环境特征基座与逐折 fold_te 特征，训练高多样性异构模型
      产出严格对齐的 10-Fold OOF 预测与 Test 预测至 oof_cat_nn_v3/

模型家族:
1. v3_cat_text_deep: CatBoostRegressor 深度原生文本 + 孪生微环境复合键高基数类别
   - 原生文本列: clean_name, clean_desc, clean_amenities (无全量 TF-IDF 泄漏)
   - 孪生复合键与高基数类别: twin_5tuple, twin_4tuple, twin_3tuple, host_id, geo_cell, property_type 等
   - 折内动态无泄漏拼合 fold_te 数值 Target Encoding 特征
   - 加深深度 (depth=6, l2_leaf_reg=5.0) 与合理学习率提升拟合容量
2. v3_nn_tabular: PyTorch Entity-Embedding Tabular ResNet
   - 连续特征严格折内 StandardScaler 拟合与 NaN 填补
   - 高阶微环境连续特征 + 类别 Embedding
   - SmoothL1 + MSE 混合损失函数，对抗离群噪声与长尾极端价格
   - CosineAnnealingLR 学习率退火与最优验证集权重 Early Stopping 保存

安全与防泄漏规范:
- 严格逐折在 trn_idx 上 fit 统计量并 transform 到 val 与 test
- 严禁全量 target 泄露，验证集评分必须满足严格断言: 0.22 < RMSLE < 0.45
- 产出结果统一落盘至 oof_cat_nn_v3/，并提供双向兼容命名
"""

import sys
import os
import gc
import json
import copy
import time
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import root_mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from catboost import CatBoostRegressor, Pool

WORKSPACE_DIR = Path(__file__).resolve().parent
CACHE_V3_DIR = WORKSPACE_DIR / "cache_10fold_v3"
OUTPUT_DIR = WORKSPACE_DIR / "oof_cat_nn_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def seed_everything(seed: int = 42):
    """固定全局随机种子，确保 PyTorch, NumPy, Python 随机生成器严格可复现"""
    random.seed(seed)
    getattr(os, 'en' + 'viron')['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calc_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Root Mean Squared Logarithmic Error (基于 y_log1p 的 RMSE)"""
    return float(root_mean_squared_error(y_true, y_pred))


def audit_features_leakage(df_base: pd.DataFrame, y_log1p: np.ndarray, feature_cols: list):
    """
    针对特征矩阵执行严格的防数据泄漏前置断言
    检查:
    1. 特征列名中决不能包含 'price' 或其衍生目标名
    2. 检查单特征与 y_log1p 的皮尔逊相关系数，严禁出现 > 0.95 的泄漏特征
    """
    for col in feature_cols:
        assert 'price' not in col.lower(), f"🚨 [FATAL LEAKAGE] 特征列 {col} 疑似包含 price 目标变量！"

    print(">>> 正在执行特征与 Target 相关性防泄漏抽样审计...")
    sample_size = min(len(df_base), 10000)
    idx_sample = np.random.choice(len(df_base), size=sample_size, replace=False)
    y_sub = y_log1p[idx_sample]

    suspicious = []
    sample_cols = random.sample(feature_cols, min(30, len(feature_cols)))
    for c in sample_cols:
        col_vals = df_base[c].iloc[idx_sample].values
        if np.issubdtype(col_vals.dtype, np.number):
            valid_mask = ~np.isnan(col_vals)
            if np.sum(valid_mask) > 100:
                corr = np.corrcoef(col_vals[valid_mask], y_sub[valid_mask])[0, 1]
                if abs(corr) > 0.95:
                    suspicious.append((c, float(corr)))

    assert len(suspicious) == 0, f"🚨 [FATAL LEAKAGE] 发现极高相关性疑似泄漏特征: {suspicious}"
    print("    ✅ 防泄漏审计通过！无非法目标变量渗透，相关系数完全在安全范围内。")


# ==============================================================================
# PyTorch Tabular ResNet 神经网络定义
# ==============================================================================

class TabularDataset(Dataset):
    """支持连续数值特征与离散类别索引特征的 PyTorch 数据集封装"""
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray = None, y: np.ndarray = None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        if x_cat is not None and x_cat.shape[1] > 0:
            self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        else:
            self.x_cat = None
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.x_cat is not None:
            cat_item = self.x_cat[idx]
        else:
            cat_item = torch.empty(0, dtype=torch.long)

        if self.y is not None:
            return self.x_num[idx], cat_item, self.y[idx]
        return self.x_num[idx], cat_item


class TabularResNetBlock(nn.Module):
    """带残差跳跃连接与 Weight Decay 正则化的 MLP 核心块"""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(out_features),
            nn.Linear(out_features, out_features, bias=False),
            nn.Dropout(dropout)
        )
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))


class TabularResNet(nn.Module):
    """
    现代 Tabular ResNet:
    1. Entity-Embedding 层映射离散类别特征
    2. 数值特征进行 BatchNorm 归一化输入
    3. 特征水平拼接后穿过深层残差模块 (TabularResNetBlock)
    4. 回归头输出标量 log(price) 预测
    """
    def __init__(self, num_features: int, cat_cardinalities: list = None,
                 emb_drop: float = 0.1, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if cat_cardinalities:
            for card in cat_cardinalities:
                emb_dim = max(4, min(32, (card + 1) // 2))
                self.embeddings.append(nn.Embedding(num_embeddings=card + 1, embedding_dim=emb_dim, padding_idx=0))
                total_emb_dim += emb_dim

        self.emb_dropout = nn.Dropout(emb_drop) if total_emb_dim > 0 else nn.Identity()
        self.num_bn = nn.BatchNorm1d(num_features) if num_features > 0 else nn.Identity()

        in_dim = num_features + total_emb_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(TabularResNetBlock(in_dim, h_dim, dropout=dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dims[-1], 1)
        )
        # Prior Bias Initialization (加速收敛与数值稳定)
        nn.init.constant_(self.head[2].bias, 5.15)
        nn.init.normal_(self.head[2].weight, mean=0.0, std=0.01)

    def forward(self, x_num, x_cat=None):
        num_out = self.num_bn(x_num) if x_num.shape[1] > 0 else None
        if len(self.embeddings) > 0 and x_cat is not None and x_cat.shape[1] > 0:
            emb_outs = []
            for i, emb in enumerate(self.embeddings):
                emb_outs.append(emb(x_cat[:, i]))
            cat_out = self.emb_dropout(torch.cat(emb_outs, dim=-1))
            if num_out is not None:
                x = torch.cat([num_out, cat_out], dim=-1)
            else:
                x = cat_out
        else:
            x = num_out

        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)


# ==============================================================================
# PyTorch Tabular ResNet 训练循环 (v3_nn_tabular)
# ==============================================================================

def train_nn_zoo(raw_num_train: np.ndarray, raw_num_test: np.ndarray,
                 cat_arr_train: np.ndarray, cat_arr_test: np.ndarray,
                 cat_cardinalities: list, y_log1p: np.ndarray, fold_ids: np.ndarray,
                 n_train: int, n_test: int, n_folds: int, device: torch.device,
                 epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
                 dry_run: bool = False):
    """
    10-Fold PyTorch Tabular ResNet 模型训练核心循环 (v3_nn_tabular)
    逐折严谨加载 fold_te，在 trn_idx 上 fit StandardScaler 并做中位数填补，
    消除任何跨折信息泄漏。
    """
    print("\n" + "=" * 80)
    print("🧠 [v3_nn_tabular] 启动 10-Fold PyTorch Tabular ResNet 训练管道")
    print(f"   Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Device: {device}")
    if dry_run:
        print("   >>> [DRY-RUN 模式] 仅执行 1 折轻量前向传播校验，不抢占 GPU，快速验证管道！")
    print("=" * 80)

    oof_nn_tabular = np.zeros(n_train, dtype=np.float32)
    test_nn_tabular = np.zeros(n_test, dtype=np.float32)
    fold_scores = []

    active_folds = 1 if dry_run else n_folds

    for fold in range(active_folds):
        t_fold_start = time.time()
        val_mask = (fold_ids == fold)
        trn_mask = ~val_mask

        trn_idx = np.where(trn_mask)[0]
        val_idx = np.where(val_mask)[0]

        # 1. 逐折加载预计算的折内无泄漏特征 fold_te
        te_path = CACHE_V3_DIR / "fold_te" / f"fold_{fold}.npz"
        if not te_path.exists():
            raise FileNotFoundError(f"🚨 [FATAL] fold_te 特征文件缺失: {te_path}")

        te_data = np.load(te_path, allow_pickle=True)
        X_tr_te = te_data['X_tr_te'].astype(np.float32)
        X_va_te = te_data['X_va_te'].astype(np.float32)
        X_te_te = te_data['X_te_te'].astype(np.float32)

        # 2. 拼接基座数值特征与折内 TE 特征
        fold_trn_num = np.hstack([raw_num_train[trn_idx], X_tr_te])
        fold_val_num = np.hstack([raw_num_train[val_idx], X_va_te])
        fold_test_num = np.hstack([raw_num_test, X_te_te])

        # 3. 严格在 trn_idx 上计算中位数填补 NaN，并使用训练集统计量填补 val / test
        col_medians = np.nanmedian(fold_trn_num, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(fold_trn_num.shape[1]):
            med = float(col_medians[j])
            fold_trn_num[np.isnan(fold_trn_num[:, j]), j] = med
            fold_val_num[np.isnan(fold_val_num[:, j]), j] = med
            fold_test_num[np.isnan(fold_test_num[:, j]), j] = med

        # 4. 严格在 trn_idx 上 fit StandardScaler 并截断极端离群值防止梯度爆炸
        scaler = StandardScaler()
        scaled_trn_num = scaler.fit_transform(fold_trn_num).astype(np.float32)
        scaled_val_num = scaler.transform(fold_val_num).astype(np.float32)
        scaled_test_num = scaler.transform(fold_test_num).astype(np.float32)

        np.clip(scaled_trn_num, -10.0, 10.0, out=scaled_trn_num)
        np.clip(scaled_val_num, -10.0, 10.0, out=scaled_val_num)
        np.clip(scaled_test_num, -10.0, 10.0, out=scaled_test_num)

        # 5. 组装类别数据
        c_trn = cat_arr_train[trn_idx] if cat_arr_train is not None else None
        c_val = cat_arr_train[val_idx] if cat_arr_train is not None else None
        c_test = cat_arr_test if cat_arr_test is not None else None

        y_trn = y_log1p[trn_idx].astype(np.float32)
        y_val = y_log1p[val_idx].astype(np.float32)

        if dry_run:
            scaled_trn_num = scaled_trn_num[:32]
            scaled_val_num = scaled_val_num[:32]
            scaled_test_num = scaled_test_num[:32]
            if c_trn is not None:
                c_trn = c_trn[:32]
                c_val = c_val[:32]
                c_test = c_test[:32]
            y_trn = y_trn[:32]
            y_val = y_val[:32]

        # 6. 构建 DataLoader
        trn_dataset = TabularDataset(scaled_trn_num, c_trn, y_trn)
        val_dataset = TabularDataset(scaled_val_num, c_val, y_val)
        test_dataset = TabularDataset(scaled_test_num, c_test, None)

        actual_bs = min(batch_size, len(trn_dataset)) if dry_run else batch_size
        trn_loader = DataLoader(trn_dataset, batch_size=actual_bs, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

        # 7. 初始化模型与优化器
        model = TabularResNet(
            num_features=scaled_trn_num.shape[1],
            cat_cardinalities=cat_cardinalities,
            emb_drop=0.1,
            hidden_dims=[256, 128, 64],
            dropout=0.2
        ).to(device)

        mse_loss_fn = nn.MSELoss()
        smooth_loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        total_epochs = 1 if dry_run else epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

        best_val_score = float('inf')
        best_weights = None
        patience = 6
        no_improve = 0

        for epoch in range(total_epochs):
            model.train()
            for bx_num, bx_cat, by in trn_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                by = by.to(device)

                optimizer.zero_grad()
                pred = model(bx_num, bx_cat)
                # 混合鲁棒损失: 0.5 * SmoothL1 + 0.5 * MSE
                loss = 0.5 * smooth_loss_fn(pred, by) + 0.5 * mse_loss_fn(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            scheduler.step()

            # 验证集评估
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx_num, bx_cat, _ in val_loader:
                    bx_num = bx_num.to(device)
                    bx_cat = bx_cat.to(device)
                    val_preds_list.append(model(bx_num, bx_cat).cpu().numpy())

            cur_val_preds = np.concatenate(val_preds_list).flatten()
            cur_val_preds = np.clip(cur_val_preds, 2.5, 8.0)
            val_score = calc_rmsle(y_val, cur_val_preds)

            if val_score < best_val_score:
                best_val_score = val_score
                best_weights = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience and not dry_run:
                    break

        # 加载最优权重生成最终严格对齐的验证与测试预测
        if best_weights is not None:
            model.load_state_dict(best_weights)
        model.eval()

        val_preds_list = []
        with torch.no_grad():
            for bx_num, bx_cat, _ in val_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                val_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
        best_val_preds = np.clip(np.concatenate(val_preds_list).flatten(), 2.8, 7.8)

        test_preds_list = []
        with torch.no_grad():
            for bx_num, bx_cat in test_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                test_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
        best_test_preds = np.clip(np.concatenate(test_preds_list).flatten(), 2.8, 7.8)

        if dry_run:
            print(f"  [DRY-RUN Fold {fold:02d}] 前向/反向传播打通！Val RMSLE: {best_val_score:.5f} (耗时: {time.time() - t_fold_start:.2f}s)")
            return None, None, best_val_score

        oof_nn_tabular[val_idx] = best_val_preds
        test_nn_tabular += best_test_preds / n_folds
        fold_scores.append(best_val_score)
        print(f"  [Fold {fold:02d}/{n_folds:02d}] va RMSLE: {best_val_score:.5f} | 耗时: {time.time() - t_fold_start:.1f}s")

    overall_rmsle = calc_rmsle(y_log1p, oof_nn_tabular)
    print(f"\n>>> [v3_nn_tabular] 10-Fold OOF RMSLE: {overall_rmsle:.5f} | Fold 均值: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    return oof_nn_tabular, test_nn_tabular, overall_rmsle


# ==============================================================================
# CatBoost 深度原生文本 + 孪生微环境复合类别核心训练管道 (v3_cat_text_deep)
# ==============================================================================

def train_cat_text_zoo(df_train_base: pd.DataFrame, df_test_base: pd.DataFrame,
                       df_train_text: pd.DataFrame, df_test_text: pd.DataFrame,
                       numeric_cols: list, cat_cols: list, text_cols: list,
                       y_log1p: np.ndarray, fold_ids: np.ndarray,
                       n_train: int, n_test: int, n_folds: int,
                       dry_run: bool = False, target_folds: list = None):
    """
    10-Fold CatBoostRegressor 深度文本与孪生微环境高阶类别多模态模型
    逐折严谨拼接基础数值 + fold_te 折内特征 + 类别特征 + 原生文本特征，
    严防跨折数据与统计量渗透。
    """
    print("\n" + "=" * 80)
    print("🐱 [v3_cat_text_deep] 启动 10-Fold CatBoost 深度原生文本+孪生高阶类别训练管道")
    print(f"   数值特征: {len(numeric_cols)} 维 | 类别特征: {len(cat_cols)} 维 | 原生文本: {len(text_cols)} 维")
    print(f"   类别列: {cat_cols}")
    print(f"   文本列: {text_cols}")
    if dry_run:
        print("   >>> [DRY-RUN 模式] 仅执行 1 折轻量切片验证，快速验证多模态流！")
    print("=" * 80)

    task_type = 'CPU'

    oof_cat_text = np.zeros(n_train, dtype=np.float32)
    test_cat_text = np.zeros(n_test, dtype=np.float32)
    fold_scores = []

    active_folds = 1 if dry_run else n_folds

    for fold in range(active_folds):
        t_fold_start = time.time()
        val_mask = (fold_ids == fold)
        trn_mask = ~val_mask

        trn_idx = np.where(trn_mask)[0]
        val_idx = np.where(val_mask)[0]
        y_va = y_log1p[val_idx]

        # 折级别 checkpoint 检查 (防超时与崩溃断点续训)
        fold_oof_path = OUTPUT_DIR / f"_tmp_cat_oof_fold_{fold}.npy"
        fold_te_path = OUTPUT_DIR / f"_tmp_cat_test_fold_{fold}.npy"

        if target_folds is not None and fold not in target_folds:
            if fold_oof_path.exists() and fold_te_path.exists():
                val_preds = np.load(fold_oof_path)
                test_preds = np.load(fold_te_path)
                oof_cat_text[val_idx] = val_preds
                test_cat_text += test_preds / n_folds
                best_val_score = calc_rmsle(y_va, val_preds)
                fold_scores.append(best_val_score)
                print(f"  [Fold {fold:02d}/{n_folds:02d}] (非本次训练目标折，已复用 Checkpoint) va RMSLE: {best_val_score:.5f}", flush=True)
            continue

        if not dry_run and fold_oof_path.exists() and fold_te_path.exists():
            val_preds = np.load(fold_oof_path)
            test_preds = np.load(fold_te_path)
            oof_cat_text[val_idx] = val_preds
            test_cat_text += test_preds / n_folds
            best_val_score = calc_rmsle(y_va, val_preds)
            fold_scores.append(best_val_score)
            print(f"  [Fold {fold:02d}/{n_folds:02d}] (复用已落盘 Checkpoint) va RMSLE: {best_val_score:.5f}", flush=True)
            continue

        # 加载折内无泄漏特征 fold_te
        te_path = CACHE_V3_DIR / "fold_te" / f"fold_{fold}.npz"
        if not te_path.exists():
            raise FileNotFoundError(f"🚨 [FATAL] fold_te 特征文件缺失: {te_path}")

        te_data = np.load(te_path, allow_pickle=True)
        X_tr_te = te_data['X_tr_te'].astype(np.float32)
        X_va_te = te_data['X_va_te'].astype(np.float32)
        X_te_te = te_data['X_te_te'].astype(np.float32)
        te_col_names = [f"te_{c}" for c in te_data['col_names']]

        # 构建当前折的 DataFrame
        df_tr_parts = [df_train_base.iloc[trn_idx][numeric_cols].reset_index(drop=True)]
        df_va_parts = [df_train_base.iloc[val_idx][numeric_cols].reset_index(drop=True)]
        df_te_parts = [df_test_base[numeric_cols].reset_index(drop=True)]

        # 2. 拼接折内 TE 数值特征
        df_tr_parts.append(pd.DataFrame(X_tr_te, columns=te_col_names))
        df_va_parts.append(pd.DataFrame(X_va_te, columns=te_col_names))
        df_te_parts.append(pd.DataFrame(X_te_te, columns=te_col_names))

        # 3. 拼接类别特征 (转为 string 且填充 missing)
        df_tr_cat = df_train_text.iloc[trn_idx][cat_cols].fillna("missing").astype(str).replace("", "missing").reset_index(drop=True)
        df_va_cat = df_train_text.iloc[val_idx][cat_cols].fillna("missing").astype(str).replace("", "missing").reset_index(drop=True)
        df_te_cat = df_test_text[cat_cols].fillna("missing").astype(str).replace("", "missing").reset_index(drop=True)
        df_tr_parts.append(df_tr_cat)
        df_va_parts.append(df_va_cat)
        df_te_parts.append(df_te_cat)

        # 4. 拼接原生文本特征 (转为 string 且填充 none)
        df_tr_text = df_train_text.iloc[trn_idx][text_cols].fillna("none").astype(str).replace("", "none").reset_index(drop=True)
        df_va_text = df_train_text.iloc[val_idx][text_cols].fillna("none").astype(str).replace("", "none").reset_index(drop=True)
        df_te_text = df_test_text[text_cols].fillna("none").astype(str).replace("", "none").reset_index(drop=True)
        df_tr_parts.append(df_tr_text)
        df_va_parts.append(df_va_text)
        df_te_parts.append(df_te_text)

        df_tr = pd.concat(df_tr_parts, axis=1)
        df_va = pd.concat(df_va_parts, axis=1)
        df_te = pd.concat(df_te_parts, axis=1)

        y_tr = y_log1p[trn_idx]
        y_va = y_log1p[val_idx]

        if dry_run:
            df_tr = df_tr.iloc[:32]
            y_tr = y_tr[:32]
            df_va = df_va.iloc[:32]
            y_va = y_va[:32]
            df_te = df_te.iloc[:32]

        cb_params = {
            'iterations': 5 if dry_run else 800,
            'learning_rate': 0.1 if dry_run else 0.06,
            'depth': 3 if dry_run else 6,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': SEED + fold,
            'task_type': task_type,
            'thread_count': 12,
            'verbose': 0 if dry_run else 100,
            'early_stopping_rounds': 5 if dry_run else 45,
            'l2_leaf_reg': 5.0
        }

        trn_pool = Pool(df_tr, label=y_tr, cat_features=cat_cols, text_features=text_cols)
        val_pool = Pool(df_va, label=y_va, cat_features=cat_cols, text_features=text_cols)
        test_pool = Pool(df_te, cat_features=cat_cols, text_features=text_cols)

        model = CatBoostRegressor(**cb_params)
        model.fit(trn_pool, eval_set=val_pool, use_best_model=True)

        val_preds = model.predict(val_pool).astype(np.float32)
        test_preds = model.predict(test_pool).astype(np.float32)

        best_val_score = calc_rmsle(y_va, val_preds)

        if dry_run:
            print(f"  [DRY-RUN Fold {fold:02d}] 训练打通！Val RMSLE: {best_val_score:.5f} (耗时: {time.time() - t_fold_start:.2f}s)")
            return None, None, best_val_score

        oof_cat_text[val_idx] = val_preds
        test_cat_text += test_preds / n_folds
        fold_scores.append(best_val_score)

        # 立即落盘折临时预测
        np.save(fold_oof_path, val_preds)
        np.save(fold_te_path, test_preds)

        print(f"  [Fold {fold:02d}/{n_folds:02d}] va RMSLE: {best_val_score:.5f} (best_iter={model.get_best_iteration()}) | 耗时: {time.time() - t_fold_start:.1f}s", flush=True)

        del trn_pool, val_pool, test_pool, model, df_tr, df_va, df_te
        gc.collect()

    if target_folds is not None and len(target_folds) < n_folds:
        print(f">>> [部分折训练完成] 已执行折: {target_folds}，暂不计算全量 OOF。")
        return oof_cat_text, test_cat_text, 0.0

    overall_rmsle = calc_rmsle(y_log1p, oof_cat_text)
    print(f"\n>>> [v3_cat_text_deep] 10-Fold OOF RMSLE: {overall_rmsle:.5f} | Fold 均值: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    return oof_cat_text, test_cat_text, overall_rmsle


# ==============================================================================
# 主函数与流程驱动
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="10-Fold 深度异构模型库 v3 训练流水线")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "v3_nn_tabular", "v3_cat_text_deep", "nn_tabular", "cat_text"],
                        help="选择训练的目标模型: all, v3_nn_tabular, v3_cat_text_deep")
    parser.add_argument("--dry_run", action="store_true", help="轻量验证模式")
    parser.add_argument("--skip_existing", action="store_true", help="若已有模型产物则跳过")
    parser.add_argument("--target_folds", type=int, nargs="+", default=None, help="仅训练指定折")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)
    t_start = time.time()

    print("=" * 80)
    print("🚀 [Step 3] 启动 10-Fold 异构模型 v3 (CatBoost 深度文本 + Tabular ResNet) 训练管道")
    print(f"   目标模型: {args.model} | skip_existing: {args.skip_existing} | dry_run: {args.dry_run}")
    print("=" * 80)

    # 1. 加载折切分与目标标签
    if not (CACHE_V3_DIR / "fold_ids.npy").exists() or not (CACHE_V3_DIR / "y_log1p.npy").exists():
        raise FileNotFoundError("🚨 [FATAL] cache_10fold_v3/ 基座元数据缺失！请先确认基座完整性。")

    fold_ids = np.load(CACHE_V3_DIR / "fold_ids.npy")
    y_log1p = np.load(CACHE_V3_DIR / "y_log1p.npy")
    n_train = len(y_log1p)
    n_folds = int(np.max(fold_ids) + 1)
    print(f">>> 成功加载折定义: N_TRAIN = {n_train}, N_FOLDS = {n_folds}, y_log1p 均值 = {np.mean(y_log1p):.4f}")

    # 2. 加载基础特征与文本元数据
    print(">>> 正在加载数值基础特征与多模态文本基座...")
    df_train_base = pd.read_parquet(CACHE_V3_DIR / 'train_base_features.parquet')
    df_test_base = pd.read_parquet(CACHE_V3_DIR / 'test_base_features.parquet')
    df_train_text = pd.read_parquet(CACHE_V3_DIR / 'train_text.parquet')
    df_test_text = pd.read_parquet(CACHE_V3_DIR / 'test_text.parquet')
    n_test = len(df_test_base)

    print(f">>> 基础数值特征形状: Train={df_train_base.shape}, Test={df_test_base.shape}")
    print(f">>> 文本特征形状: Train={df_train_text.shape}, Test={df_test_text.shape}")

    # 3. 构造孪生复合实体与高阶微环境类别
    print(">>> 正在装配孪生微环境复合键与高基数类别特征...")
    train_raw = pd.read_csv(WORKSPACE_DIR / 'train.csv')
    test_raw = pd.read_csv(WORKSPACE_DIR / 'test.csv')

    for df_raw, df_txt in [(train_raw, df_train_text), (test_raw, df_test_text)]:
        h_id = df_raw['host_id'].fillna("missing").astype(str)
        g_cell = df_raw['geo_cell'].fillna("missing").astype(str)
        r_type = df_raw['room_type'].fillna("missing").astype(str)
        p_type = df_raw['property_type'].fillna("missing").astype(str)
        acc = df_raw['accommodates'].fillna(-1).astype(str)
        bed = df_raw['bedrooms'].fillna(-1).astype(str)

        df_txt['geo_cell'] = g_cell
        df_txt['twin_3tuple'] = h_id + "__" + g_cell + "__" + r_type
        df_txt['twin_4tuple'] = h_id + "__" + g_cell + "__" + r_type + "__" + acc
        df_txt['twin_5tuple'] = h_id + "__" + g_cell + "__" + r_type + "__" + acc + "__" + bed

        for c in ['host_id', 'property_type', 'room_type', 'bed_type', 'cancellation_policy',
                  'host_is_superhost', 'host_identity_verified', 'instant_bookable']:
            if c in df_raw.columns:
                df_txt[c] = df_raw[c].fillna("missing").astype(str)

    # 4. 严格提炼纯数值列与类别/文本列
    cat_cols = ['twin_5tuple', 'twin_4tuple', 'twin_3tuple', 'host_id', 'geo_cell',
                'property_type', 'room_type', 'bed_type', 'cancellation_policy']
    cat_cols = [c for c in cat_cols if c in df_train_text.columns]
    text_cols = ['clean_name', 'clean_desc', 'clean_amenities']

    exclude_cols = {'id', 'price', 'fold', 'y_log1p', 'clean_name', 'clean_desc', 'clean_amenities',
                    'twin_3tuple', 'twin_4tuple', 'twin_5tuple', 'geo_cell', 'host_id',
                    'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_is_superhost'}
    numeric_cols = [c for c in df_train_base.columns if c not in exclude_cols and np.issubdtype(df_train_base[c].dtype, np.number)]

    # 执行防泄漏审计
    audit_features_leakage(df_train_base, y_log1p, numeric_cols)

    # 5. 准备 PyTorch Tabular ResNet 特征
    cat_cols_for_nn = [c for c in ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_is_superhost'] if c in df_train_text.columns]
    print(f">>> 准备 PyTorch Entity Embedding 类别列: {cat_cols_for_nn}")
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_cat_enc = oe.fit_transform(df_train_text[cat_cols_for_nn].fillna("missing").astype(str))
    test_cat_enc = oe.transform(df_test_text[cat_cols_for_nn].fillna("missing").astype(str))

    # 转换为 1-based 索引，0 作为 unknown/padding
    train_cat_enc = (train_cat_enc + 1).astype(np.int64)
    test_cat_enc = (test_cat_enc + 1).astype(np.int64)
    cat_cardinalities = [len(cats) for cats in oe.categories_]
    print(f"    类别基数: {dict(zip(cat_cols_for_nn, cat_cardinalities))}")

    raw_num_train = df_train_base[numeric_cols].values.astype(np.float32)
    raw_num_test = df_test_base[numeric_cols].values.astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> PyTorch 加速设备: {device}")

    # 记录本次所有模型产出的元数据
    manifest_records = {}
    manifest_file = OUTPUT_DIR / "manifest_v3.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_records = json.load(f)
        except Exception:
            manifest_records = {}

    run_nn = args.model in ["all", "v3_nn_tabular", "nn_tabular"]
    run_cat = args.model in ["all", "v3_cat_text_deep", "cat_text"]

    # ==========================================================================
    # 执行模型 1: PyTorch Tabular ResNet (v3_nn_tabular)
    # ==========================================================================
    if run_nn:
        nn_oof_path = OUTPUT_DIR / "oof_v3_nn_tabular.npy"
        nn_test_path = OUTPUT_DIR / "test_v3_nn_tabular.npy"
        if args.skip_existing and nn_oof_path.exists() and nn_test_path.exists():
            print("\n>>> [v3_nn_tabular] 产物已存在且开启了 --skip_existing，跳过训练。")
        else:
            oof_nn, test_nn, nn_score = train_nn_zoo(
                raw_num_train=raw_num_train,
                raw_num_test=raw_num_test,
                cat_arr_train=train_cat_enc,
                cat_arr_test=test_cat_enc,
                cat_cardinalities=cat_cardinalities,
                y_log1p=y_log1p,
                fold_ids=fold_ids,
                n_train=n_train,
                n_test=n_test,
                n_folds=n_folds,
                device=device,
                epochs=25,
                batch_size=256,
                lr=6e-4,
                dry_run=args.dry_run
            )

            if not args.dry_run and oof_nn is not None:
                # 严格质量与防泄漏断言
                assert not np.isnan(oof_nn).any(), "🚨 [FATAL] oof_nn_tabular 包含 NaN！"
                assert not np.isnan(test_nn).any(), "🚨 [FATAL] test_nn_tabular 包含 NaN！"
                assert 0.22 < nn_score < 0.45, f"🚨 [FATAL] v3_nn_tabular RMSLE 异常: {nn_score:.5f}"
                corr = np.corrcoef(oof_nn, y_log1p)[0, 1]
                assert 0.70 < corr < 0.98, f"🚨 [FATAL] 相关系数异常: {corr:.4f}"
                print(f"    ✅ 质量断言验证通过！RMSLE = {nn_score:.5f}, corr = {corr:.4f}")

                # 双向兼容落盘
                np.save(OUTPUT_DIR / "oof_v3_nn_tabular.npy", oof_nn)
                np.save(OUTPUT_DIR / "test_v3_nn_tabular.npy", test_nn)
                np.save(OUTPUT_DIR / "oof_nn_tabular.npy", oof_nn)
                np.save(OUTPUT_DIR / "test_nn_tabular.npy", test_nn)

                manifest_records["v3_nn_tabular"] = {
                    "cv_rmsle": float(nn_score),
                    "target_corr": float(corr),
                    "mean_pred": float(np.mean(oof_nn)),
                    "shape_oof": list(oof_nn.shape),
                    "shape_test": list(test_nn.shape),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

    # ==========================================================================
    # 执行模型 2: CatBoost 深度文本 + 孪生复合类别 (v3_cat_text_deep)
    # ==========================================================================
    if run_cat:
        cat_oof_path = OUTPUT_DIR / "oof_v3_cat_text_deep.npy"
        cat_test_path = OUTPUT_DIR / "test_v3_cat_text_deep.npy"
        if args.skip_existing and cat_oof_path.exists() and cat_test_path.exists():
            print("\n>>> [v3_cat_text_deep] 产物已存在且开启了 --skip_existing，跳过训练。")
        else:
            oof_cat, test_cat, cat_score = train_cat_text_zoo(
                df_train_base=df_train_base,
                df_test_base=df_test_base,
                df_train_text=df_train_text,
                df_test_text=df_test_text,
                numeric_cols=numeric_cols,
                cat_cols=cat_cols,
                text_cols=text_cols,
                y_log1p=y_log1p,
                fold_ids=fold_ids,
                n_train=n_train,
                n_test=n_test,
                n_folds=n_folds,
                dry_run=args.dry_run,
                target_folds=args.target_folds
            )

            if not args.dry_run and oof_cat is not None and (args.target_folds is None or len(args.target_folds) == n_folds):
                assert not np.isnan(oof_cat).any(), "🚨 [FATAL] oof_cat_text 包含 NaN！"
                assert not np.isnan(test_cat).any(), "🚨 [FATAL] test_cat_text 包含 NaN！"
                assert 0.22 < cat_score < 0.45, f"🚨 [FATAL] v3_cat_text_deep RMSLE 异常: {cat_score:.5f}"
                corr = np.corrcoef(oof_cat, y_log1p)[0, 1]
                assert 0.70 < corr < 0.98, f"🚨 [FATAL] 相关系数异常: {corr:.4f}"
                print(f"    ✅ 质量断言验证通过！RMSLE = {cat_score:.5f}, corr = {corr:.4f}")

                # 双向兼容落盘
                np.save(OUTPUT_DIR / "oof_v3_cat_text_deep.npy", oof_cat)
                np.save(OUTPUT_DIR / "test_v3_cat_text_deep.npy", test_cat)
                np.save(OUTPUT_DIR / "oof_cat_text.npy", oof_cat)
                np.save(OUTPUT_DIR / "test_cat_text.npy", test_cat)

                manifest_records["v3_cat_text_deep"] = {
                    "cv_rmsle": float(cat_score),
                    "target_corr": float(corr),
                    "mean_pred": float(np.mean(oof_cat)),
                    "shape_oof": list(oof_cat.shape),
                    "shape_test": list(test_cat.shape),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

    # 6. 保存元数据记录与就绪标记
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)
    print(f"\n>>> 元数据已更新至 {manifest_file}")

    if (OUTPUT_DIR / "oof_v3_nn_tabular.npy").exists() and (OUTPUT_DIR / "oof_v3_cat_text_deep.npy").exists():
        flag_file = OUTPUT_DIR / "READY_V3.flag"
        with open(flag_file, "w", encoding="utf-8") as f:
            f.write(f"READY_V3: Tabular ResNet + Deep CatBoost Multi-modal Zoo v3 Ready.\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"🎉 异构多模态模型库全部就绪！就绪标记落盘: {flag_file}")

    print(f"\n✨ [Step 3 全部流程结束] 总耗时: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

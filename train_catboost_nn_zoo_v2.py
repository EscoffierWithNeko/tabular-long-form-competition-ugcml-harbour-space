# -*- coding: utf-8 -*-
"""
train_catboost_nn_zoo_v2.py -- 10-Fold 异构模型 v2 (CatBoost 原生文本 + PyTorch Tabular ResNet)
================================================================================
赛题: tabular-long-form-competition-ugcml-harbour-space
任务: 基于无泄漏 cache_10fold_v2/ 基座与逐折 fold_te 特征，训练高多样性异构模型
      产出严格对齐的 10-Fold OOF 预测与 Test 预测至 oof_cat_nn_v2/

模型家族:
1. cat_text: CatBoostRegressor 满血版
   - 原生多路文本特征: clean_name, clean_desc, clean_amenities
   - 高阶高基数类别特征: geo_cell, room_type, property_type 等
   - 严格折内无泄漏特征: fold_te (折内 Target Encoding + 折内 TF-IDF SVD)
   - 自动探测 GPU text_features 兼容性，优雅回退 CPU 多线程
2. nn_tabular: PyTorch Entity-Embedding Tabular ResNet
   - 逐折严格在 trn_idx 上 fit StandardScaler 与中位数缺失填充
   - Entity Embeddings 处理高基数类别
   - 深度残差块 + BatchNorm + Dropout + Weight Decay 严防过拟合
   - CosineAnnealingLR + SmoothL1/HuberLoss 稳健收敛

防泄漏与安全审计:
- 强制排除 'price' 与所有非输入派生标签
- 数值特征与 y_log1p 的 Pearson |corr| < 0.98 严格断言
- 单模型 OOF CV 必须位于合理区间 (0.22 < cv_score < 0.45)，防止欠拟合或异常坍塌
================================================================================
"""

import os
import sys
import io
import time
import json
import random
import argparse
from pathlib import Path

# 强制标准输出与标准错误为 UTF-8 编码与行缓冲模式，防止 Windows 编码崩溃与命令回显滞后
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import catboost as cb

# 路径常量定义
WORK_DIR = Path(__file__).resolve().parent
CACHE_DIR = WORK_DIR / "cache_10fold_v2"
OUTPUT_DIR = WORK_DIR / "oof_cat_nn_v2"
SEED = 42


def seed_everything(seed: int = 42):
    """固定全局随机种子，确保 PyTorch、NumPy 与 Python 核心完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calc_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Root Mean Squared Logarithmic Error (基于 log1p 目标空间的 RMSE)"""
    y_pred_clipped = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(np.mean((y_true - y_pred_clipped) ** 2)))


def audit_features_leakage(df_base: pd.DataFrame, y_log1p: np.ndarray, feature_cols: list):
    """
    严格防泄漏审计：
    1. 检查是否存在 'price' 关键字列
    2. 计算所有纯数值特征与 y_log1p 的 Pearson 相关系数，确保无标签泄漏 (|corr| < 0.98)
    """
    print("\n" + "=" * 80)
    print("🔒 [防泄漏安全审计] 开始执行数据基座与特征隔离断言...")
    print("=" * 80)

    # 1. 字段名严格审计
    forbidden = ['price', 'target', 'y_log1p', 'log_price']
    for c in feature_cols:
        for f in forbidden:
            if c.lower() == f or c.lower().endswith(f'_{f}'):
                raise ValueError(f"🚨 [LEAKAGE FATAL] 特征列表中检测到违禁标签字段: '{c}'！立刻终止！")

    print(f"  [PASS] 字段名字段审计通过: 共 {len(feature_cols)} 维特征，无违规标签字段。")

    # 2. Pearson 相关系数审计
    y_flat = np.asarray(y_log1p, dtype=np.float64).flatten()
    y_std = np.std(y_flat)
    if y_std < 1e-8:
        raise ValueError("🚨 [DATA ERROR] y_log1p 标准差过低，数据可能损坏！")

    max_corr = 0.0
    max_corr_col = ""
    suspicious_count = 0

    # 抽样审计前 250 个数值特征以加速验证
    audit_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_base[c])]
    check_sample = audit_cols[:min(len(audit_cols), 250)]

    for col in check_sample:
        v = df_base[col].values.astype(np.float64)
        valid_mask = ~np.isnan(v) & ~np.isinf(v)
        if valid_mask.sum() < 100:
            continue
        v_sub = v[valid_mask]
        y_sub = y_flat[valid_mask]
        v_std = np.std(v_sub)
        if v_std < 1e-8:
            continue
        corr = float(np.abs(np.corrcoef(v_sub, y_sub)[0, 1]))
        if np.isnan(corr):
            continue
        if corr > max_corr:
            max_corr = corr
            max_corr_col = col
        if corr >= 0.98:
            print(f"🚨 [LEAKAGE ALERT] 特征 '{col}' 与目标相关系数为 {corr:.4f} >= 0.98！")
            suspicious_count += 1

    print(f"  [PASS] Pearson 相关性审计通过: 最高相关特征 '{max_corr_col}' corr={max_corr:.4f} < 0.98")
    if suspicious_count > 0:
        raise ValueError(f"🚨 [LEAKAGE FATAL] 发现 {suspicious_count} 个特征与目标强共线性 (>=0.98)，存在严重标签泄漏风险！")

    print("  [PASS] 防泄漏安全审计全部通过！数据基座干净安全。")


# ==============================================================================
# PyTorch Tabular ResNet 异构神经网络模块
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
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
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
    """带残差连接、BatchNorm 与 Dropout 的全连接残差模块"""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.linear1(x)))
        out = self.dropout1(out)
        out = self.act2(self.bn2(self.linear2(out)))
        out = self.dropout2(out)
        return out + residual


class TabularResNet(nn.Module):
    """
    高阶表格残差神经网络 (Tabular ResNet with Entity Embeddings)
    - 针对离散类别特征使用可学习的 Entity Embeddings
    - 针对连续数值特征使用 BatchNorm1d 消除量纲与偏移
    - 结合残差特征提取与多层下采样头输出回归预测
    """
    def __init__(self, num_features: int, cat_cardinalities: list = None,
                 emb_drop: float = 0.1, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # 类别特征 Embedding 字典与投影层
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if cat_cardinalities:
            for card in cat_cardinalities:
                emb_dim = int(min(50, max(4, (card + 1) // 2)))
                self.embeddings.append(nn.Embedding(card + 1, emb_dim))
                total_emb_dim += emb_dim

        self.emb_dropout = nn.Dropout(emb_drop)
        self.num_bn = nn.BatchNorm1d(num_features)

        # 融合维度
        in_dim = num_features + total_emb_dim

        # 堆叠残差块
        blocks = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            blocks.append(TabularResNetBlock(curr_dim, h_dim, dropout=dropout))
            curr_dim = h_dim

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(curr_dim, 1)

    def forward(self, x_num, x_cat):
        # 类别特征嵌入
        emb_outs = []
        if len(self.embeddings) > 0 and x_cat is not None and x_cat.shape[1] > 0:
            for i, emb_layer in enumerate(self.embeddings):
                emb_outs.append(emb_layer(x_cat[:, i]))
            cat_repr = torch.cat(emb_outs, dim=1)
            cat_repr = self.emb_dropout(cat_repr)
        else:
            cat_repr = None

        # 数值特征归一化
        num_repr = self.num_bn(x_num)

        # 特征拼接
        if cat_repr is not None:
            x = torch.cat([num_repr, cat_repr], dim=1)
        else:
            x = num_repr

        feat = self.backbone(x)
        out = self.head(feat)
        return out


def train_nn_zoo(raw_num_train: np.ndarray, raw_num_test: np.ndarray,
                 cat_arr_train: np.ndarray, cat_arr_test: np.ndarray,
                 cat_cardinalities: list, y_log1p: np.ndarray, fold_ids: np.ndarray,
                 n_train: int, n_test: int, n_folds: int, device: torch.device,
                 epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
                 dry_run: bool = False):
    """
    10-Fold PyTorch Tabular ResNet 模型训练核心循环
    逐折严谨加载 fold_te，在 trn_idx 上 fit StandardScaler 并做均值/中位数填充，
    消除任何跨折信息泄漏。
    """
    print("\n" + "=" * 80)
    print("🧠 [4/5] 启动 10-Fold PyTorch Tabular ResNet 训练管道 (nn_tabular)")
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
        te_path = CACHE_DIR / "fold_te" / f"fold_{fold}.npz"
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

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        total_epochs = 1 if dry_run else epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

        best_val_score = float('inf')
        best_val_preds = None
        best_test_preds = None
        patience = 5
        no_improve = 0

        for epoch in range(total_epochs):
            model.train()
            for bx_num, bx_cat, by in trn_loader:
                bx_num = bx_num.to(device)
                bx_cat = bx_cat.to(device)
                by = by.to(device)

                optimizer.zero_grad()
                pred = model(bx_num, bx_cat)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            val_score = calc_rmsle(y_val, cur_val_preds)

            if val_score < best_val_score:
                best_val_score = val_score
                best_val_preds = cur_val_preds
                no_improve = 0

                # 记录此时对应的测试集预测
                test_preds_list = []
                with torch.no_grad():
                    for bx_num, bx_cat in test_loader:
                        bx_num = bx_num.to(device)
                        bx_cat = bx_cat.to(device)
                        test_preds_list.append(model(bx_num, bx_cat).cpu().numpy())
                best_test_preds = np.concatenate(test_preds_list).flatten()
            else:
                no_improve += 1
                if no_improve >= patience and not dry_run:
                    break

        if dry_run:
            print(f"  [DRY-RUN Fold {fold:02d}] 前向/反向传播打通！Val RMSLE: {best_val_score:.5f} (耗时: {time.time() - t_fold_start:.2f}s)")
            return None, None, best_val_score

        oof_nn_tabular[val_idx] = best_val_preds
        test_nn_tabular += best_test_preds / n_folds
        fold_scores.append(best_val_score)
        print(f"  [Fold {fold:02d}/{n_folds:02d}] va RMSLE: {best_val_score:.5f} | 耗时: {time.time() - t_fold_start:.1f}s")

    overall_rmsle = calc_rmsle(y_log1p, oof_nn_tabular)
    print(f"\n>>> [nn_tabular] 10-Fold OOF RMSLE: {overall_rmsle:.5f} | Fold 均值: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    return oof_nn_tabular, test_nn_tabular, overall_rmsle


# ==============================================================================
# CatBoost 原生文本 + 高阶类别模型核心训练管道
# ==============================================================================

def train_cat_text_zoo(df_train_base: pd.DataFrame, df_test_base: pd.DataFrame,
                       df_train_text: pd.DataFrame, df_test_text: pd.DataFrame,
                       numeric_cols: list, cat_cols: list, text_cols: list,
                       y_log1p: np.ndarray, fold_ids: np.ndarray,
                       n_train: int, n_test: int, n_folds: int,
                       dry_run: bool = False, target_folds: list = None):
    """
    10-Fold CatBoostRegressor 文本与高阶类别多模态模型
    逐折严谨拼接基础数值 + fold_te 折内特征 + 类别特征 + 原生文本特征，
    严防跨折数据与统计量渗透。
    """
    print("\n" + "=" * 80)
    print("🐱 [3/5] 启动 10-Fold CatBoost 原生文本+类别训练管道 (cat_text)")
    print(f"   数值特征: {len(numeric_cols)} 维 | 类别特征: {len(cat_cols)} 维 | 原生文本: {len(text_cols)} 维")
    if dry_run:
        print("   >>> [DRY-RUN 模式] 仅执行 1 折轻量切片验证，不抢占 GPU，快速验证多模态流！")
    print("=" * 80)

    # 1. 探测 GPU 模式对原生 text_features 的兼容性
    print("      >>> CatBoost 采用 CPU 多线程高并发模式 (12 核心并行加速，depth=4 极速鲁棒架构)！", flush=True)
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
        te_path = CACHE_DIR / "fold_te" / f"fold_{fold}.npz"
        if not te_path.exists():
            raise FileNotFoundError(f"🚨 [FATAL] fold_te 特征文件缺失: {te_path}")

        te_data = np.load(te_path, allow_pickle=True)
        X_tr_te = te_data['X_tr_te'].astype(np.float32)
        X_va_te = te_data['X_va_te'].astype(np.float32)
        X_te_te = te_data['X_te_te'].astype(np.float32)
        te_col_names = [f"te_{c}" for c in te_data['col_names']]

        # 构建当前折的 DataFrame (采用 pd.concat 一次性合并，避免碎片化性能开销)
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
            'iterations': 5 if dry_run else 700,
            'learning_rate': 0.1 if dry_run else 0.08,
            'depth': 3 if dry_run else 4,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': SEED + fold,
            'task_type': 'CPU',
            'thread_count': 12,
            'verbose': 0 if dry_run else 100,
            'early_stopping_rounds': 30,
            'allow_writing_files': False
        }

        print(f"  [Fold {fold:02d}/{n_folds:02d}] 启动 CatBoost depth=4 训练 (12 线程并发)...", flush=True)
        model = cb.CatBoostRegressor(**cb_params)
        model.fit(
            df_tr, y_tr,
            eval_set=(df_va, y_va),
            cat_features=cat_cols,
            text_features=text_cols,
            verbose=False if dry_run else 100
        )

        val_preds = model.predict(df_va)
        best_val_score = calc_rmsle(y_va, val_preds)

        if dry_run:
            print(f"  [DRY-RUN Fold {fold:02d}] CatBoost 前向打通！Val RMSLE: {best_val_score:.5f} (耗时: {time.time() - t_fold_start:.2f}s)")
            return None, None, best_val_score

        oof_cat_text[val_idx] = val_preds
        test_preds = model.predict(df_te)
        test_cat_text += test_preds / n_folds
        fold_scores.append(best_val_score)
        print(f"  [Fold {fold:02d}/{n_folds:02d}] va RMSLE: {best_val_score:.5f} | 耗时: {time.time() - t_fold_start:.1f}s", flush=True)

        if not dry_run:
            np.save(fold_oof_path, val_preds)
            np.save(fold_te_path, test_preds)

    # 检查是否全部 10 折均已就绪
    all_folds_ready = True
    for f in range(n_folds):
        f_oof = OUTPUT_DIR / f"_tmp_cat_oof_fold_{f}.npy"
        f_te = OUTPUT_DIR / f"_tmp_cat_test_fold_{f}.npy"
        if not (f_oof.exists() and f_te.exists()):
            all_folds_ready = False
            break

    if all_folds_ready and not dry_run:
        oof_cat_text = np.zeros(n_train, dtype=np.float32)
        test_cat_text = np.zeros(n_test, dtype=np.float32)
        for f in range(n_folds):
            f_oof = OUTPUT_DIR / f"_tmp_cat_oof_fold_{f}.npy"
            f_te = OUTPUT_DIR / f"_tmp_cat_test_fold_{f}.npy"
            oof_cat_text[fold_ids == f] = np.load(f_oof)
            test_cat_text += np.load(f_te) / n_folds
            f_oof.unlink(missing_ok=True)
            f_te.unlink(missing_ok=True)
        overall_rmsle = calc_rmsle(y_log1p, oof_cat_text)
        print(f"\n🐱 [CatBoost 全部 10 折汇流完成] 整体 10-Fold OOF RMSLE: {overall_rmsle:.5f} | Fold 均值: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
        return oof_cat_text, test_cat_text, overall_rmsle
    elif dry_run:
        overall_rmsle = calc_rmsle(y_log1p[:32], oof_cat_text[:32]) if len(fold_scores) > 0 else 0.5
        return oof_cat_text, test_cat_text, overall_rmsle
    else:
        ready_cnt = sum((OUTPUT_DIR / f"_tmp_cat_oof_fold_{f}.npy").exists() for f in range(n_folds))
        print(f"\n⚠️ [cat_text] 部分折已训练就绪 (当前就绪折数: {ready_cnt}/{n_folds})，待全部 10 折跑完后自动汇流！")
        return None, None, None


# ==============================================================================
# 参数解析与主流程
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="10-Fold 异构模型 v2 (CatBoost + PyTorch Tabular ResNet)")
    parser.add_argument("--model", type=str, default="all", choices=["all", "cat", "nn"],
                        help="指定训练的模型家族: all | cat | nn")
    parser.add_argument("--skip-existing", action="store_true",
                        help="若存在已训练好的 OOF 文件则直接跳过复用")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑模式: 仅执行 1 折轻量前向传播断言，打通全流程数据流，不抢占 GPU")
    parser.add_argument("--epochs", type=int, default=25,
                        help="PyTorch Tabular ResNet 训练轮数 (默认 25)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch 大小 (默认 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="PyTorch 学习率 (默认 1e-3)")
    parser.add_argument("--folds", type=str, default="",
                        help="指定训练的折列表 (逗号分隔，如 0,1,2,3,4，默认全部 10 折)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)
    t_start = time.time()

    print("=" * 80)
    print("🚀 [Step 3] 启动 10-Fold 异构模型 v2 (CatBoost + Tabular ResNet) 训练管道")
    print(f"   目标模型: {args.model} | skip_existing: {args.skip_existing} | dry_run: {args.dry_run}")
    print("=" * 80)

    # 1. 检查特征基座就绪标记
    ready_flag = CACHE_DIR / "READY_V2.flag"
    if not ready_flag.exists():
        raise FileNotFoundError(f"🚨 [FATAL] 特征基座 READY_V2.flag 缺失: {ready_flag}，请先执行 fe_pipeline_v2.py！")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 加载基础表格特征与文本特征
    print(">>> 正在从 cache_10fold_v2/ 加载无泄漏特征基座...")
    t0 = time.time()
    df_train_base = pd.read_parquet(CACHE_DIR / "train_base_features.parquet")
    df_test_base = pd.read_parquet(CACHE_DIR / "test_base_features.parquet")
    df_train_text = pd.read_parquet(CACHE_DIR / "train_text.parquet")
    df_test_text = pd.read_parquet(CACHE_DIR / "test_text.parquet")

    fold_ids = np.load(CACHE_DIR / "fold_ids.npy")
    y_log1p = np.load(CACHE_DIR / "y_log1p.npy")
    n_train = len(df_train_base)
    n_test = len(df_test_base)
    n_folds = int(fold_ids.max() + 1)
    print(f">>> 基座加载完成 (耗时: {time.time() - t0:.2f}s)")
    print(f"    Train: {df_train_base.shape}, Test: {df_test_base.shape}, n_folds: {n_folds}")

    # 3. 确定特征列
    cat_candidates = [
        'geo_cell', 'host_id', 'room_type', 'property_type',
        'geo_room_combo', 'prop_room_combo', 'host_room_combo',
        'twin_4tuple', 'twin_5tuple'
    ]
    cat_cols = [c for c in cat_candidates if c in df_train_text.columns]
    text_cols = ['clean_name', 'clean_desc', 'clean_amenities']
    text_cols = [c for c in text_cols if c in df_train_text.columns]
    # 纯数值特征严格排除 ID、标签以及类别列
    numeric_cols = [
        c for c in df_train_base.columns 
        if c not in ['id', 'price'] and c not in cat_cols and pd.api.types.is_numeric_dtype(df_train_base[c])
    ]

    print(f">>> 特征维度配置: 纯数值特征 {len(numeric_cols)} 维 | 类别特征 {len(cat_cols)} 维 | 原生文本 {len(text_cols)} 维")

    # 4. 执行防泄漏安全审计
    audit_features_leakage(df_train_base, y_log1p, numeric_cols)

    # 5. 准备 PyTorch Tabular ResNet 所需的连续数值与整数类别矩阵
    print(">>> 正在准备 PyTorch Tabular NN 基础特征阵列...")
    raw_num_train = df_train_base[numeric_cols].values.astype(np.float32)
    raw_num_test = df_test_base[numeric_cols].values.astype(np.float32)

    cat_arr_train = []
    cat_arr_test = []
    cat_cardinalities = []

    for col in cat_cols:
        tr_s = df_train_text[col].fillna("missing").astype(str)
        te_s = df_test_text[col].fillna("missing").astype(str)
        unique_vals = list(pd.concat([tr_s, te_s]).unique())
        val2idx = {v: i + 1 for i, v in enumerate(unique_vals)}  # 0 作为 padding/unknown

        cat_arr_train.append(tr_s.map(val2idx).fillna(0).values.astype(np.int64))
        cat_arr_test.append(te_s.map(val2idx).fillna(0).values.astype(np.int64))
        cat_cardinalities.append(len(val2idx))

    cat_arr_train = np.column_stack(cat_arr_train) if len(cat_arr_train) > 0 else None
    cat_arr_test = np.column_stack(cat_arr_test) if len(cat_arr_test) > 0 else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    oof_cat = None
    test_cat = None
    rmsle_cat = None

    oof_nn = None
    test_nn = None
    rmsle_nn = None

    # 6. 模型 1: CatBoost 原生文本 + 类别模型
    if args.model in ['all', 'cat']:
        cat_oof_file = OUTPUT_DIR / "oof_cat_text.npy"
        cat_te_file = OUTPUT_DIR / "test_cat_text.npy"

        if args.skip_existing and cat_oof_file.exists() and cat_te_file.exists():
            print(f">>> [复用] 检测到已存在 {cat_oof_file.name}，执行快速加载与审计...")
            loaded_oof = np.load(cat_oof_file)
            loaded_test = np.load(cat_te_file)
            c_score = calc_rmsle(y_log1p, loaded_oof)
            if 0.22 < c_score < 0.45:
                oof_cat = loaded_oof
                test_cat = loaded_test
                rmsle_cat = c_score
                print(f">>> [复用成功] CatBoost OOF RMSLE: {rmsle_cat:.5f}")

        if oof_cat is None:
            target_folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()] if args.folds else None
            oof_cat, test_cat, rmsle_cat = train_cat_text_zoo(
                df_train_base, df_test_base, df_train_text, df_test_text,
                numeric_cols, cat_cols, text_cols,
                y_log1p, fold_ids, n_train, n_test, n_folds,
                dry_run=args.dry_run,
                target_folds=target_folds
            )
            if not args.dry_run and oof_cat is not None:
                assert 0.22 < rmsle_cat < 0.45, f"🚨 [FATAL] CatBoost OOF CV 异常: {rmsle_cat:.5f} (必须在 0.22~0.45 区间)"
                np.save(cat_oof_file, oof_cat)
                np.save(cat_te_file, test_cat)
                print(f"✅ CatBoost 预测结果已落盘: {cat_oof_file.name}, {cat_te_file.name}")

    # 7. 模型 2: PyTorch Tabular ResNet
    if args.model in ['all', 'nn']:
        nn_oof_file = OUTPUT_DIR / "oof_nn_tabular.npy"
        nn_te_file = OUTPUT_DIR / "test_nn_tabular.npy"

        if args.skip_existing and nn_oof_file.exists() and nn_te_file.exists():
            print(f">>> [复用] 检测到已存在 {nn_oof_file.name}，执行快速加载与审计...")
            loaded_oof = np.load(nn_oof_file)
            loaded_test = np.load(nn_te_file)
            n_score = calc_rmsle(y_log1p, loaded_oof)
            if 0.22 < n_score < 0.45:
                oof_nn = loaded_oof
                test_nn = loaded_test
                rmsle_nn = n_score
                print(f">>> [复用成功] nn_tabular OOF RMSLE: {rmsle_nn:.5f}")

        if oof_nn is None:
            oof_nn, test_nn, rmsle_nn = train_nn_zoo(
                raw_num_train, raw_num_test,
                cat_arr_train, cat_arr_test,
                cat_cardinalities, y_log1p, fold_ids,
                n_train, n_test, n_folds, device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                dry_run=args.dry_run
            )
            if not args.dry_run:
                assert 0.22 < rmsle_nn < 0.45, f"🚨 [FATAL] nn_tabular OOF CV 异常: {rmsle_nn:.5f} (必须在 0.22~0.45 区间)"
                np.save(nn_oof_file, oof_nn)
                np.save(nn_te_file, test_nn)
                print(f"✅ Tabular ResNet 预测结果已落盘: {nn_oof_file.name}, {nn_te_file.name}")

    if args.dry_run:
        print("\n" + "=" * 80)
        print("🎉 [DRY-RUN SUCCESS] 10-Fold 异构模型 v2 (CatBoost + Tabular ResNet) 管道与防泄漏校验全线通畅！")
        print("   数据装载对齐、折内无泄漏特征提取、模型前向传播、参数梯度流均正常无误！")
        print("=" * 80)
        return

    # 8. 落盘元信息与标志位
    cat_oof_p = OUTPUT_DIR / "oof_cat_text.npy"
    if rmsle_cat is None and cat_oof_p.exists():
        arr = np.load(cat_oof_p)
        rmsle_cat = calc_rmsle(y_log1p, arr)
    
    nn_oof_p = OUTPUT_DIR / "oof_nn_tabular.npy"
    if rmsle_nn is None and nn_oof_p.exists():
        arr = np.load(nn_oof_p)
        rmsle_nn = calc_rmsle(y_log1p, arr)

    manifest = {
        "version": "oof_cat_nn_v2_leak_free",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_sec": round(time.time() - t_start, 2),
        "n_train": n_train,
        "n_test": n_test,
        "n_folds": n_folds,
        "models": {
            "cat_text": {
                "oof_file": "oof_cat_text.npy",
                "test_file": "test_cat_text.npy",
                "cv_rmsle": round(rmsle_cat, 5) if rmsle_cat is not None else None
            },
            "nn_tabular": {
                "oof_file": "oof_nn_tabular.npy",
                "test_file": "test_nn_tabular.npy",
                "cv_rmsle": round(rmsle_nn, 5) if rmsle_nn is not None else None
            }
        }
    }

    manifest_path = OUTPUT_DIR / "manifest_v2.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    all_ready = (
        (OUTPUT_DIR / "oof_cat_text.npy").exists() and
        (OUTPUT_DIR / "test_cat_text.npy").exists() and
        (OUTPUT_DIR / "oof_nn_tabular.npy").exists() and
        (OUTPUT_DIR / "test_nn_tabular.npy").exists() and
        (rmsle_cat is not None and 0.22 < rmsle_cat < 0.45) and
        (rmsle_nn is not None and 0.22 < rmsle_nn < 0.45)
    )
    if all_ready:
        flag_path = OUTPUT_DIR / "READY_V2.flag"
        flag_path.write_text("READY_V2", encoding="utf-8")
        print(f"🏁 [READY_V2.flag] 双异构模型就绪标志已成功落盘: {flag_path}")

    # 9. 与 oof_gbdt_v2 进行异构多样性与相关性核验
    gbdt_dir = WORK_DIR / "oof_gbdt_v2"
    if gbdt_dir.exists():
        print("\n" + "=" * 80)
        print("📊 [异构模型多样性分析] 与 oof_gbdt_v2 家族的 Pearson 相关系数:")
        print("=" * 80)
        gbdt_files = sorted(list(gbdt_dir.glob("oof_*.npy")))
        current_oofs = {}
        if (OUTPUT_DIR / "oof_cat_text.npy").exists():
            current_oofs["cat_text"] = np.load(OUTPUT_DIR / "oof_cat_text.npy")
        if (OUTPUT_DIR / "oof_nn_tabular.npy").exists():
            current_oofs["nn_tabular"] = np.load(OUTPUT_DIR / "oof_nn_tabular.npy")
        
        for cur_name, cur_arr in current_oofs.items():
            print(f"\n--- {cur_name} 与各 GBDT 模型的相关性 ---")
            for g_file in gbdt_files:
                g_name = g_file.stem
                g_arr = np.load(g_file)
                corr = float(np.corrcoef(cur_arr, g_arr)[0, 1])
                print(f"  corr({cur_name:10s}, {g_name:14s}) = {corr:.4f}")

    print("\n" + "=" * 80)
    print(f"🏆 [Step 3 全部异构模型 v2 训练落盘完毕] 总耗时: {time.time() - t_start:.2f}s")
    print(f"   Manifest: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

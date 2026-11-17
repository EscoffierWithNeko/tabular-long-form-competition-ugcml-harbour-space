# -*- coding: utf-8 -*-
"""
fe_pipeline_v2.py -- 严格折内无泄漏 10-Fold 高阶特征工程基座 v2
===================================================================
核心升级定位:
1. 严格继承与复用已审计的 cache_10fold/ 基础资产与文本语料。
2. 基础特征扩充 (train_base_v2 / test_base_v2):
   - prop_room_combo 与 host_room_combo 组内相对位置 (diff / ratio / zscore)
   - geo_cell 与 geo_room_combo 下 luxury_amenities_sum, amenities_count, space_capacity_index 组内统计
   - 房东规模画像 (host_accommodates_mean/std, host_bedrooms_mean, is_professional_host)
   - 房源高端关键词与伦敦高价地段关键词正则提取
   - 非线性对数变换与 instant_bookable_bin 字段
3. 严格 10-Fold 折内特征基准 (cache_10fold_v2/fold_te/fold_{fold}.npz):
   - 【核心升级 1】LightGBM Peer-Baseline 残差基准 (inner 5-Fold 预测与非线性残差 res_lgb)
   - 【核心升级 2】全实体多平滑 + 分位数残差 TE & 单人容纳 TE (7大实体, inner 5-Fold 隔离)
   - 【核心升级 3】Text Zoo v2 + 8维 TruncatedSVD (name/desc/full/amen 多路 TF-IDF + Ridge)
4. 输出无泄漏完整矩阵至 cache_10fold_v2/，并通过 READY_V2.flag 亮起验证。
"""

import os
import re
import gc
import json
import time
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 0. 全局常量与工作目录配置
# -----------------------------------------------------------------------------
SEED = 42
WORKSPACE_DIR = Path(r"F:\desktop\MyRouter\Nekoya\nekolab\[Kaggle]tabular-long-form-competition-ugcml-harbour-space")
CACHE_V1_DIR = WORKSPACE_DIR / 'cache_10fold'
CACHE_V2_DIR = WORKSPACE_DIR / 'cache_10fold_v2'


def clean_html_text(text: str) -> str:
    """清理 HTML 标签、多余空格和特殊转义字符"""
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_smoothed_mean(sums, counts, global_mean, m_smooth):
    """经验贝叶斯平滑均值"""
    return (sums + m_smooth * global_mean) / (counts + m_smooth)


def compute_group_quantile_iqr(series_val, series_cat, tr_mask=None):
    """计算分组中位数与 IQR (q75 - q25)"""
    df_tmp = pd.DataFrame({'cat': series_cat, 'val': series_val})
    if tr_mask is not None:
        df_tmp = df_tmp.iloc[tr_mask]
    grp = df_tmp.groupby('cat')['val']
    q25 = grp.quantile(0.25)
    q50 = grp.quantile(0.50)
    q75 = grp.quantile(0.75)
    iqr = q75 - q25
    return q50, iqr


# -----------------------------------------------------------------------------
# 1. 构建 v2 基础表格特征 (基于 v1 资产增量扩充，严格防泄漏)
# -----------------------------------------------------------------------------
def build_v2_base_features(train_base_v1: pd.DataFrame, test_base_v1: pd.DataFrame,
                          train_csv: pd.DataFrame, test_csv: pd.DataFrame,
                          train_text: pd.DataFrame, test_text: pd.DataFrame):
    """
    在已有无泄漏 train_base_features / test_base_features 基础上，
    严格按照任务书扩充无标签 v2 基础特征。
    """
    print(">>> [Step 1] 开始构建 v2 无标签基础特征扩充...")
    t0 = time.time()
    n_train = len(train_base_v1)
    
    # 合并基座特征表 (已包含 v1 衍生的基础字段)
    df_base = pd.concat([train_base_v1, test_base_v1], axis=0, ignore_index=True)
    df_raw = pd.concat([train_csv, test_csv], axis=0, ignore_index=True)
    df_txt = pd.concat([train_text, test_text], axis=0, ignore_index=True)
    
    # 彻底杜绝泄漏：显式检查并删除 price 列
    if 'price' in df_base.columns:
        df_base = df_base.drop(columns=['price'])
        
    # 1. 补全 prop_room_combo 与 host_room_combo 组内相对位置特征 (diff / ratio / zscore)
    # 涉及字段: accommodates, bedrooms_safe, bathrooms_safe, luxury_amenities_sum, review_scores_rating
    combo_target_cols = ['accommodates', 'bedrooms_safe', 'bathrooms_safe', 'luxury_amenities_sum', 'review_scores_rating']
    for grp_col in ['prop_room_combo', 'host_room_combo']:
        if grp_col not in df_base.columns:
            if grp_col == 'prop_room_combo':
                df_base['prop_room_combo'] = df_base['property_type'].astype(str) + '__' + df_base['room_type'].astype(str)
            else:
                df_base['host_room_combo'] = df_base['host_id'].astype(str) + '__' + df_base['room_type'].astype(str)
                
        for val_col in combo_target_cols:
            mean_s = df_base.groupby(grp_col)[val_col].transform('mean')
            std_s = df_base.groupby(grp_col)[val_col].transform('std').fillna(1.0)
            df_base[f'{val_col}_diff_to_{grp_col}'] = (df_base[val_col] - mean_s).astype(np.float32)
            df_base[f'{val_col}_ratio_to_{grp_col}'] = (df_base[val_col] / (mean_s + 0.1)).astype(np.float32)
            df_base[f'{val_col}_zscore_to_{grp_col}'] = ((df_base[val_col] - mean_s) / (std_s + 1e-4)).astype(np.float32)
            
    # 2. 补充 geo_cell 与 geo_room_combo 下 luxury_amenities_sum, amenities_count, space_capacity_index 组内统计
    geo_extra_vals = ['luxury_amenities_sum', 'amenities_count', 'space_capacity_index']
    for grp_col in ['geo_cell', 'geo_room_combo']:
        for val_col in geo_extra_vals:
            if val_col in df_base.columns:
                mean_s = df_base.groupby(grp_col)[val_col].transform('mean')
                std_s = df_base.groupby(grp_col)[val_col].transform('std').fillna(1.0)
                df_base[f'{val_col}_diff_to_{grp_col}'] = (df_base[val_col] - mean_s).astype(np.float32)
                df_base[f'{val_col}_ratio_to_{grp_col}'] = (df_base[val_col] / (mean_s + 0.1)).astype(np.float32)
                df_base[f'{val_col}_zscore_to_{grp_col}'] = ((df_base[val_col] - mean_s) / (std_s + 1e-4)).astype(np.float32)

    # 3. 房东规模画像 (host_accommodates_mean/std, host_bedrooms_mean, is_professional_host)
    df_base['host_accommodates_mean'] = df_base.groupby('host_id')['accommodates'].transform('mean').astype(np.float32)
    df_base['host_accommodates_std'] = df_base.groupby('host_id')['accommodates'].transform('std').fillna(0).astype(np.float32)
    df_base['host_bedrooms_mean'] = df_base.groupby('host_id')['bedrooms_safe'].transform('mean').astype(np.float32)
    
    calc_listings = df_base['calculated_host_listings_count'].fillna(1.0)
    df_base['is_professional_host'] = (calc_listings >= 5.0).astype(np.float32)
    
    # 4. 房源关键词与价格信号特征 (从 clean_name 与 clean_desc 正则匹配)
    clean_name_s = df_txt['clean_name'].fillna('').astype(str).str.lower()
    clean_desc_s = df_txt['clean_desc'].fillna('').astype(str).str.lower()
    
    price_keywords = [
        'luxury', 'penthouse', 'villa', 'boutique', 'designer', 'studio', 'shared', 
        'budget', 'central', 'garden', 'balcony', 'terrace', 'ensuite', 'river', 
        'panoramic', 'historic', 'renovated'
    ]
    for kw in price_keywords:
        pat = rf'\b{kw}\b'
        df_base[f'v2_name_kw_{kw}'] = clean_name_s.str.contains(pat, regex=True).astype(np.float32)
        df_base[f'v2_desc_kw_{kw}'] = clean_desc_s.str.contains(pat, regex=True).astype(np.float32)
        
    london_prime_locations = [
        'kensington', 'mayfair', 'chelsea', 'soho', 'westminster', 
        'notting hill', 'marylebone', 'knightsbridge', 'canary wharf', 
        'shoreditch', 'camden'
    ]
    for loc in london_prime_locations:
        safe_loc = loc.replace(' ', '_')
        pat = rf'\b{loc}\b'
        df_base[f'v2_name_loc_{safe_loc}'] = clean_name_s.str.contains(pat, regex=True).astype(np.float32)
        df_base[f'v2_desc_loc_{safe_loc}'] = clean_desc_s.str.contains(pat, regex=True).astype(np.float32)

    # 5. 非线性变换与 instant_bookable_bin
    df_base['log1p_accommodates'] = np.log1p(df_base['accommodates'].clip(lower=0)).astype(np.float32)
    df_base['log1p_total_rooms'] = np.log1p(df_base['total_rooms'].clip(lower=0)).astype(np.float32)
    df_base['log1p_minimum_nights'] = np.log1p(df_base['minimum_nights'].fillna(1.0).clip(lower=0)).astype(np.float32)
    df_base['log1p_number_of_reviews'] = np.log1p(df_base['number_of_reviews'].fillna(0).clip(lower=0)).astype(np.float32)
    df_base['log1p_host_listings_count'] = np.log1p(df_base['host_listings_count'].fillna(1.0).clip(lower=0)).astype(np.float32)
    
    if 'instant_bookable' in df_raw.columns:
        ib_s = df_raw['instant_bookable'].astype(str).str.lower()
        df_base['instant_bookable_bin'] = ib_s.isin(['t', 'true', '1']).astype(np.float32)
    else:
        df_base['instant_bookable_bin'] = 0.0

    print(f">>> [Step 1] v2 基础特征构建完成，耗时: {time.time() - t0:.2f}s，当前特征总列数: {df_base.shape[1]}")
    
    train_base_v2 = df_base.iloc[:n_train].copy().reset_index(drop=True)
    test_base_v2 = df_base.iloc[n_train:].copy().reset_index(drop=True)
    
    # 严格防泄漏断言
    train_base_v2 = train_base_v2.drop(columns=['price'], errors='ignore')
    test_base_v2 = test_base_v2.drop(columns=['price'], errors='ignore')
    assert 'price' not in train_base_v2.columns, "Fatal: 'price' found in train_base_v2!"
    assert 'price' not in test_base_v2.columns, "Fatal: 'price' found in test_base_v2!"
    
    return train_base_v2, test_base_v2
# -----------------------------------------------------------------------------
# 2. 严格折内无泄漏 Target Encoding & Residual TE 辅助函数
# -----------------------------------------------------------------------------
def get_oof_te_for_series(tr_series, tr_target, va_series, te_series, inner_splits, m_smooth=10.0, calc_std=False, calc_median=False, calc_iqr=False):
    """
    通用严格无泄漏编码：
    tr 内部采用 inner_splits 进行 5-Fold OOF 统计，
    va 与 te 采用完整 tr 统计拟合映射。
    """
    # 强制重置 index 杜绝 pandas 字典组装时的 index 错位隐患
    tr_series = tr_series.reset_index(drop=True)
    va_series = va_series.reset_index(drop=True)
    te_series = te_series.reset_index(drop=True)
    
    n_tr = len(tr_series)
    n_va = len(va_series)
    n_te = len(te_series)
    
    global_mean = float(np.mean(tr_target))
    global_std = float(np.std(tr_target)) if calc_std else 0.0
    global_median = float(np.median(tr_target)) if (calc_median or calc_iqr) else 0.0
    global_iqr = float(np.percentile(tr_target, 75) - np.percentile(tr_target, 25)) if calc_iqr else 0.0
    
    if calc_median:
        default_val = global_median
    elif calc_iqr:
        default_val = global_iqr
    elif calc_std:
        default_val = global_std
    else:
        default_val = global_mean
        
    tr_out = np.full(n_tr, default_val, dtype=np.float32)
    va_out = np.full(n_va, default_val, dtype=np.float32)
    te_out = np.full(n_te, default_val, dtype=np.float32)
    
    # 1. 训练集 inner 5-Fold OOF
    for in_tr_idx, in_va_idx in inner_splits:
        sub_cat = tr_series.iloc[in_tr_idx].values
        sub_y = tr_target[in_tr_idx]
        val_s = tr_series.iloc[in_va_idx]
        
        if calc_median:
            sub_median = pd.DataFrame({'cat': sub_cat, 'y': sub_y}).groupby('cat')['y'].median()
            in_glob_med = float(np.median(sub_y))
            tr_out[in_va_idx] = val_s.map(sub_median).fillna(in_glob_med).values.astype(np.float32)
        elif calc_iqr:
            sub_df = pd.DataFrame({'cat': sub_cat, 'y': sub_y})
            grp = sub_df.groupby('cat')['y']
            q25 = grp.quantile(0.25)
            q75 = grp.quantile(0.75)
            sub_iqr = (q75 - q25).fillna(0.0)
            in_glob_iqr = float(np.percentile(sub_y, 75) - np.percentile(sub_y, 25))
            tr_out[in_va_idx] = val_s.map(sub_iqr).fillna(in_glob_iqr).values.astype(np.float32)
        elif calc_std:
            sub_std = pd.DataFrame({'cat': sub_cat, 'y': sub_y}).groupby('cat')['y'].std().fillna(0.0)
            in_glob_std = float(np.std(sub_y))
            tr_out[in_va_idx] = val_s.map(sub_std).fillna(in_glob_std).values.astype(np.float32)
        else:
            # 平滑均值
            sub_stats = pd.DataFrame({'cat': sub_cat, 'y': sub_y}).groupby('cat')['y'].agg(['sum', 'count'])
            in_glob_m = float(np.mean(sub_y))
            smoothed = ((sub_stats['sum'] + m_smooth * in_glob_m) / (sub_stats['count'] + m_smooth)).fillna(in_glob_m)
            tr_out[in_va_idx] = val_s.map(smoothed).fillna(in_glob_m).values.astype(np.float32)

    # 2. 验证集与测试集使用全量 tr 统计拟合
    full_cat = tr_series.values
    if calc_median:
        full_median = pd.DataFrame({'cat': full_cat, 'y': tr_target}).groupby('cat')['y'].median()
        va_out = va_series.map(full_median).fillna(global_median).values.astype(np.float32)
        te_out = te_series.map(full_median).fillna(global_median).values.astype(np.float32)
    elif calc_iqr:
        full_df = pd.DataFrame({'cat': full_cat, 'y': tr_target})
        grp = full_df.groupby('cat')['y']
        full_iqr = (grp.quantile(0.75) - grp.quantile(0.25)).fillna(global_iqr)
        va_out = va_series.map(full_iqr).fillna(global_iqr).values.astype(np.float32)
        te_out = te_series.map(full_iqr).fillna(global_iqr).values.astype(np.float32)
    elif calc_std:
        full_std = pd.DataFrame({'cat': full_cat, 'y': tr_target}).groupby('cat')['y'].std().fillna(global_std)
        va_out = va_series.map(full_std).fillna(global_std).values.astype(np.float32)
        te_out = te_series.map(full_std).fillna(global_std).values.astype(np.float32)
    else:
        full_stats = pd.DataFrame({'cat': full_cat, 'y': tr_target}).groupby('cat')['y'].agg(['sum', 'count'])
        smoothed = ((full_stats['sum'] + m_smooth * global_mean) / (full_stats['count'] + m_smooth)).fillna(global_mean)
        va_out = va_series.map(smoothed).fillna(global_mean).values.astype(np.float32)
        te_out = te_series.map(smoothed).fillna(global_mean).values.astype(np.float32)
        
    return tr_out, va_out, te_out


# -----------------------------------------------------------------------------
# 3. 严格 10-Fold 折内高阶特征工程核心流水线
# -----------------------------------------------------------------------------
def run_10fold_v2_feature_pipeline(train_base_v2: pd.DataFrame, test_base_v2: pd.DataFrame,
                                  train_text_df: pd.DataFrame, test_text_df: pd.DataFrame,
                                  y_log1p: np.ndarray, fold_ids: np.ndarray):
    """
    运行完整的 10-Fold 折内高阶流水线:
    1. LightGBM Peer-Baseline 残差基准与 res_lgb 提取
    2. 7大核心实体 Target TE, Residual TE, Unit TE
    3. Text Zoo v2 + 8维 TruncatedSVD
    4. 折内 npz 严谨保存与防泄漏核验
    """
    print(">>> [Step 2] 启动严格 10-Fold 折内无泄漏特征提取管道...")
    t_start = time.time()
    n_train = len(train_base_v2)
    n_test = len(test_base_v2)
    n_folds = 10
    
    # 排除非纯数值和复合键字段，筛选用于 Peer-LGB 的基座纯数值特征
    exclude_for_num = [
        'id', 'price', 'name', 'description', 'amenities', 'bathrooms_text',
        'first_review', 'last_review', 'host_is_superhost', 'host_identity_verified',
        'host_has_profile_pic', 'geo_cell', 'host_id', 'property_type', 'room_type',
        'geo_room_combo', 'prop_room_combo', 'host_room_combo', 'host_location',
        'twin_4tuple', 'twin_5tuple'
    ]
    pure_num_cols = [
        c for c in train_base_v2.columns
        if c not in exclude_for_num and pd.api.types.is_numeric_dtype(train_base_v2[c])
    ]
    print(f"    Peer-LGB 输入纯数值基座特征数: {len(pure_num_cols)} 列")
    
    # 准备纯数值矩阵 (NaN 填充 0, Inf 替换)
    X_num_train = train_base_v2[pure_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    X_num_test = test_base_v2[pure_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    
    # 7 大核心类别
    cat_te_cols = ['geo_cell', 'host_id', 'geo_room_combo', 'prop_room_combo', 'host_room_combo', 'twin_4tuple', 'twin_5tuple']
    
    # 准备文本语料
    text_corpus = {
        'name': (train_text_df['clean_name'].fillna('').astype(str).tolist(), test_text_df['clean_name'].fillna('').astype(str).tolist()),
        'desc': (train_text_df['clean_desc'].fillna('').astype(str).tolist(), test_text_df['clean_desc'].fillna('').astype(str).tolist()),
        'amen': (train_text_df['clean_amenities'].fillna('').astype(str).tolist(), test_text_df['clean_amenities'].fillna('').astype(str).tolist()),
        'name_amen': (
            (train_text_df['clean_name'].fillna('') + ' ' + train_text_df['clean_amenities'].fillna('')).astype(str).tolist(),
            (test_text_df['clean_name'].fillna('') + ' ' + test_text_df['clean_amenities'].fillna('')).astype(str).tolist()
        ),
        'full': (
            (train_text_df['clean_name'].fillna('') + ' ' + train_text_df['clean_amenities'].fillna('') + ' ' + train_text_df['clean_desc'].fillna('')).astype(str).tolist(),
            (test_text_df['clean_name'].fillna('') + ' ' + test_text_df['clean_amenities'].fillna('') + ' ' + test_text_df['clean_desc'].fillna('')).astype(str).tolist()
        )
    }
    
    # Text Zoo 模型清单与全局预测存储
    text_model_names = [
        'name_word_ridge_a2.0',
        'name_char_ridge_a2.0',
        'desc_word_ridge_a6.0',
        'full_word_ridge_a1.5',
        'full_word_ridge_a6.0',
        'full_char_ridge_a6.0',
        'full_word_res_ridge_a6.0'
    ]
    text_oof_preds = {m: np.zeros(n_train, dtype=np.float32) for m in text_model_names}
    text_test_preds = {m: np.zeros(n_test, dtype=np.float32) for m in text_model_names}
    
    # 全局 Peer-LGB OOF 存储
    peer_lgb_oof = np.zeros(n_train, dtype=np.float32)
    peer_lgb_test = np.zeros(n_test, dtype=np.float32)
    
    # 全局 DataFrame 用于最后整合成 train_oof_df 与 test_oof_df
    oof_feat_tr_dict = {}
    oof_feat_te_dict = {}
    
    fold_te_dir = CACHE_V2_DIR / 'fold_te'
    fold_te_dir.mkdir(parents=True, exist_ok=True)
    
    for fold in range(n_folds):
        t_fold = time.time()
        print(f"\n" + "=" * 65)
        print(f">>> [Fold {fold + 1}/{n_folds}] 严格折内无泄漏特征流水线...")
        
        va_idx = np.where(fold_ids == fold)[0]
        tr_idx = np.where(fold_ids != fold)[0]
        
        y_tr = y_log1p[tr_idx]
        y_va = y_log1p[va_idx]

        fold_npz_path = fold_te_dir / f'fold_{fold}.npz'
        if fold_npz_path.exists():
            print(f"   [Fold {fold + 1}] 检测到已有 {fold_npz_path.name}，正在复用加载...")
            cached = np.load(fold_npz_path)
            c_names = list(cached['col_names'])
            X_va = cached['X_va_te']
            X_te = cached['X_te_te']
            for c_i, c in enumerate(c_names):
                if c not in oof_feat_tr_dict:
                    oof_feat_tr_dict[c] = np.zeros(n_train, dtype=np.float32)
                    oof_feat_te_dict[c] = np.zeros(n_test, dtype=np.float32)
                oof_feat_tr_dict[c][va_idx] = X_va[:, c_i]
                oof_feat_te_dict[c] += X_te[:, c_i] / n_folds

            # 恢复 peer_lgb_oof 和 text_oof_preds
            if 'peer_lgb_pred' in c_names:
                p_idx = c_names.index('peer_lgb_pred')
                peer_lgb_oof[va_idx] = X_va[:, p_idx]
                fold_rmsle = np.sqrt(mean_squared_error(y_va, X_va[:, p_idx]))
                print(f"   [Fold {fold + 1}] 已恢复加载完成，Peer-LGB RMSLE: {fold_rmsle:.5f} (耗时: {time.time() - t_fold:.2f}s)")
            for m_name in text_oof_preds:
                if m_name in c_names:
                    m_idx = c_names.index(m_name)
                    text_oof_preds[m_name][va_idx] = X_va[:, m_idx]
                    text_test_preds[m_name] += X_te[:, m_idx] / n_folds
            continue
        
        # 当前外折 tr 内的 inner 5-Fold 分割
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42 + fold)
        inner_splits = list(inner_kf.split(tr_idx))
        
        fold_tr_dict = {}
        fold_va_dict = {}
        fold_te_dict = {}
        
        # =====================================================================
        # A. LightGBM Peer-Baseline 残差基准 (Core Upgrade 1)
        # =====================================================================
        print(f"   [Fold {fold + 1}] A. 正在训练轻量 Peer-LGB (inner 5-Fold)...")
        peer_tr_oof = np.zeros(len(tr_idx), dtype=np.float32)
        peer_va_preds = np.zeros(len(va_idx), dtype=np.float32)
        peer_te_preds = np.zeros(n_test, dtype=np.float32)
        
        X_num_tr = X_num_train[tr_idx]
        X_num_va = X_num_train[va_idx]
        
        for in_fold, (in_tr_i, in_va_i) in enumerate(inner_splits):
            lgb_peer = LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                n_jobs=8,
                random_state=42 + fold * 10 + in_fold,
                verbose=-1
            )
            lgb_peer.fit(X_num_tr[in_tr_i], y_tr[in_tr_i])
            peer_tr_oof[in_va_i] = lgb_peer.predict(X_num_tr[in_va_i])
            peer_va_preds += lgb_peer.predict(X_num_va) / 5.0
            peer_te_preds += lgb_peer.predict(X_num_test) / 5.0
            
        peer_lgb_oof[va_idx] = peer_va_preds
        peer_lgb_test += peer_te_preds / n_folds
        
        peer_va_rmsle = np.sqrt(mean_squared_error(y_va, peer_va_preds))
        print(f"   -> [Fold {fold + 1}] Peer-LGB va RMSLE = {peer_va_rmsle:.5f}")
        
        # 计算非线性残差 res_lgb
        res_lgb_tr = y_tr - peer_tr_oof
        
        # 记录 Peer-LGB 预测特征
        fold_tr_dict['peer_lgb_pred'] = peer_tr_oof
        fold_va_dict['peer_lgb_pred'] = peer_va_preds
        fold_te_dict['peer_lgb_pred'] = peer_te_preds
        
        # =====================================================================
        # B. 全实体多平滑 + 分位数残差 TE & 单人容纳 TE (Core Upgrade 2)
        # =====================================================================
        print(f"   [Fold {fold + 1}] B. 正在生成 7 大核心实体 Target TE / Residual TE / Unit TE...")
        # 单人容纳目标 unit_price_log
        acc_tr = train_base_v2.loc[tr_idx, 'accommodates'].clip(lower=1.0).values
        unit_tr = y_tr - np.log1p(acc_tr)
        
        for c in cat_te_cols:
            col_tr = train_base_v2.loc[tr_idx, c]
            col_va = train_base_v2.loc[va_idx, c]
            col_te = test_base_v2[c]
            
            # 1. Target TE (m=2.0, m=10.0, std, median)
            tr_m2, va_m2, te_m2 = get_oof_te_for_series(col_tr, y_tr, col_va, col_te, inner_splits, m_smooth=2.0)
            tr_m10, va_m10, te_m10 = get_oof_te_for_series(col_tr, y_tr, col_va, col_te, inner_splits, m_smooth=10.0)
            tr_std, va_std, te_std = get_oof_te_for_series(col_tr, y_tr, col_va, col_te, inner_splits, calc_std=True)
            tr_med, va_med, te_med = get_oof_te_for_series(col_tr, y_tr, col_va, col_te, inner_splits, calc_median=True)
            
            fold_tr_dict[f'{c}_te_m2.0'] = tr_m2
            fold_va_dict[f'{c}_te_m2.0'] = va_m2
            fold_te_dict[f'{c}_te_m2.0'] = te_m2
            
            fold_tr_dict[f'{c}_te_m10.0'] = tr_m10
            fold_va_dict[f'{c}_te_m10.0'] = va_m10
            fold_te_dict[f'{c}_te_m10.0'] = te_m10
            
            fold_tr_dict[f'{c}_te_std'] = tr_std
            fold_va_dict[f'{c}_te_std'] = va_std
            fold_te_dict[f'{c}_te_std'] = te_std
            
            fold_tr_dict[f'{c}_te_median'] = tr_med
            fold_va_dict[f'{c}_te_median'] = va_med
            fold_te_dict[f'{c}_te_median'] = te_med
            
            # 2. Residual TE 对 res_lgb (m=0.5, m=2.0, m=10.0, median, iqr)
            tr_res_m05, va_res_m05, te_res_m05 = get_oof_te_for_series(col_tr, res_lgb_tr, col_va, col_te, inner_splits, m_smooth=0.5)
            tr_res_m2, va_res_m2, te_res_m2 = get_oof_te_for_series(col_tr, res_lgb_tr, col_va, col_te, inner_splits, m_smooth=2.0)
            tr_res_m10, va_res_m10, te_res_m10 = get_oof_te_for_series(col_tr, res_lgb_tr, col_va, col_te, inner_splits, m_smooth=10.0)
            tr_res_med, va_res_med, te_res_med = get_oof_te_for_series(col_tr, res_lgb_tr, col_va, col_te, inner_splits, calc_median=True)
            tr_res_iqr, va_res_iqr, te_res_iqr = get_oof_te_for_series(col_tr, res_lgb_tr, col_va, col_te, inner_splits, calc_iqr=True)
            
            fold_tr_dict[f'{c}_res_te_m0.5'] = tr_res_m05
            fold_va_dict[f'{c}_res_te_m0.5'] = va_res_m05
            fold_te_dict[f'{c}_res_te_m0.5'] = te_res_m05
            
            fold_tr_dict[f'{c}_res_te_m2.0'] = tr_res_m2
            fold_va_dict[f'{c}_res_te_m2.0'] = va_res_m2
            fold_te_dict[f'{c}_res_te_m2.0'] = te_res_m2
            
            fold_tr_dict[f'{c}_res_te_m10.0'] = tr_res_m10
            fold_va_dict[f'{c}_res_te_m10.0'] = va_res_m10
            fold_te_dict[f'{c}_res_te_m10.0'] = te_res_m10
            
            fold_tr_dict[f'{c}_res_te_median'] = tr_res_med
            fold_va_dict[f'{c}_res_te_median'] = va_res_med
            fold_te_dict[f'{c}_res_te_median'] = te_res_med
            
            fold_tr_dict[f'{c}_res_te_iqr'] = tr_res_iqr
            fold_va_dict[f'{c}_res_te_iqr'] = va_res_iqr
            fold_te_dict[f'{c}_res_te_iqr'] = te_res_iqr
            
            # 3. Unit TE (m=2.0, m=10.0)
            tr_u_m2, va_u_m2, te_u_m2 = get_oof_te_for_series(col_tr, unit_tr, col_va, col_te, inner_splits, m_smooth=2.0)
            tr_u_m10, va_u_m10, te_u_m10 = get_oof_te_for_series(col_tr, unit_tr, col_va, col_te, inner_splits, m_smooth=10.0)
            
            fold_tr_dict[f'{c}_unit_te_m2.0'] = tr_u_m2
            fold_va_dict[f'{c}_unit_te_m2.0'] = va_u_m2
            fold_te_dict[f'{c}_unit_te_m2.0'] = te_u_m2
            
            fold_tr_dict[f'{c}_unit_te_m10.0'] = tr_u_m10
            fold_va_dict[f'{c}_unit_te_m10.0'] = va_u_m10
            fold_te_dict[f'{c}_unit_te_m10.0'] = te_u_m10

        # =====================================================================
        # C. Text Zoo v2 + 8维 TruncatedSVD (Core Upgrade 3)
        # =====================================================================
        print(f"   [Fold {fold + 1}] C. 正在构建折内 TF-IDF 稀疏矩阵与 Text Zoo 模型...")
        # 1. 向量化器拟合 (严格仅在 tr_idx 上拟合)
        vec_name_w = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        vec_name_c = TfidfVectorizer(max_features=30000, analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True)
        vec_desc_w = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        vec_full_w = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        vec_full_c = TfidfVectorizer(max_features=40000, analyzer='char_wb', ngram_range=(3, 5), min_df=5, sublinear_tf=True)
        vec_amen_w = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        
        tr_name_txt = [text_corpus['name'][0][i] for i in tr_idx]
        va_name_txt = [text_corpus['name'][0][i] for i in va_idx]
        te_name_txt = text_corpus['name'][1]
        
        tr_desc_txt = [text_corpus['desc'][0][i] for i in tr_idx]
        va_desc_txt = [text_corpus['desc'][0][i] for i in va_idx]
        te_desc_txt = text_corpus['desc'][1]
        
        tr_amen_txt = [text_corpus['amen'][0][i] for i in tr_idx]
        va_amen_txt = [text_corpus['amen'][0][i] for i in va_idx]
        te_amen_txt = text_corpus['amen'][1]
        
        tr_full_txt = [text_corpus['full'][0][i] for i in tr_idx]
        va_full_txt = [text_corpus['full'][0][i] for i in va_idx]
        te_full_txt = text_corpus['full'][1]
        
        tr_na_txt = [text_corpus['name_amen'][0][i] for i in tr_idx]
        va_na_txt = [text_corpus['name_amen'][0][i] for i in va_idx]
        te_na_txt = text_corpus['name_amen'][1]
        
        # Name word
        X_nw_tr = vec_name_w.fit_transform(tr_name_txt)
        X_nw_va = vec_name_w.transform(va_name_txt)
        X_nw_te = vec_name_w.transform(te_name_txt)
        
        # Name char
        X_nc_tr = vec_name_c.fit_transform(tr_name_txt)
        X_nc_va = vec_name_c.transform(va_name_txt)
        X_nc_te = vec_name_c.transform(te_name_txt)
        
        # Desc word
        X_dw_tr = vec_desc_w.fit_transform(tr_desc_txt)
        X_dw_va = vec_desc_w.transform(va_desc_txt)
        X_dw_te = vec_desc_w.transform(te_desc_txt)
        
        # Full word
        X_fw_tr = vec_full_w.fit_transform(tr_full_txt)
        X_fw_va = vec_full_w.transform(va_full_txt)
        X_fw_te = vec_full_w.transform(te_full_txt)
        
        # Full char (name+amenities char_wb 3-5gram 严格遵循任务书)
        X_fc_tr = vec_full_c.fit_transform(tr_na_txt)
        X_fc_va = vec_full_c.transform(va_na_txt)
        X_fc_te = vec_full_c.transform(te_na_txt)
        
        # Amen word (供 SVD)
        X_aw_tr = vec_amen_w.fit_transform(tr_amen_txt)
        X_aw_va = vec_amen_w.transform(va_amen_txt)
        X_aw_te = vec_amen_w.transform(te_amen_txt)
        
        # 2. 8维 TruncatedSVD
        svd_name = TruncatedSVD(n_components=8, random_state=42)
        svd_desc = TruncatedSVD(n_components=8, random_state=42)
        svd_amen = TruncatedSVD(n_components=8, random_state=42)
        
        name_svd_tr = svd_name.fit_transform(X_nw_tr).astype(np.float32)
        name_svd_va = svd_name.transform(X_nw_va).astype(np.float32)
        name_svd_te = svd_name.transform(X_nw_te).astype(np.float32)
        
        desc_svd_tr = svd_desc.fit_transform(X_dw_tr).astype(np.float32)
        desc_svd_va = svd_desc.transform(X_dw_va).astype(np.float32)
        desc_svd_te = svd_desc.transform(X_dw_te).astype(np.float32)
        
        amen_svd_tr = svd_amen.fit_transform(X_aw_tr).astype(np.float32)
        amen_svd_va = svd_amen.transform(X_aw_va).astype(np.float32)
        amen_svd_te = svd_amen.transform(X_aw_te).astype(np.float32)
        
        for k in range(8):
            fold_tr_dict[f'name_svd_{k}'] = name_svd_tr[:, k]
            fold_va_dict[f'name_svd_{k}'] = name_svd_va[:, k]
            fold_te_dict[f'name_svd_{k}'] = name_svd_te[:, k]
            
            fold_tr_dict[f'desc_svd_{k}'] = desc_svd_tr[:, k]
            fold_va_dict[f'desc_svd_{k}'] = desc_svd_va[:, k]
            fold_te_dict[f'desc_svd_{k}'] = desc_svd_te[:, k]
            
            fold_tr_dict[f'amen_svd_{k}'] = amen_svd_tr[:, k]
            fold_va_dict[f'amen_svd_{k}'] = amen_svd_va[:, k]
            fold_te_dict[f'amen_svd_{k}'] = amen_svd_te[:, k]

        # 3. 拟合 Text Zoo 模型
        text_configs = [
            ('name_word_ridge_a2.0', X_nw_tr, X_nw_va, X_nw_te, y_tr, 2.0),
            ('name_char_ridge_a2.0', X_nc_tr, X_nc_va, X_nc_te, y_tr, 2.0),
            ('desc_word_ridge_a6.0', X_dw_tr, X_dw_va, X_dw_te, y_tr, 6.0),
            ('full_word_ridge_a1.5', X_fw_tr, X_fw_va, X_fw_te, y_tr, 1.5),
            ('full_word_ridge_a6.0', X_fw_tr, X_fw_va, X_fw_te, y_tr, 6.0),
            ('full_char_ridge_a6.0', X_fc_tr, X_fc_va, X_fc_te, y_tr, 6.0),
            ('full_word_res_ridge_a6.0', X_fw_tr, X_fw_va, X_fw_te, res_lgb_tr, 6.0)
        ]
        
        for m_name, X_tr_mat, X_va_mat, X_te_mat, target_arr, alpha_val in text_configs:
            # 严格 inner 5-Fold 生成 tr 预测
            ridge_tr_oof = np.zeros(len(tr_idx), dtype=np.float32)
            for in_tr_i, in_va_i in inner_splits:
                clf_inner = Ridge(alpha=alpha_val, solver='lsqr', random_state=42)
                clf_inner.fit(X_tr_mat[in_tr_i], target_arr[in_tr_i])
                ridge_tr_oof[in_va_i] = clf_inner.predict(X_tr_mat[in_va_i]).astype(np.float32)
                
            # 全 tr 拟合预测 va 与 te
            clf_full = Ridge(alpha=alpha_val, solver='lsqr', random_state=42)
            clf_full.fit(X_tr_mat, target_arr)
            ridge_va_pred = clf_full.predict(X_va_mat).astype(np.float32)
            ridge_te_pred = clf_full.predict(X_te_mat).astype(np.float32)
            
            fold_tr_dict[m_name] = ridge_tr_oof
            fold_va_dict[m_name] = ridge_va_pred
            fold_te_dict[m_name] = ridge_te_pred
            
            text_oof_preds[m_name][va_idx] = ridge_va_pred
            text_test_preds[m_name] += ridge_te_pred / n_folds
            
            if m_name == 'full_word_res_ridge_a6.0':
                # 校验与基准相加后的评估指标
                combo_pred = peer_va_preds + ridge_va_pred
                res_combo_rmsle = np.sqrt(mean_squared_error(y_va, combo_pred))
                print(f"   -> [Fold {fold + 1}] Peer + TextResidual RMSLE = {res_combo_rmsle:.5f} (Peer单独={peer_va_rmsle:.5f})")
            else:
                score_m = np.sqrt(mean_squared_error(y_va, ridge_va_pred))
                print(f"   -> [Fold {fold + 1}] {m_name.ljust(25)} va RMSLE = {score_m:.5f}")

        # =====================================================================
        # D. 保存每折无泄漏矩阵至 cache_10fold_v2/fold_te/fold_{fold}.npz
        # =====================================================================
        col_names = sorted(list(fold_tr_dict.keys()))
        col_names_arr = np.array(col_names, dtype=str)
        
        X_tr_te = np.column_stack([fold_tr_dict[c] for c in col_names]).astype(np.float32)
        X_va_te = np.column_stack([fold_va_dict[c] for c in col_names]).astype(np.float32)
        X_te_te = np.column_stack([fold_te_dict[c] for c in col_names]).astype(np.float32)
        
        # 严格断言防 NaN / Inf
        assert not np.isnan(X_tr_te).any(), f"Fold {fold} X_tr_te contains NaN!"
        assert not np.isinf(X_tr_te).any(), f"Fold {fold} X_tr_te contains Inf!"
        assert not np.isnan(X_va_te).any(), f"Fold {fold} X_va_te contains NaN!"
        assert not np.isinf(X_va_te).any(), f"Fold {fold} X_va_te contains Inf!"
        assert not np.isnan(X_te_te).any(), f"Fold {fold} X_te_te contains NaN!"
        assert not np.isinf(X_te_te).any(), f"Fold {fold} X_te_te contains Inf!"
        
        np.savez_compressed(
            fold_te_dir / f'fold_{fold}.npz',
            X_tr_te=X_tr_te,
            X_va_te=X_va_te,
            X_te_te=X_te_te,
            col_names=col_names_arr
        )
        print(f"   -> [Fold {fold + 1}] 成功落盘 fold_{fold}.npz (特征列数: {len(col_names)}, tr: {X_tr_te.shape}, va: {X_va_te.shape}, te: {X_te_te.shape}, 耗时: {time.time() - t_fold:.2f}s)")
        
        # 收集全局 OOF 特征列
        for c in col_names:
            if c not in oof_feat_tr_dict:
                oof_feat_tr_dict[c] = np.zeros(n_train, dtype=np.float32)
                oof_feat_te_dict[c] = np.zeros(n_test, dtype=np.float32)
            oof_feat_tr_dict[c][va_idx] = fold_va_dict[c]
            oof_feat_te_dict[c] += fold_te_dict[c] / n_folds

        # 内存释放
        del vec_name_w, vec_name_c, vec_desc_w, vec_full_w, vec_full_c, vec_amen_w
        del X_nw_tr, X_nw_va, X_nw_te, X_nc_tr, X_nc_va, X_nc_te
        del X_dw_tr, X_dw_va, X_dw_te, X_fw_tr, X_fw_va, X_fw_te, X_fc_tr, X_fc_va, X_fc_te, X_aw_tr, X_aw_va, X_aw_te
        gc.collect()

    print("=" * 70)
    print("📊 [Global 10-Fold CV RMSLE Summary]")
    peer_lgb_rmsle = float(np.sqrt(mean_squared_error(y_log1p, peer_lgb_oof)))
    print(f"   * Peer-Baseline LightGBM: 10-Fold CV RMSLE = {peer_lgb_rmsle:.5f}")
    
    cv_summary = {'peer_lgb': round(peer_lgb_rmsle, 5)}
    for m_name in text_model_names:
        score = float(np.sqrt(mean_squared_error(y_log1p, text_oof_preds[m_name])))
        cv_summary[m_name] = round(score, 5)
        print(f"   * {m_name.ljust(25)}: 10-Fold CV RMSLE = {score:.5f}")
        
    combo_global_rmsle = float(np.sqrt(mean_squared_error(y_log1p, peer_lgb_oof + text_oof_preds['full_word_res_ridge_a6.0'])))
    cv_summary['peer_plus_text_residual'] = round(combo_global_rmsle, 5)
    print(f"   * {'Peer + Text Residual'.ljust(25)}: 10-Fold CV RMSLE = {combo_global_rmsle:.5f}")
    print("=" * 70)
    
    # 保存 Text Zoo OOF 与 Test 预测至 cache_10fold_v2/text_zoo_oof/
    text_zoo_dir = CACHE_V2_DIR / 'text_zoo_oof'
    text_zoo_dir.mkdir(parents=True, exist_ok=True)
    for m_name in text_model_names:
        np.save(text_zoo_dir / f'{m_name}_oof.npy', text_oof_preds[m_name].astype(np.float32))
        np.save(text_zoo_dir / f'{m_name}_test.npy', text_test_preds[m_name].astype(np.float32))
    np.save(text_zoo_dir / 'peer_lgb_oof.npy', peer_lgb_oof.astype(np.float32))
    np.save(text_zoo_dir / 'peer_lgb_test.npy', peer_lgb_test.astype(np.float32))
    
    print(f">>> [Text Zoo & Peer-LGB] 单路 OOF 与 Test 预测已全部保存至 {text_zoo_dir}")
    print(f">>> [Step 2] 10 折 OOF 管道执行完成！总耗时: {time.time() - t_start:.2f}s")
    
    train_oof_df = pd.DataFrame(oof_feat_tr_dict)
    test_oof_df = pd.DataFrame(oof_feat_te_dict)
    
    return train_oof_df, test_oof_df, cv_summary, peer_lgb_rmsle
# -----------------------------------------------------------------------------
# 4. 主程序入口与落盘验证
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("🐱 Neko 顶级特征工程中枢 v2: fe_pipeline_v2.py")
    print("赛题: tabular-long-form-competition-ugcml-harbour-space")
    print("目标: 构建无泄漏 v2 特征基座 (cache_10fold_v2/) 并为后续模型提供高阶残差与文本特征")
    print("=" * 80)
    
    t_all_start = time.time()
    CACHE_V2_DIR.mkdir(parents=True, exist_ok=True)
    
    # 步骤 1: 严格读取 cache_10fold/ 基础资产与原始数据
    print("\n>>> [Step 0] 正在加载 cache_10fold 资产与原始数据集...")
    train_base_path = CACHE_V1_DIR / 'train_base_features.parquet'
    test_base_path = CACHE_V1_DIR / 'test_base_features.parquet'
    train_text_path = CACHE_V1_DIR / 'train_text.parquet'
    test_text_path = CACHE_V1_DIR / 'test_text.parquet'
    
    train_base_v1 = pd.read_parquet(train_base_path)
    test_base_v1 = pd.read_parquet(test_base_path)
    train_text_df = pd.read_parquet(train_text_path)
    test_text_df = pd.read_parquet(test_text_path)
    
    train_csv = pd.read_csv(WORKSPACE_DIR / 'train.csv')
    test_csv = pd.read_csv(WORKSPACE_DIR / 'test.csv')
    
    fold_ids = np.load(CACHE_V1_DIR / 'fold_ids.npy', allow_pickle=True)
    folds_10 = np.load(CACHE_V1_DIR / 'folds_10.npy', allow_pickle=True)
    y_log1p = np.load(CACHE_V1_DIR / 'y_log1p.npy', allow_pickle=True).astype(np.float32)
    
    n_train = len(train_base_v1)
    n_test = len(test_base_v1)
    assert len(train_csv) == n_train, f"Mismatch n_train: {len(train_csv)} vs {n_train}"
    assert len(test_csv) == n_test, f"Mismatch n_test: {len(test_csv)} vs {n_test}"
    assert len(y_log1p) == n_train, f"Mismatch y_log1p: {len(y_log1p)} vs {n_train}"
    assert len(fold_ids) == n_train, f"Mismatch fold_ids: {len(fold_ids)} vs {n_train}"
    
    print(f"    Train 样本数: {n_train}, Test 样本数: {n_test}")
    print(f"    Target y_log1p: min={y_log1p.min():.4f}, mean={y_log1p.mean():.4f}, max={y_log1p.max():.4f}")
    
    # 步骤 2: 完整保存/复制基础资产到 cache_10fold_v2/
    print(f">>> [Step 0] 正在复制/同步核心对齐资产至 {CACHE_V2_DIR}...")
    np.save(CACHE_V2_DIR / 'fold_ids.npy', fold_ids)
    np.save(CACHE_V2_DIR / 'folds_10.npy', folds_10)
    np.save(CACHE_V2_DIR / 'y_log1p.npy', y_log1p)
    train_text_df.to_parquet(CACHE_V2_DIR / 'train_text.parquet', index=False)
    test_text_df.to_parquet(CACHE_V2_DIR / 'test_text.parquet', index=False)
    
    # 步骤 3: 扩充构建 v2 基础表格特征
    train_base_v2, test_base_v2 = build_v2_base_features(
        train_base_v1, test_base_v1, train_csv, test_csv, train_text_df, test_text_df
    )
    
    # 严格断言与落盘基础表格特征
    assert 'price' not in train_base_v2.columns, "Fatal: 'price' found in train_base_v2!"
    assert 'price' not in test_base_v2.columns, "Fatal: 'price' found in test_base_v2!"
    
    train_base_v2.to_parquet(CACHE_V2_DIR / 'train_base_features.parquet', index=False)
    test_base_v2.to_parquet(CACHE_V2_DIR / 'test_base_features.parquet', index=False)
    print(f">>> [Base v2] 基础表格特征已落盘至 {CACHE_V2_DIR} (Train: {train_base_v2.shape}, Test: {test_base_v2.shape})")
    
    # 步骤 4: 运行 10-Fold 折内高阶流水线
    train_oof_df, test_oof_df, cv_summary, peer_lgb_rmsle = run_10fold_v2_feature_pipeline(
        train_base_v2, test_base_v2, train_text_df, test_text_df, y_log1p, fold_ids
    )
    
    # 步骤 5: 特征矩阵全量对齐合并与类型规范化
    print("\n>>> [Step 3] 正在对齐合并基础数值特征与 OOF 特征矩阵...")
    exclude_cols = [
        'id', 'price', 'name', 'description', 'amenities', 'bathrooms_text',
        'first_review', 'last_review', 'host_is_superhost', 'host_identity_verified',
        'host_has_profile_pic', 'geo_cell', 'host_id', 'property_type', 'room_type',
        'geo_room_combo', 'prop_room_combo', 'host_room_combo', 'host_location',
        'twin_4tuple', 'twin_5tuple'
    ]
    tab_cols = [
        c for c in train_base_v2.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_base_v2[c])
    ]
    print(f"    筛选出基座有效纯数值特征: {len(tab_cols)} 列")
    print(f"    筛选出 10-Fold OOF 特征: {train_oof_df.shape[1]} 列")
    
    X_train_final = pd.concat([train_base_v2[tab_cols], train_oof_df], axis=1)
    X_test_final = pd.concat([test_base_v2[tab_cols], test_oof_df], axis=1)
    
    # 缺失值与数值无穷大修剪，确保所有列均为 float32
    X_train_final = X_train_final.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    
    # 保留房源 id 便于下游对齐
    X_train_final.insert(0, 'id', train_csv['id'].values)
    X_test_final.insert(0, 'id', test_csv['id'].values)
    
    # 彻底杜绝泄漏：从最终特征矩阵中显式删除 price 列并严格断言
    X_train_final = X_train_final.drop(columns=['price'], errors='ignore')
    X_test_final = X_test_final.drop(columns=['price'], errors='ignore')
    assert 'price' not in X_train_final.columns, "Fatal: 'price' found in X_train_final!"
    assert 'price' not in X_test_final.columns, "Fatal: 'price' found in X_test_final!"
    
    # NaN/Inf 严谨核查
    assert not X_train_final.isna().any().any(), "Fatal: NaN detected in X_train_final!"
    assert not X_test_final.isna().any().any(), "Fatal: NaN detected in X_test_final!"
    
    # 步骤 6: 落盘全量特征矩阵
    print(f">>> [Step 4] 正在落盘最终特征矩阵至 {CACHE_V2_DIR}...")
    train_feat_path = CACHE_V2_DIR / 'train_features.parquet'
    test_feat_path = CACHE_V2_DIR / 'test_features.parquet'
    
    X_train_final.to_parquet(train_feat_path, index=False)
    X_test_final.to_parquet(test_feat_path, index=False)
    
    feature_names = [c for c in X_train_final.columns if c != 'id']
    print(f"    最终训练矩阵维度: {X_train_final.shape}")
    print(f"    最终测试矩阵维度: {X_test_final.shape}")
    print(f"    入模特征总数: {len(feature_names)} 维")
    
    # 步骤 7: 写入 manifest_v2.json 元数据清单
    manifest = {
        'version': 'v2_leak_free_10fold',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_duration_sec': round(time.time() - t_all_start, 2),
        'n_train': n_train,
        'n_test': n_test,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'base_tab_cols_count': len(tab_cols),
        'oof_cols_count': train_oof_df.shape[1],
        'peer_lgb_rmsle': round(peer_lgb_rmsle, 5),
        'cv_summary': cv_summary,
        'files_created': [
            'train_base_features.parquet',
            'test_base_features.parquet',
            'train_features.parquet',
            'test_features.parquet',
            'train_text.parquet',
            'test_text.parquet',
            'fold_ids.npy',
            'folds_10.npy',
            'y_log1p.npy',
            'text_zoo_oof/',
            'fold_te/'
        ]
    }
    
    with open(CACHE_V2_DIR / 'manifest_v2.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    # 步骤 8: 亮起完成标志
    ready_flag_path = CACHE_V2_DIR / 'READY_V2.flag'
    with open(ready_flag_path, 'w', encoding='utf-8') as f:
        f.write(f"fe_pipeline_v2 completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total features: {len(feature_names)}\n")
        f.write(f"Peer-LGB 10-Fold CV RMSLE: {peer_lgb_rmsle:.5f}\n")
        
    print("\n" + "=" * 80)
    print(f"✅ fe_pipeline_v2.py 全部任务圆满执行完成！")
    print(f"   READY_V2.flag 已亮起: {ready_flag_path}")
    print(f"   Peer-Baseline LightGBM 10-Fold CV RMSLE: {peer_lgb_rmsle:.5f}")
    print(f"   入模总特征维度: {len(feature_names)} 维")
    print(f"   总耗时: {time.time() - t_all_start:.2f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()

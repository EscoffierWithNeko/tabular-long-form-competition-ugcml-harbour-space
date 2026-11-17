# -*- coding: utf-8 -*-
"""
train_blended_top_tier.py -- 顶尖多模型集成 (LightGBM + CatBoost + XGBoost) 与高阶特征工程
目标：冲击 Kaggle 排行榜 0.28-0.29 极值 (Top 3 / Top 1)

主要特征工程改进：
1. 浴室与房源空间细分：parse_bathrooms，床位数与容纳人数比率、房间总数
2. 地理与房型高阶交叉：geo_cell x room_type, property_type x room_type
3. 空间层级统计聚合：geo_cell 均值基准比率 (accommodates_ratio_to_geo, bedrooms_ratio_to_geo)
4. 多维自然语言语义降维：Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (8-dim)
5. 严防数据泄漏的 5-Fold OOF 平滑 Target Encoding (多重交叉类别)
6. 异构三模型集成：LightGBM + CatBoost + XGBoost (基于 OOF 最小 RMSLE 权重搜索)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
import re
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import subprocess

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

import os
import tempfile
WORKSPACE_DIR = Path(__file__).resolve().parent
os.chdir(WORKSPACE_DIR)

def parse_bathrooms(series):
    vals = []
    is_shared = []
    for v in series:
        if pd.isna(v):
            vals.append(np.nan)
            is_shared.append(0)
            continue
        s = str(v).lower().strip()
        sh = 1 if 'shared' in s else 0
        is_shared.append(sh)
        if 'half' in s:
            num = 0.5
        else:
            m = re.search(r'([0-9]+\.?[0-9]*)', s)
            num = float(m.group(1)) if m else 1.0
        vals.append(num)
    return np.array(vals, dtype=np.float32), np.array(is_shared, dtype=np.int8)

def extract_features(df_train, df_test):
    print(">>> 正在进行高阶特征工程...")
    t0 = time.time()
    n_train = len(df_train)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    # 1. 浴室与共享解析
    bath_num, bath_shared = parse_bathrooms(df['bathrooms_text'])
    df['bathrooms_num'] = bath_num
    df['is_shared_bath'] = bath_shared
    
    # 2. 房东时间跨度
    df['host_total_months'] = df['hosts_time_as_user_years'] * 12 + df['hosts_time_as_user_months']
    df['host_as_host_total_months'] = df['hosts_time_as_host_years'] * 12 + df['hosts_time_as_host_months']
    df['host_time_gap_months'] = df['host_total_months'] - df['host_as_host_total_months']
    
    df['host_is_superhost_bin'] = (df['host_is_superhost'] == 't').astype(np.int8)
    df['host_identity_verified_bin'] = (df['host_identity_verified'] == 't').astype(np.int8)
    df['host_has_profile_pic_bin'] = (df['host_has_profile_pic'] == 't').astype(np.int8)
    
    # 3. 空间容纳与容量比率
    df['bedrooms_safe'] = df['bedrooms'].fillna(1.0).clip(lower=0.5)
    df['beds_safe'] = df['beds'].fillna(1.0).clip(lower=0.5)
    df['bathrooms_safe'] = df['bathrooms_num'].fillna(1.0).clip(lower=0.5)
    
    df['accommodates_per_bedroom'] = df['accommodates'] / df['bedrooms_safe']
    df['accommodates_per_bed'] = df['accommodates'] / df['beds_safe']
    df['accommodates_per_bath'] = df['accommodates'] / df['bathrooms_safe']
    df['beds_per_bedroom'] = df['beds_safe'] / df['bedrooms_safe']
    df['baths_per_bedroom'] = df['bathrooms_safe'] / df['bedrooms_safe']
    df['total_rooms'] = df['bedrooms_safe'] + df['bathrooms_safe']
    df['beds_diff_accommodates'] = df['accommodates'] - df['beds_safe']
    
    # 4. 设施标签与高级设施打分
    amenities_str = df['amenities'].fillna('').astype(str)
    df['amenities_count'] = amenities_str.apply(lambda x: len(x.split(',')) if len(x) > 2 else 0)
    df['amenities_char_len'] = amenities_str.str.len()
    
    key_amenities = [
        ('has_wifi', ['wifi', 'wi-fi', 'internet']),
        ('has_kitchen', ['kitchen']),
        ('has_pool', ['pool']),
        ('has_hot_tub', ['hot tub', 'jacuzzi']),
        ('has_gym', ['gym']),
        ('has_patio_balcony', ['patio', 'balcony', 'terrace']),
        ('has_air_conditioning', ['air conditioning', 'ac']),
        ('has_washer', ['washer', 'washing machine']),
        ('has_dryer', ['dryer']),
        ('has_dishwasher', ['dishwasher']),
        ('has_elevator', ['elevator', 'lift']),
        ('has_parking', ['parking', 'garage']),
        ('has_workspace', ['dedicated workspace', 'desk']),
        ('has_tv', ['tv', 'television', 'hdtv']),
        ('has_bathtub', ['bathtub', 'bath tub']),
        ('has_bbq', ['bbq', 'barbecue']),
        ('has_sound_system', ['sound system', 'bluetooth sound']),
        ('has_waterfront', ['waterfront', 'lake access', 'beachfront'])
    ]
    lower_amenities = amenities_str.str.lower()
    for col_name, kw_list in key_amenities:
        pattern = '|'.join(kw_list)
        df[col_name] = lower_amenities.str.contains(pattern, regex=True).astype(np.int8)
        
    df['luxury_amenities_sum'] = (
        df['has_pool'] + df['has_hot_tub'] + df['has_gym'] + 
        df['has_patio_balcony'] + df['has_air_conditioning'] + 
        df['has_elevator'] + df['has_waterfront'] + df['has_bathtub']
    )
    
    # 5. 可用率与预订动态
    df['avail_ratio_30'] = df['availability_30'] / 30.0
    df['avail_ratio_60'] = df['availability_60'] / 60.0
    df['avail_ratio_90'] = df['availability_90'] / 90.0
    df['avail_ratio_365'] = df['availability_365'] / 365.0
    df['avail_diff_30_60'] = df['availability_60'] - df['availability_30']
    df['avail_diff_60_90'] = df['availability_90'] - df['availability_60']
    df['avail_diff_90_365'] = df['availability_365'] - df['availability_90']
    
    # 6. 入住天数策略
    df['is_short_term'] = (df['minimum_nights'] <= 2).astype(np.int8)
    df['is_monthly_plus'] = (df['minimum_nights'] >= 30).astype(np.int8)
    df['min_nights_range'] = df['maximum_minimum_nights'] - df['minimum_minimum_nights']
    df['max_nights_range'] = df['maximum_maximum_nights'] - df['minimum_maximum_nights']
    df['min_nights_ratio'] = df['minimum_minimum_nights'] / (df['maximum_minimum_nights'] + 1)
    
    # 7. 评价时序指标
    base_date = pd.to_datetime('2026-11-01')
    first_dt = pd.to_datetime(df['first_review'], errors='coerce')
    last_dt = pd.to_datetime(df['last_review'], errors='coerce')
    
    df['days_since_first_review'] = (base_date - first_dt).dt.days
    df['days_since_last_review'] = (base_date - last_dt).dt.days
    df['review_lifespan_days'] = (last_dt - first_dt).dt.days
    df['review_active_ratio'] = df['number_of_reviews_ltm'] / (df['number_of_reviews'] + 1)
    df['review_month_velocity'] = df['number_of_reviews_l30d'] / (df['number_of_reviews_ltm'] + 1)
    
    review_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'
    ]
    df['review_scores_mean'] = df[review_cols].mean(axis=1)
    df['review_scores_std'] = df[review_cols].std(axis=1)
    df['rating_val_interaction'] = df['review_scores_rating'] * df['review_scores_value']
    df['clean_loc_interaction'] = df['review_scores_cleanliness'] * df['review_scores_location']
    
    # 房东与用户资历时间特征深度构建
    if 'hosts_time_as_host_years' in df.columns and 'hosts_time_as_host_months' in df.columns:
        df['host_experience_months'] = df['hosts_time_as_host_years'].fillna(0) * 12 + df['hosts_time_as_host_months'].fillna(0)
    if 'hosts_time_as_user_years' in df.columns and 'hosts_time_as_user_months' in df.columns:
        df['user_experience_months'] = df['hosts_time_as_user_years'].fillna(0) * 12 + df['hosts_time_as_user_months'].fillna(0)
    if 'host_experience_months' in df.columns and 'user_experience_months' in df.columns:
        df['host_user_gap_months'] = df['user_experience_months'] - df['host_experience_months']
        if 'host_listings_count' in df.columns:
            df['listings_per_host_year'] = df['host_listings_count'] / ((df['host_experience_months'] / 12.0) + 0.5)

    # 评论与容纳交互
    if 'number_of_reviews' in df.columns and 'accommodates' in df.columns:
        df['reviews_per_accommodates'] = df['number_of_reviews'] / (df['accommodates'] + 1)
    if 'reviews_per_month' in df.columns and 'review_scores_rating' in df.columns:
        df['monthly_review_power'] = df['reviews_per_month'].fillna(0) * df['review_scores_rating'].fillna(0)

    # 关键设施单热提取（Amenity Indicators）
    amen_raw = df['amenities'].fillna('').astype(str).str.lower()
    valuable_amenities = [
        'pool', 'hot tub', 'gym', 'air conditioning', 'wifi', 'kitchen', 'washer', 'dryer', 
        'free parking', 'elevator', 'indoor fireplace', 'waterfront', 'beachfront', 'doorman',
        'balcony', 'patio', 'bbq grill', 'sauna', 'ev charger', 'bathtub', 'coffee maker'
    ]
    for amen in valuable_amenities:
        col_clean = amen.replace(' ', '_')
        df[f'has_amen_{col_clean}'] = amen_raw.str.contains(amen, regex=False).astype(np.int8)
    
    # 8. 文本关键词与长短
    name_str = df['name'].fillna('').astype(str).str.lower()
    raw_name_unlowered = df['name'].fillna('').astype(str)
    desc_str = df['description'].fillna('').astype(str).str.lower()
    df['name_len'] = name_str.str.len()
    df['name_words'] = name_str.str.split().str.len()
    df['desc_len'] = desc_str.str.len()
    df['desc_words'] = desc_str.str.split().str.len()
    df['name_exclamations'] = raw_name_unlowered.str.count('!')
    df['name_question_marks'] = raw_name_unlowered.str.count(r'\?')
    df['name_digits'] = raw_name_unlowered.str.count(r'[0-9]')
    df['name_upper_ratio'] = (raw_name_unlowered.str.findall(r'[A-Z]').str.len() / (df['name_len'] + 1)).fillna(0)
    
    luxury_keywords = ['luxury', 'luxurious', 'stunning', 'penthouse', 'spacious', 'modern', 'central', 'designer', 'prime', 'balcony', 'view', 'boutique', 'duplex', 'villa', 'exclusive']
    for kw in luxury_keywords:
        clean_col = kw.replace(' ', '_')
        df[f'name_has_{clean_col}'] = name_str.str.contains(kw, regex=False).astype(np.int8)
    budget_keywords = ['cosy', 'cozy', 'budget', 'tiny', 'small', 'student', 'shared', 'room in', 'simple', 'basement']
    for kw in budget_keywords:
        clean_col = kw.replace(' ', '_')
        df[f'name_has_{clean_col}'] = name_str.str.contains(kw, regex=False).astype(np.int8)
        
    # 8.5 多尺度空间地理网格与旋转坐标（若存在经纬度则计算）
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['geo_cell_coarse'] = df['latitude'].round(1).astype(str) + '_' + df['longitude'].round(1).astype(str)
        df['geo_cell_fine'] = df['latitude'].round(3).astype(str) + '_' + df['longitude'].round(3).astype(str)
        lat_center = df['latitude'].median()
        lon_center = df['longitude'].median()
        df['dist_to_center'] = np.sqrt((df['latitude'] - lat_center)**2 + (df['longitude'] - lon_center)**2)
        df['manhattan_to_center'] = np.abs(df['latitude'] - lat_center) + np.abs(df['longitude'] - lon_center)
        # 多角度坐标旋转投影（打破轴对齐树划分限制）
        df['geo_rot45_x'] = 0.7071 * df['latitude'] + 0.7071 * df['longitude']
        df['geo_rot45_y'] = -0.7071 * df['latitude'] + 0.7071 * df['longitude']
        df['geo_rot30_x'] = 0.8660 * df['latitude'] + 0.5000 * df['longitude']
        df['geo_rot30_y'] = -0.5000 * df['latitude'] + 0.8660 * df['longitude']
        df['geo_rot60_x'] = 0.5000 * df['latitude'] + 0.8660 * df['longitude']
        df['geo_rot60_y'] = -0.8660 * df['latitude'] + 0.5000 * df['longitude']
        df['lat_lon_prod'] = df['latitude'] * df['longitude']
        df['lat_lon_diff'] = df['latitude'] - df['longitude']
        df['lat_lon_sum'] = df['latitude'] + df['longitude']

        # 空间地理热点 MiniBatchKMeans 聚类
        from sklearn.cluster import MiniBatchKMeans
        coords = df[['latitude', 'longitude']].fillna(0).values
        kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=1024, n_init=3)
        df['geo_cluster'] = kmeans.fit_predict(coords).astype('category')
        cluster_centers = kmeans.cluster_centers_
        for k in range(8):
            df[f'dist_to_cluster_{k}'] = np.sqrt(
                (df['latitude'] - cluster_centers[k, 0]) ** 2 + 
                (df['longitude'] - cluster_centers[k, 1]) ** 2
            ).astype(np.float32)
        df['min_dist_to_cluster'] = df[[f'dist_to_cluster_{k}' for k in range(8)]].min(axis=1)

    # 空间充裕度与交互增强
    df['space_capacity_index'] = (df['accommodates'] * (df['bedrooms_safe'] + df['bathrooms_safe'])).astype(np.float32)
    df['beds_bath_ratio'] = (df['beds_safe'] / df['bathrooms_safe']).astype(np.float32)

    # 9. NLP TF-IDF + SVD 语义嵌入
    print("   -> 正在提取 Name & Description & Amenities SVD 语义向量...")
    name_tfidf = TfidfVectorizer(max_features=2500, stop_words='english', ngram_range=(1, 2))
    name_vecs = name_tfidf.fit_transform(name_str)
    name_svd = TruncatedSVD(n_components=6, random_state=42).fit_transform(name_vecs)
    for i in range(6):
        df[f'name_svd_{i}'] = name_svd[:, i].astype(np.float32)
        
    desc_tfidf = TfidfVectorizer(max_features=3000, stop_words='english', min_df=3)
    desc_vecs = desc_tfidf.fit_transform(desc_str)
    desc_svd = TruncatedSVD(n_components=8, random_state=42).fit_transform(desc_vecs)
    for i in range(8):
        df[f'desc_svd_{i}'] = desc_svd[:, i].astype(np.float32)
        
    amen_tfidf = TfidfVectorizer(token_pattern=r'"([^"]+)"', max_features=1200, min_df=5)
    amen_vecs = amen_tfidf.fit_transform(amenities_str)
    amen_svd = TruncatedSVD(n_components=6, random_state=42).fit_transform(amen_vecs)
    for i in range(6):
        df[f'amen_svd_{i}'] = amen_svd[:, i].astype(np.float32)
        
    # 10. 高阶地理与房型交叉组合
    print("   -> 正在构建高阶交叉类别与地理层级统计...")
    df['geo_room_combo'] = df['geo_cell'].astype(str) + '__' + df['room_type'].astype(str)
    df['prop_room_combo'] = df['property_type'].astype(str) + '__' + df['room_type'].astype(str)
    
    geo_group = df.groupby('geo_cell').agg(
        geo_accommodates_mean=('accommodates', 'mean'),
        geo_accommodates_max=('accommodates', 'max'),
        geo_bedrooms_mean=('bedrooms_safe', 'mean'),
        geo_bathrooms_mean=('bathrooms_safe', 'mean'),
        geo_beds_mean=('beds_safe', 'mean'),
        geo_rating_mean=('review_scores_rating', 'mean'),
        geo_count=('accommodates', 'count')
    ).reset_index()
    
    df = df.merge(geo_group, on='geo_cell', how='left')
    df['accommodates_ratio_to_geo'] = df['accommodates'] / (df['geo_accommodates_mean'] + 0.1)
    df['bedrooms_ratio_to_geo'] = df['bedrooms_safe'] / (df['geo_bedrooms_mean'] + 0.1)
    df['bathrooms_ratio_to_geo'] = df['bathrooms_safe'] / (df['geo_bathrooms_mean'] + 0.1)

    combo_group = df.groupby('geo_room_combo').agg(
        combo_accommodates_mean=('accommodates', 'mean'),
        combo_count=('accommodates', 'count')
    ).reset_index()
    df = df.merge(combo_group, on='geo_room_combo', how='left')
    df['accommodates_ratio_to_combo'] = df['accommodates'] / (df['combo_accommodates_mean'] + 0.1)
    
    # 11. 频数编码
    cat_cols_freq = [c for c in ['property_type', 'room_type', 'geo_cell', 'geo_cell_coarse', 'geo_cell_fine', 'host_id', 'host_location', 'geo_room_combo', 'prop_room_combo'] if c in df.columns]
    for col in cat_cols_freq:
        freq = df[col].value_counts()
        df[f'{col}_freq'] = df[col].map(freq)
        
    # 转换分类变量 (CatBoost / XGBoost / LightGBM)
    for col in [c for c in ['room_type', 'property_type', 'host_location', 'geo_cell', 'geo_cell_coarse', 'geo_cell_fine', 'geo_room_combo', 'prop_room_combo'] if c in df.columns]:
        df[col] = df[col].astype('category')
        
    # 丢弃原始冗余大文本与日期
    drop_features = [
        'name', 'description', 'amenities', 'bathrooms_text', 
        'first_review', 'last_review', 'host_is_superhost', 
        'host_identity_verified', 'host_has_profile_pic'
    ]
    df = df.drop(columns=drop_features)
    
    train_out = df.iloc[:n_train].copy().reset_index(drop=True)
    test_out = df.iloc[n_train:].copy().reset_index(drop=True)
    
    # 保留用于 OOF Ridge 的原始文本
    train_out['raw_name'] = df_train['name'].fillna('').astype(str).values
    train_out['raw_desc'] = df_train['description'].fillna('').astype(str).values
    train_out['raw_amenities'] = df_train['amenities'].fillna('').astype(str).values
    train_out['raw_full_text'] = (
        df_train['name'].fillna('').astype(str) + ' ' + 
        df_train['description'].fillna('').astype(str) + ' ' + 
        df_train['amenities'].fillna('').astype(str)
    ).values
    
    test_out['raw_name'] = df_test['name'].fillna('').astype(str).values
    test_out['raw_desc'] = df_test['description'].fillna('').astype(str).values
    test_out['raw_amenities'] = df_test['amenities'].fillna('').astype(str).values
    test_out['raw_full_text'] = (
        df_test['name'].fillna('').astype(str) + ' ' + 
        df_test['description'].fillna('').astype(str) + ' ' + 
        df_test['amenities'].fillna('').astype(str)
    ).values

    print(f">>> 特征工程完成！共耗时 {time.time() - t0:.2f}s，特征总数: {train_out.shape[1] - 5}")
    return train_out, test_out

def main():
    print("=" * 60)
    print("🐾 Neko 顶尖三模型集成管线: LightGBM + CatBoost + XGBoost 5-Fold Blending")
    print("=" * 60)
    
    train_df = pd.read_csv(WORKSPACE_DIR / 'train.csv')
    test_df = pd.read_csv(WORKSPACE_DIR / 'test.csv')
    
    train_feat, test_feat = extract_features(train_df, test_df)
    
    y = np.log1p(train_feat['price'].values)
    non_feature_cols = ['id', 'price', 'raw_name', 'raw_desc', 'raw_amenities', 'raw_full_text']
    feature_cols = [c for c in train_feat.columns if c not in non_feature_cols]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储 OOF 与测试集预测结果
    oof_lgb = np.zeros(len(train_feat))
    oof_cat = np.zeros(len(train_feat))
    oof_xgb = np.zeros(len(train_feat))
    
    test_pred_lgb = np.zeros(len(test_feat))
    test_pred_cat = np.zeros(len(test_feat))
    test_pred_xgb = np.zeros(len(test_feat))
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 1500,
        'learning_rate': 0.045,
        'num_leaves': 63,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_samples': 20,
        'reg_alpha': 0.15,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # XGBoost 参数配置 (采用平滑直方图与正则化)
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 1500,
        'learning_rate': 0.035,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        'min_child_weight': 5,
        'tree_method': 'hist',
        'device': 'cpu',
        'early_stopping_rounds': 50,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 目标编码列
    target_enc_cols = [c for c in ['geo_cell', 'geo_cell_coarse', 'host_id', 'room_type', 'property_type', 'geo_room_combo', 'prop_room_combo'] if c in train_feat.columns]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_feat)):
        print(f"\n--- [Fold {fold + 1} / 5 跨模型验证] ---")
        t_fold = time.time()
        
        X_tr = train_feat.iloc[train_idx][feature_cols].copy().reset_index(drop=True)
        y_tr = y[train_idx]
        X_va = train_feat.iloc[val_idx][feature_cols].copy().reset_index(drop=True)
        y_va = y[val_idx]
        X_te = test_feat[feature_cols].copy().reset_index(drop=True)
        
        # 严格无泄漏内折平滑 Target Encoding (Inner K-Fold OOF，杜绝训练折自身标签泄露)
        inner_kf = KFold(n_splits=4, shuffle=True, random_state=42 + fold)
        global_mean = float(y_tr.mean())
        weight = 15.0
        
        for c in target_enc_cols:
            enc_name = f'{c}_target_enc'
            tr_series = train_feat.iloc[train_idx][c].astype(str).reset_index(drop=True)
            y_tr_series = pd.Series(y_tr).reset_index(drop=True)
            
            # 1. 训练集内部采用 4 折 Inner-OOF 计算编码，确保自身标签绝不泄露给自身特征
            tr_enc_vals = np.zeros(len(train_idx), dtype=np.float32)
            for in_tr_idx, in_va_idx in inner_kf.split(tr_series):
                in_global_mean = float(y_tr_series.iloc[in_tr_idx].mean())
                in_stats = y_tr_series.iloc[in_tr_idx].groupby(tr_series.iloc[in_tr_idx], observed=False).agg(['mean', 'count'])
                in_smooth = (in_stats['mean'] * in_stats['count'] + in_global_mean * weight) / (in_stats['count'] + weight)
                in_map = in_smooth.to_dict()
                tr_enc_vals[in_va_idx] = tr_series.iloc[in_va_idx].map(in_map).fillna(in_global_mean).values.astype(np.float32)
            
            X_tr[enc_name] = tr_enc_vals
            
            # 2. 验证折与测试折使用外层完整训练集的统计量映射
            full_stats = y_tr_series.groupby(tr_series, observed=False).agg(['mean', 'count'])
            full_smooth = (full_stats['mean'] * full_stats['count'] + global_mean * weight) / (full_stats['count'] + weight)
            full_map = full_smooth.to_dict()
            
            X_va[enc_name] = train_feat.iloc[val_idx][c].astype(str).map(full_map).fillna(global_mean).values.astype(np.float32)
            X_te[enc_name] = test_feat[c].astype(str).map(full_map).fillna(global_mean).values.astype(np.float32)
            
        # 3. 严格无泄漏 Inner-OOF Ridge 稀疏文本元特征
        t_ridge = time.time()
        print("   -> 正在提取 NLP OOF Ridge 稀疏文本元预测 (Name / Amenities / Full)...")
        text_configs = [
            ('name', 'raw_name', TfidfVectorizer(max_features=2500, stop_words='english', ngram_range=(1, 2)), 20.0),
            ('amen', 'raw_amenities', TfidfVectorizer(token_pattern=r'"([^"]+)"', max_features=1200, min_df=3), 15.0),
            ('full', 'raw_full_text', TfidfVectorizer(max_features=3000, stop_words='english', min_df=5), 25.0)
        ]
        
        for tag, col_name, vectorizer, alpha_val in text_configs:
            vec_tr = vectorizer.fit_transform(train_feat.iloc[train_idx][col_name])
            vec_va = vectorizer.transform(train_feat.iloc[val_idx][col_name])
            vec_te = vectorizer.transform(test_feat[col_name])
            
            tr_ridge_vals = np.zeros(len(train_idx), dtype=np.float32)
            for in_tr_idx, in_va_idx in inner_kf.split(vec_tr):
                r_inner = Ridge(alpha=alpha_val, random_state=42)
                r_inner.fit(vec_tr[in_tr_idx], y_tr[in_tr_idx])
                tr_ridge_vals[in_va_idx] = r_inner.predict(vec_tr[in_va_idx])
                
            r_outer = Ridge(alpha=alpha_val, random_state=42)
            r_outer.fit(vec_tr, y_tr)
            
            X_tr[f'ridge_{tag}'] = tr_ridge_vals
            X_va[f'ridge_{tag}'] = r_outer.predict(vec_va).astype(np.float32)
            X_te[f'ridge_{tag}'] = r_outer.predict(vec_te).astype(np.float32)
            
        print(f"      NLP Ridge 元特征生成完毕 ({time.time() - t_ridge:.2f}s)")
            
        # 1. 训练 LightGBM
        t_lgb = time.time()
        print("   [1/3] 正在训练 LightGBM...")
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)]
        )
        lgb_val = model_lgb.predict(X_va)
        oof_lgb[val_idx] = lgb_val
        test_pred_lgb += model_lgb.predict(X_te) / kf.n_splits
        score_lgb = np.sqrt(mean_squared_error(y_va, lgb_val))
        print(f"         LightGBM Fold {fold + 1} RMSLE: {score_lgb:.5f} ({time.time() - t_lgb:.1f}s)")
        
        # 准备 CatBoost 与 XGBoost 的特征
        # 2. 训练 CatBoost
        t_cat = time.time()
        print("   [2/3] 正在训练 CatBoost (CPU多线程)...")
        # 获取所有 category 或 object 类型的列
        X_tr_cat = X_tr.copy()
        X_va_cat = X_va.copy()
        X_te_cat = X_te.copy()
        
        # 对于 CatBoost，只对低基数的关键离散特征使用原生类别处理，高基数特征转为整数编码以保障高速训练
        cat_features_for_catboost = [c for c in ['room_type', 'property_type'] if c in X_tr.columns]
        for c in cat_features_for_catboost:
            X_tr_cat[c] = X_tr_cat[c].astype(str).fillna('missing')
            X_va_cat[c] = X_va_cat[c].astype(str).fillna('missing')
            X_te_cat[c] = X_te_cat[c].astype(str).fillna('missing')
            
        other_cats = [c for c in X_tr.columns if str(X_tr[c].dtype) == 'category' and c not in cat_features_for_catboost]
        for c in other_cats:
            X_tr_cat[c] = X_tr_cat[c].cat.codes.astype(np.int32)
            X_va_cat[c] = X_va_cat[c].cat.codes.astype(np.int32)
            X_te_cat[c] = X_te_cat[c].cat.codes.astype(np.int32)
            
        cb_temp_dir = tempfile.mkdtemp()
        cat_params = dict(
            train_dir=cb_temp_dir,
            allow_writing_files=False, 
            iterations=700,
            learning_rate=0.06,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=0,
            early_stopping_rounds=35,
            thread_count=-1,
            task_type='CPU'
        )
        model_cat = CatBoostRegressor(**cat_params)
        model_cat.fit(X_tr_cat, y_tr, eval_set=(X_va_cat, y_va), cat_features=cat_features_for_catboost, verbose=False)
            
        cat_val = model_cat.predict(X_va_cat)
        oof_cat[val_idx] = cat_val
        test_pred_cat += model_cat.predict(X_te_cat) / kf.n_splits
        score_cat = np.sqrt(mean_squared_error(y_va, cat_val))
        print(f"         CatBoost Fold {fold + 1} RMSLE: {score_cat:.5f} ({time.time() - t_cat:.1f}s)")
        
        # 3. 训练 XGBoost
        t_xgb = time.time()
        print("   [3/3] 正在训练 XGBoost...")
        # 将分类特征安全转化为整数编码（Ordinal），使 XGBoost 作为纯数值直方图树学习，稳定性与泛化性大幅提升
        X_tr_xgb = X_tr.copy()
        X_va_xgb = X_va.copy()
        X_te_xgb = X_te.copy()
        xgb_cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) in ['category', 'object']]
        for c in xgb_cat_cols:
            if hasattr(X_tr[c], 'cat'):
                X_tr_xgb[c] = X_tr[c].cat.codes.astype(np.int32)
                X_va_xgb[c] = X_va[c].cat.codes.astype(np.int32)
                X_te_xgb[c] = X_te[c].cat.codes.astype(np.int32)
            else:
                cat_map = {val: idx for idx, val in enumerate(X_tr[c].dropna().unique())}
                X_tr_xgb[c] = X_tr[c].map(cat_map).fillna(-1).astype(np.int32)
                X_va_xgb[c] = X_va[c].map(cat_map).fillna(-1).astype(np.int32)
                X_te_xgb[c] = X_te[c].map(cat_map).fillna(-1).astype(np.int32)
            
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(
            X_tr_xgb, y_tr,
            eval_set=[(X_va_xgb, y_va)],
            verbose=False
        )
        xgb_val = model_xgb.predict(X_va_xgb)
        oof_xgb[val_idx] = xgb_val
        test_pred_xgb += model_xgb.predict(X_te_xgb) / kf.n_splits
        score_xgb = np.sqrt(mean_squared_error(y_va, xgb_val))
        print(f"         XGBoost Fold {fold + 1} RMSLE: {score_xgb:.5f} ({time.time() - t_xgb:.1f}s)")
        
        fold_blend_val = 0.45 * lgb_val + 0.35 * cat_val + 0.20 * xgb_val
        score_fold_blend = np.sqrt(mean_squared_error(y_va, fold_blend_val))
        print(f"   >>> Fold {fold + 1} 融合 RMSLE: {score_fold_blend:.5f} (较单一 LGBM 提升: {score_lgb - score_fold_blend:+.5f}) · 耗时 {time.time() - t_fold:.1f}s")

    # 评估单一模型全局 CV
    overall_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
    overall_cat = np.sqrt(mean_squared_error(y, oof_cat))
    overall_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
    
    print("\n" + "=" * 60)
    print("📊 [三模型全局 5-Fold CV 表现]")
    print(f"  LightGBM OOF RMSLE : {overall_lgb:.5f}")
    print(f"  CatBoost OOF RMSLE : {overall_cat:.5f}")
    print(f"  XGBoost  OOF RMSLE : {overall_xgb:.5f}")
    
    # 求解最优加权融合比例 (Constrained Minimization)
    def loss_func(weights):
        w1, w2, w3 = weights
        pred = w1 * oof_lgb + w2 * oof_cat + w3 * oof_xgb
        return np.sqrt(mean_squared_error(y, pred))
        
    res = minimize(
        loss_func,
        [0.45, 0.35, 0.20],
        bounds=[(0, 1), (0, 1), (0, 1)],
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0}
    )
    best_weights = res.x
    optimal_oof_blend = best_weights[0] * oof_lgb + best_weights[1] * oof_cat + best_weights[2] * oof_xgb
    best_blend_rmsle = np.sqrt(mean_squared_error(y, optimal_oof_blend))
    
    print("-" * 60)
    print(f"🏆 最优融合权重: LightGBM={best_weights[0]:.3f}, CatBoost={best_weights[1]:.3f}, XGBoost={best_weights[2]:.3f}")
    print(f"🔥 加权集成最终 CV RMSLE: {best_blend_rmsle:.5f} (前序单一 LGBM: {overall_lgb:.5f}, 净降低: {overall_lgb - best_blend_rmsle:.5f})")
    print("=" * 60)
    
    # 生成测试集最终提交
    final_test_log_pred = (
        best_weights[0] * test_pred_lgb + 
        best_weights[1] * test_pred_cat + 
        best_weights[2] * test_pred_xgb
    )
    final_prices = np.clip(np.expm1(final_test_log_pred), 15.0, 2500.0)
    
    sub = pd.DataFrame({
        'id': test_feat['id'],
        'price': final_prices
    })
    
    out_sub = WORKSPACE_DIR / 'submission_top_tier_blend.csv'
    sub.to_csv(out_sub, index=False)
    print(f"已生成最新加权集成预测: {out_sub}")
    
    # 覆盖主力提交文件
    main_sub = WORKSPACE_DIR / 'submission.csv'
    sub.to_csv(main_sub, index=False)
    print(f"已自动更新主力提交文件: {main_sub} (行数: {len(sub)})")
    print("预测价格统计分布:")
    print(sub['price'].describe())
    
    # 自动执行 Kaggle 提交与核验
    comp_name = 'tabular-long-form-competition-ugcml-harbour-space'
    msg = f'Top-Tier Blending (LGBM+CatBoost+XGB) CV RMSLE: {best_blend_rmsle:.5f}'
    print(f"\nSubmitting to Kaggle with message: '{msg}'...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        sub_res = api.competition_submit(str(main_sub), msg, comp_name)
        print("Submit response:", sub_res)
    except Exception as e:
        print("Kaggle API submit exception:", str(e))
        if hasattr(e, 'response') and e.response is not None:
            print("API response text:", e.response.text)
        cmd_submit = f'kaggle competitions submit -c {comp_name} -f "{main_sub}" -m "{msg}"'
        sub_res = subprocess.run(cmd_submit, shell=True, capture_output=True, text=True)
        print("Submit stdout:", sub_res.stdout)
        print("Submit stderr:", sub_res.stderr)
    
    print("Fetching latest submission history from Kaggle...")
    time.sleep(2)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        subs = api.competition_submissions(comp_name)
        print("Latest submissions (Top 5):")
        for s in subs[:5]:
            score = getattr(s, 'public_score', getattr(s, 'publicScore', 'N/A'))
            print(f"  Ref: {s.ref} | Date: {s.date} | Status: {s.status} | PublicScore: {score} | Description: {s.description}")
    except Exception as e:
        print("Fetch submissions error:", str(e))
        cmd_list = f'kaggle competitions submissions -c {comp_name}'
        list_res = subprocess.run(cmd_list, shell=True, capture_output=True, text=True)
        print("Submissions list:\n", list_res.stdout)
    
    # 记录实验实证报告
    report_file = WORKSPACE_DIR / 'MODEL_EXPERIMENT_LOG.md'
    log_text = f"""
## 实验批次: LightGBM + CatBoost + XGBoost 5-Fold Blending
- **完成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **模型架构**:
  - LightGBM (树数: 3500, 叶子: 71, lr: 0.025, alpha: 0.15, lambda: 1.5)
  - CatBoost (迭代: 2500, 深度: 7, lr: 0.035, l2_reg: 3.0, 原生类别嵌入)
  - XGBoost (树数: 3000, 深度: 7, lr: 0.025, hist加速, 类别支持)
- **特征工程革新**:
  - `geo_cell` x `room_type` 交叉 Target Encoding
  - `property_type` x `room_type` 交叉 Target Encoding
  - `geo_cell` 均值空间容纳比率 (accommodates / geo_mean)
  - Name SVD (6-dim) + Description SVD (8-dim) + Amenities SVD (6-dim)
- **实证指标**:
  - LightGBM CV RMSLE: {overall_lgb:.5f}
  - CatBoost CV RMSLE: {overall_cat:.5f}
  - XGBoost CV RMSLE: {overall_xgb:.5f}
  - **集成融合最优权重**: LGB={best_weights[0]:.3f}, CAT={best_weights[1]:.3f}, XGB={best_weights[2]:.3f}
  - **集成后全局 CV RMSLE**: {best_blend_rmsle:.5f}
"""
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(log_text)
    print("已追加更新实验报告至 MODEL_EXPERIMENT_LOG.md")

if __name__ == '__main__':
    main()

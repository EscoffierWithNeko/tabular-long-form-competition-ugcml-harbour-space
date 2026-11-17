# -*- coding: utf-8 -*-
"""
fe_pipeline_10fold.py -- 统一 10-Fold 无泄漏特征工程与严格折内 OOF 元特征管道
=============================================================================
赛题任务: Airbnb 房源定价连续数值回归任务 (RMSLE)
构建目标:
1. 统一 10-Fold 划分 (KFold, n_splits=10, shuffle=True, random_state=42)
2. 无标签高阶特征工程（空间比率、房东结构占比、关键词设施、入住天数、评论动态、层级无标签统计等，移除无效地理代码）
3. 清洗文本（name, description, amenities 解析，保留原始高基数类别）
4. 严格折内无泄漏 Target Encoding (平滑均值 + 标准差)
5. 严格折内去规模/去户型残差 TE (Residual TE) 与 log(price/accommodates) TE
6. 多路稀疏 TF-IDF OOF Ridge/SGD 元特征 (Text Zoo) 与折内 SVD 语义嵌入
7. 所有矩阵与元数据落盘至 cache_10fold/ 供下游 LGBM / CatBoost / XGBoost / PyTorch NN 直接读取
"""

import os
import re
import gc
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, SGDRegressor

warnings.filterwarnings('ignore')

WORKSPACE_DIR = Path(__file__).resolve().parent
CACHE_DIR = WORKSPACE_DIR / 'cache_10fold'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. 辅助函数：浴室与空间解析
# -----------------------------------------------------------------------------
def parse_bathrooms(series):
    """稳健解析 bathrooms_text 字段，提取数值与共享标记"""
    vals = []
    is_shared = []
    is_half = []
    for v in series:
        if pd.isna(v):
            vals.append(np.nan)
            is_shared.append(0)
            is_half.append(0)
            continue
        s = str(v).lower().strip()
        sh = 1 if 'shared' in s else 0
        hf = 1 if 'half' in s else 0
        is_shared.append(sh)
        is_half.append(hf)
        if hf:
            num = 0.5
        else:
            m = re.search(r'([0-9]+\.?[0-9]*)', s)
            num = float(m.group(1)) if m else 1.0
        vals.append(num)
    return (
        np.array(vals, dtype=np.float32),
        np.array(is_shared, dtype=np.int8),
        np.array(is_half, dtype=np.int8)
    )

# -----------------------------------------------------------------------------
# 2. 辅助函数：文本清洗与设施提取
# -----------------------------------------------------------------------------
def clean_html_text(text):
    """去除 HTML 标签与多余控制字符"""
    if pd.isna(text) or not text:
        return ""
    s = str(text)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def clean_amenities_list(text):
    """解析 JSON 列表形式的 amenities，标准化为以空格分隔的短语词串"""
    if pd.isna(text) or not text:
        return ""
    items = re.findall(r'"([^"]+)"', str(text))
    if not items:
        items = re.findall(r"'([^']+)'", str(text))
    if not items:
        return ""
    cleaned_items = []
    for item in items:
        clean = re.sub(r'[^a-zA-Z0-9]+', '_', item.strip().lower()).strip('_')
        if clean:
            cleaned_items.append(clean)
    return " ".join(cleaned_items)

# -----------------------------------------------------------------------------
# 3. 基础全量特征工程 (无标签安全计算)
# -----------------------------------------------------------------------------
def build_base_tabular_features(df_train, df_test):
    """构建高阶表格无标签特征，严格移除不存在的经纬度死代码"""
    print(">>> [Step 1] 开始构建高阶表格特征 (空间比率、房东结构、评论时序与层级统计)...")
    t0 = time.time()
    n_train = len(df_train)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    # 1. 浴室解析
    bath_num, bath_shared, bath_half = parse_bathrooms(df['bathrooms_text'])
    df['bathrooms_num'] = bath_num
    df['is_shared_bath'] = bath_shared
    df['is_half_bath'] = bath_half
    
    # 2. 房东时间资历与位置、比率
    df['host_total_months'] = (df['hosts_time_as_user_years'].fillna(0) * 12 + df['hosts_time_as_user_months'].fillna(0)).astype(np.float32)
    df['host_as_host_total_months'] = (df['hosts_time_as_host_years'].fillna(0) * 12 + df['hosts_time_as_host_months'].fillna(0)).astype(np.float32)
    df['host_time_gap_months'] = df['host_total_months'] - df['host_as_host_total_months']
    df['host_is_superhost_bin'] = (df['host_is_superhost'] == 't').astype(np.int8)
    df['host_identity_verified_bin'] = (df['host_identity_verified'] == 't').astype(np.int8)
    df['host_has_profile_pic_bin'] = (df['host_has_profile_pic'] == 't').astype(np.int8)
    df['listings_per_host_year'] = (df['host_listings_count'].fillna(0) / ((df['host_as_host_total_months'] / 12.0) + 0.5)).astype(np.float32)
    
    # 房东位置 london 判定与缺失标记
    if 'host_location' in df.columns:
        df['host_location_is_na'] = df['host_location'].isna().astype(np.int8)
        df['host_location_is_london'] = df['host_location'].fillna('').astype(str).str.lower().str.contains('london', regex=False).astype(np.int8)
    else:
        df['host_location_is_na'] = 0
        df['host_location_is_london'] = 0

    # 房东响应率/接受率 (若存在则解析百分比)
    for rate_col in ['host_response_rate', 'host_acceptance_rate']:
        if rate_col in df.columns:
            df[f'{rate_col}_num'] = df[rate_col].astype(str).str.rstrip('%').replace('nan', np.nan).astype(np.float32) / 100.0
            df[f'{rate_col}_is_na'] = df[rate_col].isna().astype(np.int8)
            
    # 房东验证方式 (若存在解析数量及常见项)
    if 'host_verifications' in df.columns:
        h_ver = df['host_verifications'].fillna('').astype(str).str.lower()
        df['host_verifications_count'] = h_ver.apply(lambda x: len(x.split(',')) if len(x) > 2 else 0).astype(np.float32)
        df['host_ver_email'] = h_ver.str.contains('email', regex=False).astype(np.int8)
        df['host_ver_phone'] = h_ver.str.contains('phone', regex=False).astype(np.int8)
        df['host_ver_work_email'] = h_ver.str.contains('work_email', regex=False).astype(np.int8)
    
    # 3. 空间容纳与容量比率
    df['bedrooms_safe'] = df['bedrooms'].fillna(1.0).clip(lower=0.5).astype(np.float32)
    df['beds_safe'] = df['beds'].fillna(1.0).clip(lower=0.5).astype(np.float32)
    df['bathrooms_safe'] = df['bathrooms_num'].fillna(1.0).clip(lower=0.5).astype(np.float32)
    
    df['accommodates_per_bedroom'] = (df['accommodates'] / df['bedrooms_safe']).astype(np.float32)
    df['accommodates_per_bed'] = (df['accommodates'] / df['beds_safe']).astype(np.float32)
    df['accommodates_per_bath'] = (df['accommodates'] / df['bathrooms_safe']).astype(np.float32)
    df['beds_per_bedroom'] = (df['beds_safe'] / df['bedrooms_safe']).astype(np.float32)
    df['baths_per_bedroom'] = (df['bathrooms_safe'] / df['bedrooms_safe']).astype(np.float32)
    df['total_rooms'] = (df['bedrooms_safe'] + df['bathrooms_safe']).astype(np.float32)
    df['beds_diff_accommodates'] = (df['accommodates'] - df['beds_safe']).astype(np.float32)
    df['space_capacity_index'] = (df['accommodates'] * (df['bedrooms_safe'] + df['bathrooms_safe'])).astype(np.float32)
    df['beds_bath_ratio'] = (df['beds_safe'] / df['bathrooms_safe']).astype(np.float32)
    
    # 4. 房东房源结构占比 (补全 calculated_host_listings_count_* 系列)
    calc_base = df['calculated_host_listings_count'].fillna(1.0).clip(lower=1.0)
    df['ratio_entire_homes'] = (df['calculated_host_listings_count_entire_homes'].fillna(0) / calc_base).astype(np.float32)
    df['ratio_private_rooms'] = (df['calculated_host_listings_count_private_rooms'].fillna(0) / calc_base).astype(np.float32)
    df['ratio_shared_rooms'] = (df['calculated_host_listings_count_shared_rooms'].fillna(0) / calc_base).astype(np.float32)
    df['diff_host_listings_calc'] = (df['host_listings_count'].fillna(0) - df['calculated_host_listings_count'].fillna(0)).astype(np.float32)
    
    # 5. 关键词设施单热提取与打分
    amenities_raw = df['amenities'].fillna('').astype(str).str.lower()
    df['amenities_count'] = amenities_raw.apply(lambda x: len(x.split(',')) if len(x) > 2 else 0).astype(np.float32)
    df['amenities_char_len'] = amenities_raw.str.len().astype(np.float32)
    
    valuable_amenities = [
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
        ('has_waterfront', ['waterfront', 'lake access', 'beachfront']),
        ('has_fireplace', ['fireplace', 'indoor fireplace']),
        ('has_coffee', ['coffee maker', 'nespresso', 'espresso']),
        ('has_ev_charger', ['ev charger']),
        ('has_self_checkin', ['self check-in', 'keypad', 'smart lock'])
    ]
    for col_name, kw_list in valuable_amenities:
        pattern = '|'.join(kw_list)
        df[col_name] = amenities_raw.str.contains(pattern, regex=True).astype(np.int8)
        
    df['luxury_amenities_sum'] = (
        df['has_pool'] + df['has_hot_tub'] + df['has_gym'] + 
        df['has_patio_balcony'] + df['has_air_conditioning'] + 
        df['has_elevator'] + df['has_waterfront'] + df['has_bathtub'] +
        df['has_fireplace'] + df['has_ev_charger']
    ).astype(np.float32)
    
    # 6. 可用率动态与年末可用度 (补全 availability_eoy)
    df['avail_ratio_30'] = (df['availability_30'] / 30.0).astype(np.float32)
    df['avail_ratio_60'] = (df['availability_60'] / 60.0).astype(np.float32)
    df['avail_ratio_90'] = (df['availability_90'] / 90.0).astype(np.float32)
    df['avail_ratio_365'] = (df['availability_365'] / 365.0).astype(np.float32)
    df['avail_ratio_eoy'] = (df['availability_eoy'].fillna(0) / 365.0).astype(np.float32)
    df['avail_diff_30_60'] = (df['availability_60'] - df['availability_30']).astype(np.float32)
    df['avail_diff_60_90'] = (df['availability_90'] - df['availability_60']).astype(np.float32)
    df['avail_diff_90_365'] = (df['availability_365'] - df['availability_90']).astype(np.float32)
    df['avail_diff_90_eoy'] = (df['availability_eoy'].fillna(0) - df['availability_90']).astype(np.float32)
    df['avail_diff_eoy_365'] = (df['availability_365'] - df['availability_eoy'].fillna(0)).astype(np.float32)
    
    # 7. 入住天数策略 (补全 minimum_nights_avg_ntm, maximum_nights_avg_ntm)
    df['is_short_term'] = (df['minimum_nights'] <= 2).astype(np.int8)
    df['is_monthly_plus'] = (df['minimum_nights'] >= 30).astype(np.int8)
    df['min_nights_range'] = (df['maximum_minimum_nights'] - df['minimum_minimum_nights']).astype(np.float32)
    df['max_nights_range'] = (df['maximum_maximum_nights'] - df['minimum_maximum_nights']).astype(np.float32)
    df['min_nights_ratio'] = (df['minimum_minimum_nights'] / (df['maximum_minimum_nights'] + 1)).astype(np.float32)
    df['min_nights_avg_diff'] = (df['minimum_nights_avg_ntm'].fillna(0) - df['minimum_nights'].fillna(0)).astype(np.float32)
    df['max_nights_avg_diff'] = (df['maximum_nights_avg_ntm'].fillna(0) - df['maximum_nights'].fillna(0)).astype(np.float32)
    
    # 8. 评论时序与活跃度 (补全 number_of_reviews_ly)
    base_date = pd.to_datetime('2026-11-01')
    first_dt = pd.to_datetime(df['first_review'], errors='coerce')
    last_dt = pd.to_datetime(df['last_review'], errors='coerce')
    
    df['days_since_first_review'] = (base_date - first_dt).dt.days.fillna(-1).astype(np.float32)
    df['days_since_last_review'] = (base_date - last_dt).dt.days.fillna(-1).astype(np.float32)
    df['review_lifespan_days'] = (last_dt - first_dt).dt.days.fillna(0).astype(np.float32)
    df['review_active_ratio'] = (df['number_of_reviews_ltm'] / (df['number_of_reviews'] + 1)).astype(np.float32)
    df['review_month_velocity'] = (df['number_of_reviews_l30d'] / (df['number_of_reviews_ltm'] + 1)).astype(np.float32)
    df['reviews_per_accommodates'] = (df['number_of_reviews'] / (df['accommodates'] + 1)).astype(np.float32)
    df['monthly_review_power'] = (df['reviews_per_month'].fillna(0) * df['review_scores_rating'].fillna(0)).astype(np.float32)
    df['review_ly_ratio'] = (df['number_of_reviews_ly'].fillna(0) / (df['number_of_reviews'] + 1)).astype(np.float32)
    df['review_growth_ltm_vs_ly'] = ((df['number_of_reviews_ltm'].fillna(0) - df['number_of_reviews_ly'].fillna(0)) / (df['number_of_reviews_ly'].fillna(0) + 1)).astype(np.float32)
    
    review_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'
    ]
    df['review_scores_mean'] = df[review_cols].mean(axis=1).astype(np.float32)
    df['review_scores_std'] = df[review_cols].std(axis=1).fillna(0).astype(np.float32)
    df['rating_val_interaction'] = (df['review_scores_rating'].fillna(0) * df['review_scores_value'].fillna(0)).astype(np.float32)
    df['clean_loc_interaction'] = (df['review_scores_cleanliness'].fillna(0) * df['review_scores_location'].fillna(0)).astype(np.float32)
    
    # 9. 文本浅层统计与关键词标记
    name_str_lower = df['name'].fillna('').astype(str).str.lower()
    raw_name = df['name'].fillna('').astype(str)
    desc_str_lower = df['description'].fillna('').astype(str).str.lower()
    
    df['name_len'] = name_str_lower.str.len().astype(np.float32)
    df['name_words'] = name_str_lower.str.split().str.len().astype(np.float32)
    df['desc_len'] = desc_str_lower.str.len().astype(np.float32)
    df['desc_words'] = desc_str_lower.str.split().str.len().astype(np.float32)
    df['name_exclamations'] = raw_name.str.count('!').astype(np.float32)
    df['name_question_marks'] = raw_name.str.count(r'\?').astype(np.float32)
    df['name_digits'] = raw_name.str.count(r'[0-9]').astype(np.float32)
    df['name_upper_ratio'] = (raw_name.str.findall(r'[A-Z]').str.len() / (df['name_len'] + 1)).fillna(0).astype(np.float32)
    
    luxury_keywords = ['luxury', 'luxurious', 'stunning', 'penthouse', 'spacious', 'modern', 'central', 'designer', 'prime', 'balcony', 'view', 'boutique', 'villa', 'exclusive']
    for kw in luxury_keywords:
        df[f'name_kw_{kw}'] = name_str_lower.str.contains(kw, regex=False).astype(np.int8)
    budget_keywords = ['cosy', 'cozy', 'budget', 'tiny', 'small', 'student', 'shared', 'room in', 'simple', 'basement']
    for kw in budget_keywords:
        clean_col = kw.replace(' ', '_')
        df[f'name_kw_{clean_col}'] = name_str_lower.str.contains(kw, regex=False).astype(np.int8)

    # room_type one-hot 编码 (供残差 TE 剥离房型基准溢价)
    for rt in ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']:
        clean_rt = re.sub(r'[^a-zA-Z0-9]+', '_', rt.lower()).strip('_')
        df[f'rt_{clean_rt}'] = (df['room_type'] == rt).astype(np.float32)
        
    # 10. 高阶交叉组合与频数编码
    df['geo_room_combo'] = df['geo_cell'].astype(str) + '__' + df['room_type'].astype(str)
    df['prop_room_combo'] = df['property_type'].astype(str) + '__' + df['room_type'].astype(str)
    df['host_room_combo'] = df['host_id'].astype(str) + '__' + df['room_type'].astype(str)
    
    # 孪生房源复合键：twin_4tuple (host_id+geo_cell+room_type+accommodates) 与 twin_5tuple (再加 bedrooms)
    df['twin_4tuple'] = (
        df['host_id'].astype(str) + '__' + 
        df['geo_cell'].astype(str) + '__' + 
        df['room_type'].astype(str) + '__' + 
        df['accommodates'].fillna(-1).astype(str)
    )
    df['twin_5tuple'] = (
        df['twin_4tuple'] + '__' + 
        df['bedrooms'].fillna(-1).astype(str)
    )
    
    cat_cols_for_freq = [
        'geo_cell', 'host_id', 'property_type', 'room_type', 
        'geo_room_combo', 'prop_room_combo', 'host_room_combo',
        'twin_4tuple', 'twin_5tuple'
    ]
    for c in cat_cols_for_freq:
        freq = df[c].value_counts()
        df[f'{c}_freq'] = df[c].map(freq).astype(np.float32)
        
    # 11. 空间层级无标签统计聚合 (按 geo_cell, host_id, geo_room_combo 统计 diff / ratio / zscore)
    for grp_col in ['geo_cell', 'host_id', 'geo_room_combo']:
        for val_col in ['accommodates', 'bedrooms_safe', 'bathrooms_safe', 'review_scores_rating']:
            mean_series = df.groupby(grp_col)[val_col].transform('mean')
            std_series = df.groupby(grp_col)[val_col].transform('std').fillna(1.0)
            df[f'{grp_col}_{val_col}_mean'] = mean_series.astype(np.float32)
            df[f'{grp_col}_{val_col}_std'] = std_series.astype(np.float32)
            df[f'{val_col}_diff_to_{grp_col}'] = (df[val_col] - mean_series).astype(np.float32)
            df[f'{val_col}_ratio_to_{grp_col}'] = (df[val_col] / (mean_series + 0.1)).astype(np.float32)
            df[f'{val_col}_zscore_to_{grp_col}'] = ((df[val_col] - mean_series) / (std_series + 1e-4)).astype(np.float32)
            
    # 房东多房源与地理分散度
    df['host_listing_total_count'] = df.groupby('host_id')['id'].transform('count').astype(np.float32)
    df['host_geo_entropy'] = df.groupby('host_id')['geo_cell'].transform('nunique').astype(np.float32)
    df['host_room_type_nunique'] = df.groupby('host_id')['room_type'].transform('nunique').astype(np.float32)
    df['geo_host_nunique'] = df.groupby('geo_cell')['host_id'].transform('nunique').astype(np.float32)
    df['geo_cell_listing_count'] = df.groupby('geo_cell')['id'].transform('count').astype(np.float32)
    
    print(f">>> [Step 1] 表格基础特征构建完成，总耗时 {time.time() - t0:.2f}s，当前特征总列数: {df.shape[1]}")
    
    train_feat = df.iloc[:n_train].copy().reset_index(drop=True)
    test_feat = df.iloc[n_train:].copy().reset_index(drop=True)
    
    # 彻底杜绝泄漏：从基座特征中删除 price 列并严格断言
    train_feat = train_feat.drop(columns=['price'], errors='ignore')
    test_feat = test_feat.drop(columns=['price'], errors='ignore')
    assert 'price' not in train_feat.columns, "Fatal: 'price' found in train_feat!"
    assert 'price' not in test_feat.columns, "Fatal: 'price' found in test_feat!"
    return train_feat, test_feat

# -----------------------------------------------------------------------------
# 4. 文本清洗与文本底座落盘
# -----------------------------------------------------------------------------
def process_and_cache_text_corpus(df_train, df_test):
    """清洗 name, description, amenities，保存供原生文本模型与 TF-IDF 使用"""
    print(">>> [Step 2] 正在清洗自然语言文本 (HTML剥离、格式化与 amenities 词串化)...")
    t0 = time.time()
    
    tr_name = df_train['name'].apply(clean_html_text).values
    te_name = df_test['name'].apply(clean_html_text).values
    
    tr_desc = df_train['description'].apply(clean_html_text).values
    te_desc = df_test['description'].apply(clean_html_text).values
    
    tr_amen = df_train['amenities'].apply(clean_amenities_list).values
    te_amen = df_test['amenities'].apply(clean_amenities_list).values
    
    tr_full = [f"{n} {d} {a}".strip() for n, d, a in zip(tr_name, tr_desc, tr_amen)]
    te_full = [f"{n} {d} {a}".strip() for n, d, a in zip(te_name, te_desc, te_amen)]
    
    # 构造孪生房源复合键
    tr_twin_4 = (df_train['host_id'].astype(str) + '__' + df_train['geo_cell'].astype(str) + '__' + df_train['room_type'].astype(str) + '__' + df_train['accommodates'].fillna(-1).astype(str)).values
    te_twin_4 = (df_test['host_id'].astype(str) + '__' + df_test['geo_cell'].astype(str) + '__' + df_test['room_type'].astype(str) + '__' + df_test['accommodates'].fillna(-1).astype(str)).values
    tr_twin_5 = (pd.Series(tr_twin_4) + '__' + df_train['bedrooms'].fillna(-1).astype(str).values).values
    te_twin_5 = (pd.Series(te_twin_4) + '__' + df_test['bedrooms'].fillna(-1).astype(str).values).values

    # 构造并落盘 parquet
    train_text_df = pd.DataFrame({
        'id': df_train['id'].values,
        'clean_name': tr_name,
        'clean_desc': tr_desc,
        'clean_amenities': tr_amen,
        'clean_full_text': tr_full,
        'geo_cell': df_train['geo_cell'].astype(str).values,
        'host_id': df_train['host_id'].astype(str).values,
        'room_type': df_train['room_type'].astype(str).values,
        'property_type': df_train['property_type'].astype(str).values,
        'geo_room_combo': (df_train['geo_cell'].astype(str) + '__' + df_train['room_type'].astype(str)).values,
        'prop_room_combo': (df_train['property_type'].astype(str) + '__' + df_train['room_type'].astype(str)).values,
        'host_room_combo': (df_train['host_id'].astype(str) + '__' + df_train['room_type'].astype(str)).values,
        'twin_4tuple': tr_twin_4,
        'twin_5tuple': tr_twin_5
    })
    
    test_text_df = pd.DataFrame({
        'id': df_test['id'].values,
        'clean_name': te_name,
        'clean_desc': te_desc,
        'clean_amenities': te_amen,
        'clean_full_text': te_full,
        'geo_cell': df_test['geo_cell'].astype(str).values,
        'host_id': df_test['host_id'].astype(str).values,
        'room_type': df_test['room_type'].astype(str).values,
        'property_type': df_test['property_type'].astype(str).values,
        'geo_room_combo': (df_test['geo_cell'].astype(str) + '__' + df_test['room_type'].astype(str)).values,
        'prop_room_combo': (df_test['property_type'].astype(str) + '__' + df_test['room_type'].astype(str)).values,
        'host_room_combo': (df_test['host_id'].astype(str) + '__' + df_test['room_type'].astype(str)).values,
        'twin_4tuple': te_twin_4,
        'twin_5tuple': te_twin_5
    })
    
    train_text_df.to_parquet(CACHE_DIR / 'train_text.parquet', index=False)
    test_text_df.to_parquet(CACHE_DIR / 'test_text.parquet', index=False)
    print(f">>> [Step 2] 文本底座与原生类别清洗完成并已落盘！耗时 {time.time() - t0:.2f}s")
    
    return train_text_df, test_text_df

# -----------------------------------------------------------------------------
# 5. 严格折内无泄漏 Target Encoding & Residual TE & Text Zoo Pipeline
# -----------------------------------------------------------------------------
def run_10fold_oof_feature_pipeline(train_feat, test_feat, train_text, test_text, y_log1p):
    """
    统一 10-Fold 严格折内无泄漏特征提取管道：
    1. 目标类别平滑均值 + 标准差 Target Encoding (Inner 5-Fold OOF)
    2. 去规模/去户型残差 TE (Residual TE) 与 log(price/accommodates) TE
    3. 多路稀疏 TF-IDF OOF Ridge / SGD 元特征 (Text Zoo)
    4. 折内无监督 SVD 降维语义特征
    """
    print("=" * 70)
    print("🚀 [Step 3] 启动统一 10-Fold 严格折内防泄漏 OOF 元特征系统 (Inner 5-Fold OOF)...")
    print("=" * 70)
    t_start = time.time()
    
    n_train = len(train_feat)
    n_test = len(test_feat)
    
    # 统一 10 折
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_splits = list(kf.split(train_feat))
    
    # 保存折索引与标签
    np.save(CACHE_DIR / 'folds_10.npy', np.array(fold_splits, dtype=object), allow_pickle=True)
    np.save(CACHE_DIR / 'y_log1p.npy', y_log1p)
    
    fold_ids = np.zeros(n_train, dtype=np.int32)
    for fold, (_, val_idx) in enumerate(fold_splits):
        fold_ids[val_idx] = fold
    np.save(CACHE_DIR / 'fold_ids.npy', fold_ids)
    
    # 类别目标编码配置
    cat_te_cols = ['geo_cell', 'host_id', 'geo_room_combo', 'prop_room_combo', 'host_room_combo', 'twin_4tuple', 'twin_5tuple']
    m_smooth = 10.0
    m_low = 2.0  # 针对孪生房源低平滑度 Target Encoding
    
    # 定义待生成的 OOF 特征列
    te_mean_cols = [f'{c}_te_mean' for c in cat_te_cols]
    te_std_cols = [f'{c}_te_std' for c in cat_te_cols]
    twin_low_te_cols = ['twin_4tuple_te_m2', 'twin_5tuple_te_m2']
    res_te_cols = [f'{c}_res_te_mean' for c in ['geo_cell', 'host_id', 'geo_room_combo', 'prop_room_combo']]
    unit_te_cols = [f'{c}_unit_te_mean' for c in ['geo_cell', 'host_id', 'geo_room_combo']]
    
    # Text Zoo 元特征清单
    text_model_names = [
        'name_ridge_a1.5',
        'name_ridge_a6.0',
        'desc_ridge_a6.0',
        'desc_ridge_a20.0',
        'amen_ridge_a6.0',
        'amen_ridge_a20.0',
        'full_ridge_a1.5',
        'full_ridge_a6.0',
        'full_ridge_a20.0',
        'full_ridge_res_a6.0',
        'full_sgd'
    ]
    
    # SVD 语义特征列
    svd_cols = [f'name_svd_{i}' for i in range(6)] + [f'desc_svd_{i}' for i in range(6)] + [f'amen_svd_{i}' for i in range(6)]
    
    # 初始化训练集 OOF 矩阵与测试集预测累加矩阵
    all_oof_col_names = te_mean_cols + te_std_cols + twin_low_te_cols + res_te_cols + unit_te_cols + text_model_names + svd_cols
    train_oof_df = pd.DataFrame(0.0, index=range(n_train), columns=all_oof_col_names, dtype=np.float32)
    test_oof_df = pd.DataFrame(0.0, index=range(n_test), columns=all_oof_col_names, dtype=np.float32)
    
    # 用于记录各文本元模型外层验证集评分 (10-Fold CV RMSLE)
    text_oof_preds = {name: np.zeros(n_train, dtype=np.float32) for name in text_model_names}
    
    # 房源规模基础特征 (加入 room_type one-hot，用于拟合规模与户型基准以提取纯净地段/房东残差)
    rt_cols = [c for c in train_feat.columns if c.startswith('rt_')]
    size_features = ['accommodates', 'bedrooms_safe', 'bathrooms_safe', 'beds_safe', 'total_rooms', 'space_capacity_index'] + rt_cols
    
    # 准备文本数组
    tr_name_arr = train_text['clean_name'].values
    te_name_arr = test_text['clean_name'].values
    tr_desc_arr = train_text['clean_desc'].values
    te_desc_arr = test_text['clean_desc'].values
    tr_amen_arr = train_text['clean_amenities'].values
    te_amen_arr = test_text['clean_amenities'].values
    tr_full_arr = train_text['clean_full_text'].values
    te_full_arr = test_text['clean_full_text'].values
    
    unit_price_log = y_log1p - np.log1p(train_feat['accommodates'].clip(lower=1.0).values)
    
    # ---------------- 10-Fold 循环 ----------------
    for fold, (tr_idx, va_idx) in enumerate(fold_splits):
        t_fold = time.time()
        print(f"\n--- [Outer Fold {fold + 1} / 10] (Train: {len(tr_idx)}, Val: {len(va_idx)}) ---")
        
        # 支持断点续跑：若当前折 fold_{fold}.npz 已经就绪，直接快速恢复，省去重复计算
        fold_npz_path = CACHE_DIR / 'fold_te' / f'fold_{fold}.npz'
        if fold_npz_path.exists():
            print(f">>> [Outer Fold {fold + 1} / 10] 检测到已有完整特征矩阵 {fold_npz_path.name}，正在从缓存断点恢复...")
            npz_data = np.load(fold_npz_path)
            c_names = list(npz_data['col_names'])
            train_oof_df.loc[va_idx, c_names] = npz_data['X_va_te']
            test_oof_df[c_names] += npz_data['X_te_te'] / 10.0
            for m_name in text_model_names:
                text_oof_preds[m_name][va_idx] = train_oof_df.loc[va_idx, m_name]
            va_rmsle_full_ridge = np.sqrt(mean_squared_error(y_log1p[va_idx], train_oof_df.loc[va_idx, 'full_ridge_a6.0']))
            va_rmsle_name_ridge = np.sqrt(mean_squared_error(y_log1p[va_idx], train_oof_df.loc[va_idx, 'name_ridge_a6.0']))
            print(f"   -> [Fold {fold + 1} 验证评估 (已从缓存加载)] Full-Text Ridge RMSLE: {va_rmsle_full_ridge:.4f} | Name Ridge RMSLE: {va_rmsle_name_ridge:.4f}")
            continue
        
        y_tr = y_log1p[tr_idx]
        y_va = y_log1p[va_idx]
        unit_tr = unit_price_log[tr_idx]
        
        global_mean_y = float(y_tr.mean())
        global_std_y = float(y_tr.std())
        global_mean_unit = float(unit_tr.mean())
        
        # 内层 5-Fold 划分 (用于严格训练折自身无泄漏编码)
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42 + fold)
        
        # 收集当前外折折内无泄漏特征 (tr, va, te)
        fold_tr_te_dict = {}
        fold_va_te_dict = {}
        fold_te_te_dict = {}
        
        # =====================================================================
        # A. 基础类别 Target Encoding (均值 + 标准差)
        # =====================================================================
        for c in cat_te_cols:
            col_tr = train_feat.iloc[tr_idx][c].astype(str).reset_index(drop=True)
            col_va = train_feat.iloc[va_idx][c].astype(str).reset_index(drop=True)
            col_te = test_feat[c].astype(str).reset_index(drop=True)
            
            # 全外折训练集拟合，映射 Val 与 Test
            full_tr_stats = pd.DataFrame({'cat': col_tr, 'target': y_tr}).groupby('cat').agg(['sum', 'count', 'std'])
            f_counts = full_tr_stats[('target', 'count')]
            f_sums = full_tr_stats[('target', 'sum')]
            f_stds = full_tr_stats[('target', 'std')].fillna(global_std_y)
            f_smoothed = (f_sums + m_smooth * global_mean_y) / (f_counts + m_smooth)
            
            va_mean = col_va.map(f_smoothed).fillna(global_mean_y).values.astype(np.float32)
            va_std = col_va.map(f_stds).fillna(global_std_y).values.astype(np.float32)
            te_mean = col_te.map(f_smoothed).fillna(global_mean_y).values.astype(np.float32)
            te_std = col_te.map(f_stds).fillna(global_std_y).values.astype(np.float32)
            
            train_oof_df.loc[va_idx, f'{c}_te_mean'] = va_mean
            train_oof_df.loc[va_idx, f'{c}_te_std'] = va_std
            test_oof_df[f'{c}_te_mean'] += te_mean / 10.0
            test_oof_df[f'{c}_te_std'] += te_std / 10.0
            
            fold_va_te_dict[f'{c}_te_mean'] = va_mean
            fold_va_te_dict[f'{c}_te_std'] = va_std
            fold_te_te_dict[f'{c}_te_mean'] = te_mean
            fold_te_te_dict[f'{c}_te_std'] = te_std
            
            # --- tr_idx 严格折内 5 折 inner_kf 计算 (绝不触碰 va_idx) ---
            tr_mean = np.full(len(tr_idx), global_mean_y, dtype=np.float32)
            tr_std = np.full(len(tr_idx), global_std_y, dtype=np.float32)
            for in_tr, in_va in inner_kf.split(col_tr):
                in_mean = float(y_tr[in_tr].mean())
                in_std_g = float(y_tr[in_tr].std())
                in_stats = pd.DataFrame({'cat': col_tr.iloc[in_tr], 'target': y_tr[in_tr]}).groupby('cat').agg(['sum', 'count', 'std'])
                in_counts = in_stats[('target', 'count')]
                in_sums = in_stats[('target', 'sum')]
                in_stds = in_stats[('target', 'std')].fillna(in_std_g)
                in_smoothed = (in_sums + m_smooth * in_mean) / (in_counts + m_smooth)
                
                tr_mean[in_va] = col_tr.iloc[in_va].map(in_smoothed).fillna(in_mean).values.astype(np.float32)
                tr_std[in_va] = col_tr.iloc[in_va].map(in_stds).fillna(in_std_g).values.astype(np.float32)
            fold_tr_te_dict[f'{c}_te_mean'] = tr_mean
            fold_tr_te_dict[f'{c}_te_std'] = tr_std
            
        # 对孪生房源应用低平滑度 (m=2.0) Target Encoding
        for c in ['twin_4tuple', 'twin_5tuple']:
            col_tr = train_feat.iloc[tr_idx][c].astype(str).reset_index(drop=True)
            col_va = train_feat.iloc[va_idx][c].astype(str).reset_index(drop=True)
            col_te = test_feat[c].astype(str).reset_index(drop=True)
            
            full_tr_stats = pd.DataFrame({'cat': col_tr, 'target': y_tr}).groupby('cat').agg(['sum', 'count'])
            f_counts = full_tr_stats[('target', 'count')]
            f_sums = full_tr_stats[('target', 'sum')]
            f_smoothed_low = (f_sums + m_low * global_mean_y) / (f_counts + m_low)
            
            va_m2 = col_va.map(f_smoothed_low).fillna(global_mean_y).values.astype(np.float32)
            te_m2 = col_te.map(f_smoothed_low).fillna(global_mean_y).values.astype(np.float32)
            
            train_oof_df.loc[va_idx, f'{c}_te_m2'] = va_m2
            test_oof_df[f'{c}_te_m2'] += te_m2 / 10.0
            fold_va_te_dict[f'{c}_te_m2'] = va_m2
            fold_te_te_dict[f'{c}_te_m2'] = te_m2
            
            tr_m2 = np.full(len(tr_idx), global_mean_y, dtype=np.float32)
            for in_tr, in_va in inner_kf.split(col_tr):
                in_mean = float(y_tr[in_tr].mean())
                in_stats = pd.DataFrame({'cat': col_tr.iloc[in_tr], 'target': y_tr[in_tr]}).groupby('cat').agg(['sum', 'count'])
                in_counts = in_stats[('target', 'count')]
                in_sums = in_stats[('target', 'sum')]
                in_smoothed_low = (in_sums + m_low * in_mean) / (in_counts + m_low)
                tr_m2[in_va] = col_tr.iloc[in_va].map(in_smoothed_low).fillna(in_mean).values.astype(np.float32)
            fold_tr_te_dict[f'{c}_te_m2'] = tr_m2
            
        # =====================================================================
        # B. 去规模/去户型残差 TE (Residual TE) 与单人容纳 TE
        # =====================================================================
        # 拟合基础规模 Ridge 模型
        X_size_tr = train_feat.iloc[tr_idx][size_features].fillna(0).values
        X_size_va = train_feat.iloc[va_idx][size_features].fillna(0).values
        X_size_te = test_feat[size_features].fillna(0).values
        
        # 折内 Inner-OOF 计算残差基准 r_inner_tr (严格 5 折 inner_kf)
        r_inner_tr = np.zeros(len(tr_idx), dtype=np.float32)
        for in_tr_idx, in_va_idx in inner_kf.split(X_size_tr):
            ridge_size = Ridge(alpha=10.0, solver='lsqr')
            ridge_size.fit(X_size_tr[in_tr_idx], y_tr[in_tr_idx])
            pred_in_va = ridge_size.predict(X_size_tr[in_va_idx])
            r_inner_tr[in_va_idx] = y_tr[in_va_idx] - pred_in_va
            
        global_mean_r = float(r_inner_tr.mean())
        
        # 编码残差 TE
        for c in ['geo_cell', 'host_id', 'geo_room_combo', 'prop_room_combo']:
            col_tr = train_feat.iloc[tr_idx][c].astype(str).reset_index(drop=True)
            col_va = train_feat.iloc[va_idx][c].astype(str).reset_index(drop=True)
            col_te = test_feat[c].astype(str).reset_index(drop=True)
            
            full_stats = pd.DataFrame({'cat': col_tr, 'res': r_inner_tr}).groupby('cat').agg(['sum', 'count'])
            f_counts = full_stats[('res', 'count')]
            f_sums = full_stats[('res', 'sum')]
            f_smoothed_r = (f_sums + m_smooth * global_mean_r) / (f_counts + m_smooth)
            
            va_res = col_va.map(f_smoothed_r).fillna(global_mean_r).values.astype(np.float32)
            te_res = col_te.map(f_smoothed_r).fillna(global_mean_r).values.astype(np.float32)
            train_oof_df.loc[va_idx, f'{c}_res_te_mean'] = va_res
            test_oof_df[f'{c}_res_te_mean'] += te_res / 10.0
            fold_va_te_dict[f'{c}_res_te_mean'] = va_res
            fold_te_te_dict[f'{c}_res_te_mean'] = te_res
            
            tr_res = np.full(len(tr_idx), global_mean_r, dtype=np.float32)
            for in_tr, in_va in inner_kf.split(col_tr):
                in_mean_r = float(r_inner_tr[in_tr].mean())
                in_stats = pd.DataFrame({'cat': col_tr.iloc[in_tr], 'res': r_inner_tr[in_tr]}).groupby('cat').agg(['sum', 'count'])
                in_counts = in_stats[('res', 'count')]
                in_sums = in_stats[('res', 'sum')]
                in_smoothed_r = (in_sums + m_smooth * in_mean_r) / (in_counts + m_smooth)
                tr_res[in_va] = col_tr.iloc[in_va].map(in_smoothed_r).fillna(in_mean_r).values.astype(np.float32)
            fold_tr_te_dict[f'{c}_res_te_mean'] = tr_res
            
        # 编码单人容纳 unit TE
        for c in ['geo_cell', 'host_id', 'geo_room_combo']:
            col_tr = train_feat.iloc[tr_idx][c].astype(str).reset_index(drop=True)
            col_va = train_feat.iloc[va_idx][c].astype(str).reset_index(drop=True)
            col_te = test_feat[c].astype(str).reset_index(drop=True)
            
            full_stats = pd.DataFrame({'cat': col_tr, 'unit': unit_tr}).groupby('cat').agg(['sum', 'count'])
            f_counts = full_stats[('unit', 'count')]
            f_sums = full_stats[('unit', 'sum')]
            f_smoothed_u = (f_sums + m_smooth * global_mean_unit) / (f_counts + m_smooth)
            
            va_unit = col_va.map(f_smoothed_u).fillna(global_mean_unit).values.astype(np.float32)
            te_unit = col_te.map(f_smoothed_u).fillna(global_mean_unit).values.astype(np.float32)
            train_oof_df.loc[va_idx, f'{c}_unit_te_mean'] = va_unit
            test_oof_df[f'{c}_unit_te_mean'] += te_unit / 10.0
            fold_va_te_dict[f'{c}_unit_te_mean'] = va_unit
            fold_te_te_dict[f'{c}_unit_te_mean'] = te_unit
            
            tr_unit = np.full(len(tr_idx), global_mean_unit, dtype=np.float32)
            for in_tr, in_va in inner_kf.split(col_tr):
                in_mean_u = float(unit_tr[in_tr].mean())
                in_stats = pd.DataFrame({'cat': col_tr.iloc[in_tr], 'unit': unit_tr[in_tr]}).groupby('cat').agg(['sum', 'count'])
                in_counts = in_stats[('unit', 'count')]
                in_sums = in_stats[('unit', 'sum')]
                in_smoothed_u = (in_sums + m_smooth * in_mean_u) / (in_counts + m_smooth)
                tr_unit[in_va] = col_tr.iloc[in_va].map(in_smoothed_u).fillna(in_mean_u).values.astype(np.float32)
            fold_tr_te_dict[f'{c}_unit_te_mean'] = tr_unit

        # =====================================================================
        # C. 多路稀疏 TF-IDF OOF Ridge / SGD 元特征 (Text Zoo)
        # =====================================================================
        print("   -> 正在提取多路文本 TF-IDF 并拟合 Ridge / SGD 元特征...")
        # 1. Name TF-IDF
        vec_name = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_name_tr = vec_name.fit_transform(tr_name_arr[tr_idx])
        X_name_va = vec_name.transform(tr_name_arr[va_idx])
        X_name_te = vec_name.transform(te_name_arr)
        
        # 2. Desc TF-IDF
        vec_desc = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_desc_tr = vec_desc.fit_transform(tr_desc_arr[tr_idx])
        X_desc_va = vec_desc.transform(tr_desc_arr[va_idx])
        X_desc_te = vec_desc.transform(te_desc_arr)
        
        # 3. Amen TF-IDF
        vec_amen = TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b', max_features=12000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_amen_tr = vec_amen.fit_transform(tr_amen_arr[tr_idx])
        X_amen_va = vec_amen.transform(tr_amen_arr[va_idx])
        X_amen_te = vec_amen.transform(te_amen_arr)
        
        # 4. Full Text TF-IDF
        vec_full = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_full_tr = vec_full.fit_transform(tr_full_arr[tr_idx])
        X_full_va = vec_full.transform(tr_full_arr[va_idx])
        X_full_te = vec_full.transform(te_full_arr)
        
        # 定义子模型执行清单: (feature_matrix_tr, va, te, model_inst, model_name, target_type)
        zoo_configs = [
            (X_name_tr, X_name_va, X_name_te, Ridge(alpha=1.5, solver='lsqr'), 'name_ridge_a1.5', 'y'),
            (X_name_tr, X_name_va, X_name_te, Ridge(alpha=6.0, solver='lsqr'), 'name_ridge_a6.0', 'y'),
            (X_desc_tr, X_desc_va, X_desc_te, Ridge(alpha=6.0, solver='lsqr'), 'desc_ridge_a6.0', 'y'),
            (X_desc_tr, X_desc_va, X_desc_te, Ridge(alpha=20.0, solver='lsqr'), 'desc_ridge_a20.0', 'y'),
            (X_amen_tr, X_amen_va, X_amen_te, Ridge(alpha=6.0, solver='lsqr'), 'amen_ridge_a6.0', 'y'),
            (X_amen_tr, X_amen_va, X_amen_te, Ridge(alpha=20.0, solver='lsqr'), 'amen_ridge_a20.0', 'y'),
            (X_full_tr, X_full_va, X_full_te, Ridge(alpha=1.5, solver='lsqr'), 'full_ridge_a1.5', 'y'),
            (X_full_tr, X_full_va, X_full_te, Ridge(alpha=6.0, solver='lsqr'), 'full_ridge_a6.0', 'y'),
            (X_full_tr, X_full_va, X_full_te, Ridge(alpha=20.0, solver='lsqr'), 'full_ridge_a20.0', 'y'),
            (X_full_tr, X_full_va, X_full_te, Ridge(alpha=6.0, solver='lsqr'), 'full_ridge_res_a6.0', 'res'),
            (X_full_tr, X_full_va, X_full_te, SGDRegressor(alpha=1e-4, max_iter=30, tol=1e-3, random_state=42), 'full_sgd', 'y'),
        ]
        
        for m_X_tr, m_X_va, m_X_te, model_obj, m_name, target_type in zoo_configs:
            m_target_tr = y_tr if target_type == 'y' else r_inner_tr
            
            # 外折全局拟合当前外折训练集，预测 va 与 te
            model_obj.fit(m_X_tr, m_target_tr)
            va_pred = model_obj.predict(m_X_va).astype(np.float32)
            te_pred = model_obj.predict(m_X_te).astype(np.float32)
            
            train_oof_df.loc[va_idx, m_name] = va_pred
            test_oof_df[m_name] += te_pred / 10.0
            text_oof_preds[m_name][va_idx] = va_pred
            
            fold_va_te_dict[m_name] = va_pred
            fold_te_te_dict[m_name] = te_pred
            
            # 为 tr_idx 在 inner_kf 下进行严格折内拟合预测 (零泄漏)
            tr_pred = np.zeros(len(tr_idx), dtype=np.float32)
            for in_tr_idx, in_va_idx in inner_kf.split(m_X_tr):
                in_model = clone(model_obj)
                in_model.fit(m_X_tr[in_tr_idx], m_target_tr[in_tr_idx])
                tr_pred[in_va_idx] = in_model.predict(m_X_tr[in_va_idx]).astype(np.float32)
            fold_tr_te_dict[m_name] = tr_pred
            
        # =====================================================================
        # D. 折内 SVD 降维语义特征 (严格外折训练集拟合，映射 Val 与 Test)
        # =====================================================================
        svd_name = TruncatedSVD(n_components=6, random_state=42).fit(X_name_tr)
        svd_desc = TruncatedSVD(n_components=6, random_state=42).fit(X_desc_tr)
        svd_amen = TruncatedSVD(n_components=6, random_state=42).fit(X_amen_tr)
        
        name_svd_tr = svd_name.transform(X_name_tr).astype(np.float32)
        desc_svd_tr = svd_desc.transform(X_desc_tr).astype(np.float32)
        amen_svd_tr = svd_amen.transform(X_amen_tr).astype(np.float32)
        
        name_svd_va = svd_name.transform(X_name_va).astype(np.float32)
        desc_svd_va = svd_desc.transform(X_desc_va).astype(np.float32)
        amen_svd_va = svd_amen.transform(X_amen_va).astype(np.float32)
        
        name_svd_te = svd_name.transform(X_name_te).astype(np.float32)
        desc_svd_te = svd_desc.transform(X_desc_te).astype(np.float32)
        amen_svd_te = svd_amen.transform(X_amen_te).astype(np.float32)
        
        for i in range(6):
            train_oof_df.loc[va_idx, f'name_svd_{i}'] = name_svd_va[:, i]
            train_oof_df.loc[va_idx, f'desc_svd_{i}'] = desc_svd_va[:, i]
            train_oof_df.loc[va_idx, f'amen_svd_{i}'] = amen_svd_va[:, i]
            
            test_oof_df[f'name_svd_{i}'] += name_svd_te[:, i] / 10.0
            test_oof_df[f'desc_svd_{i}'] += desc_svd_te[:, i] / 10.0
            test_oof_df[f'amen_svd_{i}'] += amen_svd_te[:, i] / 10.0
            
            fold_tr_te_dict[f'name_svd_{i}'] = name_svd_tr[:, i]
            fold_tr_te_dict[f'desc_svd_{i}'] = desc_svd_tr[:, i]
            fold_tr_te_dict[f'amen_svd_{i}'] = amen_svd_tr[:, i]
            
            fold_va_te_dict[f'name_svd_{i}'] = name_svd_va[:, i]
            fold_va_te_dict[f'desc_svd_{i}'] = desc_svd_va[:, i]
            fold_va_te_dict[f'amen_svd_{i}'] = amen_svd_va[:, i]
            
            fold_te_te_dict[f'name_svd_{i}'] = name_svd_te[:, i]
            fold_te_te_dict[f'desc_svd_{i}'] = desc_svd_te[:, i]
            fold_te_te_dict[f'amen_svd_{i}'] = amen_svd_te[:, i]
            
        # 打包保存每折对齐好的 fold_{fold}.npz
        fold_te_dir = CACHE_DIR / 'fold_te'
        fold_te_dir.mkdir(parents=True, exist_ok=True)
        
        col_names = np.array(all_oof_col_names, dtype=str)
        X_tr_te = np.column_stack([fold_tr_te_dict[c] for c in all_oof_col_names]).astype(np.float32)
        X_va_te = np.column_stack([fold_va_te_dict[c] for c in all_oof_col_names]).astype(np.float32)
        X_te_te = np.column_stack([fold_te_te_dict[c] for c in all_oof_col_names]).astype(np.float32)
        
        # 严格校验无 NaN / Inf
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
            col_names=col_names
        )
        print(f"   -> [Fold {fold + 1}] 已保存严格无泄漏折内特征矩阵至 fold_{fold}.npz (X_tr_te: {X_tr_te.shape}, X_va_te: {X_va_te.shape}, X_te_te: {X_te_te.shape})")
            
        # 打印当前 Fold 的单路表现
        va_rmsle_full_ridge = np.sqrt(mean_squared_error(y_va, train_oof_df.loc[va_idx, 'full_ridge_a6.0']))
        va_rmsle_name_ridge = np.sqrt(mean_squared_error(y_va, train_oof_df.loc[va_idx, 'name_ridge_a6.0']))
        print(f"   -> [Fold {fold + 1} 验证评估] Full-Text Ridge RMSLE: {va_rmsle_full_ridge:.4f} | Name Ridge RMSLE: {va_rmsle_name_ridge:.4f} | 折耗时: {time.time() - t_fold:.2f}s")
        
        # 释放显存与内存缓存
        del vec_name, vec_desc, vec_amen, vec_full
        del X_name_tr, X_name_va, X_name_te
        del X_desc_tr, X_desc_va, X_desc_te
        del X_amen_tr, X_amen_va, X_amen_te
        del X_full_tr, X_full_va, X_full_te
        gc.collect()

    print("=" * 70)
    print("📊 [Text Zoo 全局 10-Fold CV RMSLE 单路汇总]")
    cv_summary = {}
    for m_name in text_model_names:
        score = float(np.sqrt(mean_squared_error(y_log1p, text_oof_preds[m_name])))
        cv_summary[m_name] = round(score, 5)
        print(f"   * {m_name.ljust(22)}: 10-Fold CV RMSLE = {score:.5f}")
    print("=" * 70)

    # 保存 11 路 Text Zoo 的 OOF 预测与 Test 预测至 cache_10fold/text_zoo_oof/
    text_zoo_dir = CACHE_DIR / 'text_zoo_oof'
    text_zoo_dir.mkdir(parents=True, exist_ok=True)
    for m_name in text_model_names:
        oof_path = text_zoo_dir / f'{m_name}_oof.npy'
        test_path = text_zoo_dir / f'{m_name}_test.npy'
        np.save(oof_path, text_oof_preds[m_name].astype(np.float32))
        np.save(test_path, test_oof_df[m_name].values.astype(np.float32))
    print(f">>> [Text Zoo] 11 路 Text Zoo OOF 与 Test 预测已全部保存至 {text_zoo_dir}")
    print(f">>> [Step 3] 10-Fold OOF 管道执行完成！总耗时: {time.time() - t_start:.2f}s")
    
    return train_oof_df, test_oof_df, cv_summary

# -----------------------------------------------------------------------------
# 6. 主程序与对齐保存
# -----------------------------------------------------------------------------
def main():
    total_t0 = time.time()
    print("=" * 70)
    print("🐱 Neko 顶级特征工程中枢: fe_pipeline_10fold.py")
    print("=" * 70)
    
    # 读取原始数据
    train_csv_path = WORKSPACE_DIR / 'train.csv'
    test_csv_path = WORKSPACE_DIR / 'test.csv'
    
    print(f">>> 读取原始数据: {train_csv_path.name} & {test_csv_path.name}")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    n_train = len(train_df)
    n_test = len(test_df)
    y_log1p = np.log1p(train_df['price'].values).astype(np.float32)
    print(f"    Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"    Target y_log1p stats: min={y_log1p.min():.4f}, mean={y_log1p.mean():.4f}, max={y_log1p.max():.4f}, std={y_log1p.std():.4f}")
    
    # 步骤 1: 基础表格特征工程
    train_base_feat, test_base_feat = build_base_tabular_features(train_df, test_df)
    
    # 显式删除 'price' 列并严格断言
    train_base_feat = train_base_feat.drop(columns=['price'], errors='ignore')
    test_base_feat = test_base_feat.drop(columns=['price'], errors='ignore')
    assert 'price' not in train_base_feat.columns, "Fatal: 'price' found in train_base_feat!"
    assert 'price' not in test_base_feat.columns, "Fatal: 'price' found in test_base_feat!"
    
    # 保存基础表格特征 (train_base_features.parquet 与 test_base_features.parquet)
    print(f">>> 正在落盘基础表格特征至 {CACHE_DIR}...")
    train_base_feat.to_parquet(CACHE_DIR / 'train_base_features.parquet', index=False)
    test_base_feat.to_parquet(CACHE_DIR / 'test_base_features.parquet', index=False)
    
    # 步骤 2: 文本与类别保留
    train_text_df, test_text_df = process_and_cache_text_corpus(train_df, test_df)
    
    # 步骤 3: 10-Fold 严格折内 OOF 特征系统
    train_oof_df, test_oof_df, cv_summary = run_10fold_oof_feature_pipeline(
        train_base_feat, test_base_feat, train_text_df, test_text_df, y_log1p
    )
    
    # 步骤 4: 特征矩阵合并与类型规范化
    print(">>> [Step 4] 正在对齐合并全量表格与 OOF 特征矩阵...")
    # 排除非特征列（如 id, price, 原始文本与时间字符串、复合非数值键）
    exclude_cols = [
        'id', 'price', 'name', 'description', 'amenities', 'bathrooms_text',
        'first_review', 'last_review', 'host_is_superhost', 'host_identity_verified',
        'host_has_profile_pic', 'geo_cell', 'host_id', 'property_type', 'room_type',
        'geo_room_combo', 'prop_room_combo', 'host_room_combo', 'host_location',
        'twin_4tuple', 'twin_5tuple'
    ]
    
    # 严格使用 pd.api.types.is_numeric_dtype 过滤非数值/object 字段，防止 .astype(np.float32) 崩溃
    tab_cols = [
        c for c in train_base_feat.columns 
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_base_feat[c])
    ]
    print(f"    筛选出有效数值表格特征列: {len(tab_cols)} 列 (已严格排除文本与类别标识列)")
    
    # 合并基础数值特征与 OOF 特征
    X_train_final = pd.concat([train_base_feat[tab_cols], train_oof_df], axis=1)
    X_test_final = pd.concat([test_base_feat[tab_cols], test_oof_df], axis=1)
    
    # 缺失值与数值无穷大修剪，确保所有列均为 float32
    X_train_final = X_train_final.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    
    # 保留房源 id 便于下游对齐
    X_train_final.insert(0, 'id', train_df['id'].values)
    X_test_final.insert(0, 'id', test_df['id'].values)
    
    # 彻底杜绝泄漏：从最终特征矩阵中显式删除 price 列并严格断言
    X_train_final = X_train_final.drop(columns=['price'], errors='ignore')
    X_test_final = X_test_final.drop(columns=['price'], errors='ignore')
    assert 'price' not in X_train_final.columns, "Fatal: 'price' found in X_train_final!"
    assert 'price' not in X_test_final.columns, "Fatal: 'price' found in X_test_final!"
    
    # 步骤 5: 落盘 Parquet 与元数据清单
    print(f">>> [Step 5] 正在写入最终特征矩阵至 {CACHE_DIR}...")
    train_feat_path = CACHE_DIR / 'train_features.parquet'
    test_feat_path = CACHE_DIR / 'test_features.parquet'
    
    X_train_final.to_parquet(train_feat_path, index=False)
    X_test_final.to_parquet(test_feat_path, index=False)
    
    feature_names = [c for c in X_train_final.columns if c != 'id']
    
    # 产出 manifest.json
    manifest = {
        'status': 'success',
        'n_train': n_train,
        'n_test': n_test,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'text_zoo_10fold_cv_rmsle': cv_summary,
        'cached_files': [
            'train_base_features.parquet',
            'test_base_features.parquet',
            'train_features.parquet',
            'test_features.parquet',
            'train_text.parquet',
            'test_text.parquet',
            'folds_10.npy',
            'fold_ids.npy',
            'y_log1p.npy',
            'manifest.json',
            'READY.flag',
            'READY_CLEAN.flag'
        ],
        'total_elapsed_seconds': round(time.time() - total_t0, 2)
    }
    
    with open(CACHE_DIR / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    with open(CACHE_DIR / 'READY.flag', 'w', encoding='utf-8') as f:
        f.write("READY\n")
        
    with open(CACHE_DIR / 'READY_CLEAN.flag', 'w', encoding='utf-8') as f:
        f.write("READY_CLEAN\n")
        
    print("=" * 70)
    print(f"🎉 全部 10-Fold 特征基座构建成功！总耗时: {manifest['total_elapsed_seconds']}s")
    print(f"   * 训练集样本数: {n_train}, 测试集样本数: {n_test}")
    print(f"   * 最终特征总数: {len(feature_names)}")
    print(f"   * 全量落盘路径: {CACHE_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()

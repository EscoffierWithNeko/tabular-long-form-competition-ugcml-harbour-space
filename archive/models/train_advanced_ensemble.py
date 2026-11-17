import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

WORKSPACE_DIR = Path(__file__).resolve().parent

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
    # Combine for non-leaking global transformations like categorical mappings and text len
    n_train = len(df_train)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    # 1. Bathrooms parsing
    bath_num, bath_shared = parse_bathrooms(df['bathrooms_text'])
    df['bathrooms_num'] = bath_num
    df['is_shared_bath'] = bath_shared
    
    # 2. Host metadata and experience
    df['host_total_months'] = df['hosts_time_as_user_years'] * 12 + df['hosts_time_as_user_months']
    df['host_as_host_total_months'] = df['hosts_time_as_host_years'] * 12 + df['hosts_time_as_host_months']
    df['host_time_gap_months'] = df['host_total_months'] - df['host_as_host_total_months']
    
    df['host_is_superhost_bin'] = (df['host_is_superhost'] == 't').astype(np.int8)
    df['host_identity_verified_bin'] = (df['host_identity_verified'] == 't').astype(np.int8)
    df['host_has_profile_pic_bin'] = (df['host_has_profile_pic'] == 't').astype(np.int8)
    
    # 3. Capacity ratios & Room combinations
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
    
    # 4. Amenities feature engineering
    print("Parsing amenities...")
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
    
    # 5. Availability rates and dynamics
    df['avail_ratio_30'] = df['availability_30'] / 30.0
    df['avail_ratio_60'] = df['availability_60'] / 60.0
    df['avail_ratio_90'] = df['availability_90'] / 90.0
    df['avail_ratio_365'] = df['availability_365'] / 365.0
    df['avail_diff_30_60'] = df['availability_60'] - df['availability_30']
    df['avail_diff_60_90'] = df['availability_90'] - df['availability_60']
    df['avail_diff_90_365'] = df['availability_365'] - df['availability_90']
    
    # 6. Minimum & Maximum nights engineering
    df['is_short_term'] = (df['minimum_nights'] <= 2).astype(np.int8)
    df['is_monthly_plus'] = (df['minimum_nights'] >= 30).astype(np.int8)
    df['min_nights_range'] = df['maximum_minimum_nights'] - df['minimum_minimum_nights']
    df['max_nights_range'] = df['maximum_maximum_nights'] - df['minimum_maximum_nights']
    df['min_nights_ratio'] = df['minimum_minimum_nights'] / (df['maximum_minimum_nights'] + 1)
    
    # 7. Review dynamics and temporal features
    base_date = pd.to_datetime('2026-11-01')
    first_dt = pd.to_datetime(df['first_review'], errors='coerce')
    last_dt = pd.to_datetime(df['last_review'], errors='coerce')
    
    df['days_since_first_review'] = (base_date - first_dt).dt.days
    df['days_since_last_review'] = (base_date - last_dt).dt.days
    df['review_lifespan_days'] = (last_dt - first_dt).dt.days
    df['review_active_ratio'] = df['number_of_reviews_ltm'] / (df['number_of_reviews'] + 1)
    df['review_month_velocity'] = df['number_of_reviews_l30d'] / (df['number_of_reviews_ltm'] + 1)
    
    # 8. Composite Review Scores
    review_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'
    ]
    df['review_scores_mean'] = df[review_cols].mean(axis=1)
    df['review_scores_std'] = df[review_cols].std(axis=1)
    df['rating_val_interaction'] = df['review_scores_rating'] * df['review_scores_value']
    df['clean_loc_interaction'] = df['review_scores_cleanliness'] * df['review_scores_location']
    
    # 9. Text metadata & Luxury keywords
    name_str = df['name'].fillna('').astype(str).str.lower()
    desc_str = df['description'].fillna('').astype(str).str.lower()
    
    df['name_len'] = name_str.str.len()
    df['name_words'] = name_str.str.split().str.len()
    df['desc_len'] = desc_str.str.len()
    df['desc_words'] = desc_str.str.split().str.len()
    
    luxury_keywords = ['luxury', 'luxurious', 'stunning', 'penthouse', 'spacious', 'modern', 'central', 'designer', 'prime', 'balcony', 'view', 'boutique']
    for kw in luxury_keywords:
        df[f'name_has_{kw}'] = name_str.str.contains(kw, regex=False).astype(np.int8)
        
    budget_keywords = ['cosy', 'cozy', 'budget', 'tiny', 'small', 'student', 'shared', 'room in']
    for kw in budget_keywords:
        clean_col = kw.replace(' ', '_')
        df[f'name_has_{clean_col}'] = name_str.str.contains(kw, regex=False).astype(np.int8)
        
    # 10. Categorical frequency encoding
    cat_cols_freq = ['property_type', 'room_type', 'geo_cell', 'host_id', 'host_location']
    for col in cat_cols_freq:
        freq = df[col].value_counts()
        df[f'{col}_freq'] = df[col].map(freq)
        
    # Categorical label encoding for tree models
    for col in ['room_type', 'property_type', 'host_location']:
        df[col] = df[col].astype('category')
        
    # Drop raw large text and dates
    drop_features = [
        'name', 'description', 'amenities', 'bathrooms_text', 
        'first_review', 'last_review', 'host_is_superhost', 
        'host_identity_verified', 'host_has_profile_pic'
    ]
    df = df.drop(columns=drop_features)
    
    train_out = df.iloc[:n_train].copy().reset_index(drop=True)
    test_out = df.iloc[n_train:].copy().reset_index(drop=True)
    
    return train_out, test_out

def main():
    print("Loading data...")
    train_df = pd.read_csv(WORKSPACE_DIR / 'train.csv')
    test_df = pd.read_csv(WORKSPACE_DIR / 'test.csv')
    
    print(f"Raw Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    train_feat, test_feat = extract_features(train_df, test_df)
    
    y = np.log1p(train_feat['price'].values)
    feature_cols = [c for c in train_feat.columns if c not in ['id', 'price']]
    
    print(f"Extracted {len(feature_cols)} features for model training.")
    print("Sample feature names:", feature_cols[:15])
    
    # KFold Validation with Target Encoding within folds to avoid leakage
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_feat))
    test_preds = np.zeros(len(test_feat))
    
    # LightGBM Parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1
    }
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_feat)):
        X_tr = train_feat.iloc[train_idx][feature_cols].copy()
        y_tr = y[train_idx]
        X_va = train_feat.iloc[val_idx][feature_cols].copy()
        y_va = y[val_idx]
        X_te = test_feat[feature_cols].copy()
        
        # Out-of-fold smoothed Target Encoding for high-cardinality geo_cell & host_id
        for c in ['geo_cell', 'host_id', 'room_type', 'property_type']:
            col_mean = pd.Series(y_tr, index=train_idx).groupby(train_feat.iloc[train_idx][c]).agg(['mean', 'count'])
            global_mean = y_tr.mean()
            weight = 10.0
            smooth_map = (col_mean['mean'] * col_mean['count'] + global_mean * weight) / (col_mean['count'] + weight)
            
            X_tr[f'{c}_target_enc'] = train_feat.iloc[train_idx][c].map(smooth_map).fillna(global_mean)
            X_va[f'{c}_target_enc'] = train_feat.iloc[val_idx][c].map(smooth_map).fillna(global_mean)
            X_te[f'{c}_target_enc'] = test_feat[c].map(smooth_map).fillna(global_mean)
            
        # LightGBM Model
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=100)]
        )
        
        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        
        score = np.sqrt(mean_squared_error(y_va, val_pred))
        fold_scores.append(score)
        print(f"Fold {fold + 1} RMSLE: {score:.5f}")
        
        test_preds += model.predict(X_te) / kf.n_splits
        
    overall_rmsle = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\n{'='*40}")
    print(f"5-Fold CV RMSLE: {overall_rmsle:.5f} (Mean of folds: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f})")
    print(f"Previous Baseline LB: 0.39018")
    print(f"{'='*40}")
    
    # Generate submission
    sub = pd.DataFrame({
        'id': test_feat['id'],
        'price': np.expm1(test_preds)
    })
    # Clip predictions to reasonable bounds
    sub['price'] = sub['price'].clip(lower=15.0, upper=2500.0)
    
    out_path = WORKSPACE_DIR / 'submission_advanced_lgb.csv'
    sub.to_csv(out_path, index=False)
    print(f"Saved prediction to {out_path} (Rows: {len(sub)})")
    print("Prediction stats:")
    print(sub['price'].describe())
    
    if overall_rmsle < 0.3900:
        sub_target = WORKSPACE_DIR / 'submission.csv'
        sub.to_csv(sub_target, index=False)
        print(f"Successfully updated {sub_target} as new best candidate submission!")
        
if __name__ == '__main__':
    main()

import os
import re
import gc
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import subprocess

warnings.filterwarnings('ignore')

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

def build_features(df_train, df_test):
    print("Starting feature engineering on combined train + test dataset...")
    n_train = len(df_train)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    # 1. Bathrooms text parsing
    bath_num, bath_shared = parse_bathrooms(df['bathrooms_text'])
    df['bathrooms_num'] = bath_num
    df['is_shared_bath'] = bath_shared
    
    # 2. Host metadata and experience
    df['host_total_months'] = df['hosts_time_as_user_years'] * 12 + df['hosts_time_as_user_months']
    df['host_as_host_total_months'] = df['hosts_time_as_host_years'] * 12 + df['hosts_time_as_host_months']
    df['host_time_gap_months'] = (df['host_total_months'] - df['host_as_host_total_months']).clip(lower=0)
    
    df['host_is_superhost_bin'] = (df['host_is_superhost'] == 't').astype(np.int8)
    df['host_identity_verified_bin'] = (df['host_identity_verified'] == 't').astype(np.int8)
    df['host_has_profile_pic_bin'] = (df['host_has_profile_pic'] == 't').astype(np.int8)
    
    # 3. Capacity, bedroom, beds interactions
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
    
    # 4. Amenities mining
    amenities_str = df['amenities'].fillna('').astype(str).str.lower()
    df['amenities_count'] = df['amenities'].fillna('').astype(str).apply(lambda x: len(x.split(',')) if len(x) > 2 else 0)
    df['amenities_char_len'] = amenities_str.str.len()
    
    key_amenities = {
        'amen_wifi': ['wifi', 'wi-fi', 'internet'],
        'amen_kitchen': ['kitchen'],
        'amen_pool': ['pool'],
        'amen_hot_tub': ['hot tub', 'jacuzzi'],
        'amen_gym': ['gym'],
        'amen_balcony': ['patio', 'balcony', 'terrace'],
        'amen_ac': ['air conditioning', 'ac'],
        'amen_washer': ['washer', 'washing machine'],
        'amen_dryer': ['dryer'],
        'amen_dishwasher': ['dishwasher'],
        'amen_elevator': ['elevator', 'lift'],
        'amen_parking': ['parking', 'garage'],
        'amen_workspace': ['dedicated workspace', 'desk'],
        'amen_tv': ['tv', 'television', 'hdtv'],
        'amen_bathtub': ['bathtub', 'bath tub'],
        'amen_bbq': ['bbq', 'barbecue'],
        'amen_waterfront': ['waterfront', 'beachfront', 'lake access'],
        'amen_self_checkin': ['self check-in', 'lockbox', 'keypad']
    }
    for col_name, kw_list in key_amenities.items():
        pattern = '|'.join(kw_list)
        df[col_name] = amenities_str.str.contains(pattern, regex=True).astype(np.int8)
        
    # 5. Listing name NLP keyword mining
    name_str = df['name'].fillna('').astype(str).str.lower()
    df['name_len'] = name_str.str.len()
    df['name_word_count'] = name_str.str.split().str.len()
    
    name_keywords = {
        'name_luxury': 'luxur',
        'name_view': 'view',
        'name_cozy': 'cozy',
        'name_studio': 'studio',
        'name_villa': 'villa',
        'name_penthouse': 'penthouse',
        'name_downtown': 'downtown',
        'name_beach': 'beach',
        'name_modern': 'modern',
        'name_spacious': 'spacious',
        'name_central': 'central',
        'name_private': 'private'
    }
    for k, word in name_keywords.items():
        df[k] = name_str.str.contains(word, regex=False).astype(np.int8)
        
    # 6. Description lengths
    desc_str = df['description'].fillna('').astype(str)
    df['desc_char_len'] = desc_str.str.len()
    df['desc_word_count'] = desc_str.str.split().str.len()
    
    # 7. Review temporal features
    now_date = pd.to_datetime('2024-01-01')
    df['first_review_dt'] = pd.to_datetime(df['first_review'], errors='coerce')
    df['last_review_dt'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['days_since_first_review'] = (now_date - df['first_review_dt']).dt.days
    df['days_since_last_review'] = (now_date - df['last_review_dt']).dt.days
    df['review_span_days'] = (df['last_review_dt'] - df['first_review_dt']).dt.days
    df.drop(columns=['first_review_dt', 'last_review_dt'], inplace=True)
    
    # 8. Review scores composite aggregations
    score_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value'
    ]
    df['score_mean'] = df[score_cols].mean(axis=1)
    df['score_min'] = df[score_cols].min(axis=1)
    df['score_max'] = df[score_cols].max(axis=1)
    df['score_std'] = df[score_cols].std(axis=1)
    df['rating_x_reviews'] = df['review_scores_rating'] * np.log1p(df['number_of_reviews'].clip(lower=0))
    
    # 9. Availability ratios & occupancy signals
    df['avail_rate_30'] = df['availability_30'] / 30.0
    df['avail_rate_60'] = df['availability_60'] / 60.0
    df['avail_rate_90'] = df['availability_90'] / 90.0
    df['avail_rate_365'] = df['availability_365'] / 365.0
    df['avail_demand_signal'] = df['availability_30'] / (df['availability_365'] / 12.0 + 1.0)
    
    # 10. Nights bounds
    df['min_nights_gap'] = df['maximum_minimum_nights'] - df['minimum_minimum_nights']
    df['max_nights_gap'] = df['maximum_maximum_nights'] - df['minimum_maximum_nights']
    
    # 11. Geo Cell & Property Type frequency encoding
    for c in ['geo_cell', 'property_type', 'room_type', 'host_id']:
        freq = df[c].value_counts()
        df[f'{c}_freq'] = df[c].map(freq).astype(np.float32)
        
    # 12. Geographic Group Aggregations (Location context is paramount in Real Estate)
    print("Computing geo group statistics...")
    geo_stats = df.groupby('geo_cell').agg({
        'accommodates': ['mean', 'max'],
        'bedrooms_safe': ['mean'],
        'review_scores_rating': ['mean'],
        'availability_30': ['mean']
    })
    geo_stats.columns = [f'geo_{col}_{stat}' for col, stat in geo_stats.columns]
    df = df.merge(geo_stats, on='geo_cell', how='left')
    
    # Convert categoricals to category dtype
    cat_cols = ['property_type', 'room_type', 'host_location', 'geo_cell', 'host_id']
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    # Drop text and date columns
    drop_cols = ['name', 'description', 'bathrooms_text', 'amenities', 'first_review', 'last_review',
                 'host_is_superhost', 'host_identity_verified', 'host_has_profile_pic']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    # Separate train and test
    train_out = df.iloc[:n_train].copy().reset_index(drop=True)
    test_out = df.iloc[n_train:].copy().reset_index(drop=True)
    
    return train_out, test_out

def main():
    start_time = time.time()
    train_path = WORKSPACE_DIR / 'train.csv'
    test_path = WORKSPACE_DIR / 'test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_feat, test_feat = build_features(train_df, test_df)
    
    y = np.log1p(train_feat['price'].values)
    feature_cols = [c for c in train_feat.columns if c not in ['id', 'price']]
    print(f"Total features engineered: {len(feature_cols)}")
    
    # Out-of-Fold Target Encoding on high-cardinality columns
    te_cols = ['geo_cell', 'property_type', 'room_type']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_feat), dtype=np.float64)
    test_preds = np.zeros(len(test_feat), dtype=np.float64)
    fold_scores = []
    
    params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.035,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 3000,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    print("Starting 5-Fold Stratified-Free KFold LightGBM Training...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_feat)):
        X_tr = train_feat.iloc[train_idx][feature_cols].copy()
        y_tr = y[train_idx]
        X_va = train_feat.iloc[val_idx][feature_cols].copy()
        y_va = y[val_idx]
        X_te = test_feat[feature_cols].copy()
        
        # Out-of-fold smooth target encoding
        global_mean = float(y_tr.mean())
        smoothing = 20.0
        for c in te_cols:
            cat_series = train_feat.iloc[train_idx][c].astype(str)
            stats = pd.DataFrame({'val': y_tr, 'cat': cat_series}).groupby('cat')['val'].agg(['mean', 'count'])
            smooth_map = ((stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)).to_dict()
            
            enc_tr = cat_series.map(smooth_map).fillna(global_mean).values.astype(np.float32)
            enc_va = train_feat.iloc[val_idx][c].astype(str).map(smooth_map).fillna(global_mean).values.astype(np.float32)
            enc_te = test_feat[c].astype(str).map(smooth_map).fillna(global_mean).values.astype(np.float32)
            
            X_tr[f'{c}_target_enc'] = enc_tr
            X_va[f'{c}_target_enc'] = enc_va
            X_te[f'{c}_target_enc'] = enc_te
            
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=100)]
        )
        
        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        
        score = np.sqrt(mean_squared_error(y_va, val_pred))
        fold_scores.append(score)
        print(f"Fold {fold + 1} RMSLE: {score:.5f}")
        
        test_preds += model.predict(X_te) / kf.n_splits
        
    overall_rmsle = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\n{'='*50}")
    print(f"5-Fold CV RMSLE: {overall_rmsle:.5f} (Mean of folds: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f})")
    print(f"Baseline Official LB: 0.39018")
    print(f"Delta Gain on CV: {0.39018 - overall_rmsle:+.5f} (Lower is better)")
    print(f"{'='*50}")
    
    # Generate submission file
    sub = pd.DataFrame({
        'id': test_feat['id'],
        'price': np.expm1(test_preds)
    })
    # Post-processing clip bounds
    sub['price'] = sub['price'].clip(lower=15.0, upper=2500.0)
    
    sub_path = WORKSPACE_DIR / 'submission.csv'
    sub.to_csv(sub_path, index=False)
    print(f"Saved final submission to {sub_path} (Rows: {len(sub)})")
    print("Submission Summary Stats:")
    print(sub['price'].describe())
    
    # Execute Kaggle Submission
    comp_name = 'tabular-long-form-competition-ugcml-harbour-space'
    msg = f'Advanced Feature-Eng + OOF Target-Enc 5-Fold LGBM (CV RMSLE: {overall_rmsle:.5f})'
    cmd_submit = f'kaggle competitions submit -c {comp_name} -f "{sub_path}" -m "{msg}"'
    print(f"\nSubmitting to Kaggle with message: '{msg}'...")
    sub_res = subprocess.run(cmd_submit, shell=True, capture_output=True, text=True)
    print("Submit stdout:", sub_res.stdout)
    print("Submit stderr:", sub_res.stderr)
    
    print("Fetching latest submission history from Kaggle...")
    time.sleep(5)
    cmd_list = f'kaggle competitions submissions -c {comp_name}'
    list_res = subprocess.run(cmd_list, shell=True, capture_output=True, text=True)
    print("Submissions list:")
    print(list_res.stdout)
    print(f"Total pipeline finished in {time.time() - start_time:.1f} seconds.")

if __name__ == '__main__':
    main()

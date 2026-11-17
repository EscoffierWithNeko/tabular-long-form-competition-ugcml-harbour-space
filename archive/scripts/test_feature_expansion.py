import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("=== Testing NLP SVD and High-Order Feature Extraction on sample ===")
t0 = time.time()
df = pd.read_csv('train.csv', nrows=10000)

# 1. Name TF-IDF + SVD
print("1. Testing Name TF-IDF + SVD...")
name_tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
name_vecs = name_tfidf.fit_transform(df['name'].fillna('').astype(str))
svd_name = TruncatedSVD(n_components=8, random_state=42)
name_svd = svd_name.fit_transform(name_vecs)
print(f"Name SVD shape: {name_svd.shape}, elapsed: {time.time() - t0:.2f}s")

# 2. Description TF-IDF + SVD
print("2. Testing Description TF-IDF + SVD...")
t1 = time.time()
desc_tfidf = TfidfVectorizer(max_features=3000, stop_words='english', min_df=3)
desc_vecs = desc_tfidf.fit_transform(df['description'].fillna('').astype(str))
svd_desc = TruncatedSVD(n_components=12, random_state=42)
desc_svd = svd_desc.fit_transform(desc_vecs)
print(f"Desc SVD shape: {desc_svd.shape}, elapsed: {time.time() - t1:.2f}s")

# 3. Amenities TF-IDF + SVD
print("3. Testing Amenities TF-IDF + SVD...")
t2 = time.time()
amen_tfidf = TfidfVectorizer(token_pattern=r'"([^"]+)"', max_features=1000, min_df=5)
amen_vecs = amen_tfidf.fit_transform(df['amenities'].fillna('[]').astype(str))
svd_amen = TruncatedSVD(n_components=10, random_state=42)
amen_svd = svd_amen.fit_transform(amen_vecs)
print(f"Amenities SVD shape: {amen_svd.shape}, elapsed: {time.time() - t2:.2f}s")

# 4. Geo-cell higher order aggregations
print("4. Testing Geo-cell aggregations...")
t3 = time.time()
geo_stats = df.groupby('geo_cell').agg(
    geo_accommodates_mean=('accommodates', 'mean'),
    geo_accommodates_std=('accommodates', 'std'),
    geo_bedrooms_mean=('bedrooms', 'mean'),
    geo_reviews_mean=('number_of_reviews', 'mean'),
    geo_count=('accommodates', 'count')
).reset_index()
print(f"Geo stats shape: {geo_stats.shape}, elapsed: {time.time() - t3:.2f}s")
print(f"Total pipeline test completed in {time.time() - t0:.2f}s successfully!")

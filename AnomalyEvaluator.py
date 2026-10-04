import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf # <--- Use the whole library
from sklearn.metrics import silhouette_score
from FeatureEngineering import df

# SETUP PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
iso_path = os.path.join(script_dir, 'iso_forest_model.pkl')
ae_path = os.path.join(script_dir, 'autoencoder_model.h5')

# Load Data and Models
X = df.select_dtypes(include=[np.number]).values
iso_forest = joblib.load(iso_path)

# FIXED LINE: We use tf.keras.models.load_model instead of just load_model
autoencoder = tf.keras.models.load_model(ae_path, compile=False)  

print("🔍 Starting Model Evaluation...")
print("="*30)

if_preds = iso_forest.predict(X)
sample_idx = np.random.choice(len(X), 5000, replace=False)
sil_score = silhouette_score(X[sample_idx], if_preds[sample_idx])

print(f"✅ Isolation Forest Silhouette Score: {sil_score:.4f}")
print(f"✅ Total Anomalies Detected by IF: {np.sum(if_preds == -1)}")

reconstructed = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 99)
anomalies_ae = mse > threshold

print(f"✅ Autoencoder Anomaly Threshold: {threshold:.4f}")
print(f"✅ Total Anomalies Detected by AE: {np.sum(anomalies_ae)}")

print("="*30)
print("EVALUATION COMPLETE")
print("="*30)

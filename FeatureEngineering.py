import pandas as pd
import numpy as np
# This fetches the cleaned df from your Preprocessing.py file
from Preprocessing import df

print("✅ Data fetched from Preprocessing.py")

# --- 1. Rolling Mean (Moving Average) ---
# window=60 means it looks at the last 60 rows to calculate the average
df['TP2_rolling_mean'] = df['TP2'].rolling(window=60).mean()

# --- 2. Rolling Standard Deviation ---
# This shows how much the temperature is fluctuating
df['TP2_rolling_std'] = df['TP2'].rolling(window=60).std()

# --- 3. Temperature Variation ---
# .diff() calculates the difference between current row and previous row
df['TP2_variation'] = df['TP2'].diff()

# --- 4. Pressure Difference (Sensor Gap) ---
# We check if TP3 and Reservoirs exist, otherwise we create a gap based on TP2
if 'TP3' in df.columns and 'Reservoirs' in df.columns:
    df['pressure_gap'] = df['TP3'] - df['Reservoirs']
else:
    # Fallback: gap between current temp and its moving average
    df['pressure_gap'] = df['TP2'] - df['TP2_rolling_mean']

# --- 5. Motor Load States ---
# If both Pressure and Flow are 1, it is "High Load" (1), otherwise (0)
df['motor_load_state'] = ((df['Pressure_switch'] == 1) & (df['Caudal_impulses'] == 1)).astype(int)

# --- 6. Moving Averages ---
# Adding a slower moving average (window=120) to see long-term trends
df['TP2_long_avg'] = df['TP2'].rolling(window=120).mean()

# --- FINAL CLEANUP ---
# Rolling windows create 'NaN' (empty) values at the very beginning. 
# We fill them so the Machine Learning model doesn't crash.
df.fillna(method='bfill', inplace=True)

print("✅ Feature Engineering Complete!")
print("="*30)
print(f"Final Dataset Shape: {df.shape}")
print("\nFirst 5 rows with new features:")
print(df[['TP2', 'TP2_rolling_mean', 'TP2_variation', 'motor_load_state']].head())

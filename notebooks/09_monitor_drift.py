import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os

GOLD_DIR = os.path.join("data", "gold")

print("Loading baseline clinical data...")
# Load the original data your model was trained on
df_baseline = pd.read_parquet(os.path.join(GOLD_DIR, "gold_vitals_hourly.parquet"))

# We will monitor Heart Rate for this example
# Drop NaNs to ensure clean mathematical comparison
baseline_hr = df_baseline['Heart Rate'].dropna().values

print("Simulating a new patient data stream (e.g., a novel pathogen outbreak)...")
# Let's artificially create data drift
# We simulate a scenario where incoming patients have resting heart rates 15 BPM higher than normal.
new_patient_hr = baseline_hr + np.random.normal(loc=15.0, scale=5.0, size=len(baseline_hr))

print("\nRunning Kolmogorov-Smirnov (KS) Test for Data Drift...")
# The KS test compares the underlying shapes of two data distributions
statistic, p_value = ks_2samp(baseline_hr, new_patient_hr)

print(f"KS Statistic: {statistic:.4f}")
print(f"P-Value: {p_value:.4e}")

# A p-value less than 0.05 generally means the distributions are statistically different
if p_value < 0.05:
    print("\n[CRITICAL ALERT] SIGNIFICANT DATA DRIFT DETECTED!")
    print("The physiological baseline of the ICU population has shifted.")
    print("Action Required: Triggering automated pipeline to retrain the Dual-Branch LSTM.")
else:
    print("\n[OK] Data distribution is stable. No retraining needed.")
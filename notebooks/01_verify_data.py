import pandas as pd
import os

# 1. Adjusted path since you are running the script from the root directory
BRONZE_DIR = os.path.join("data", "bronze") 

# 2. Added the .gz extensions to match your files
CHARTEVENTS_PATH = os.path.join(BRONZE_DIR, "icu", "chartevents.csv.gz")
ADMISSIONS_PATH = os.path.join(BRONZE_DIR, "hosp", "admissions.csv.gz")

print("--- Checking ICU Chart Events (Vitals) ---")
try:
    # Pandas will automatically decompress the .gz file on the fly!
    df_chart = pd.read_csv(CHARTEVENTS_PATH, nrows=5)
    print("Success! Here are the first 5 rows:")
    print(df_chart[['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']].head())
except Exception as e:
    print(f"Error loading chartevents: {e}")

print("\n--- Checking Hospital Admissions ---")
try:
    df_adm = pd.read_csv(ADMISSIONS_PATH, nrows=5)
    print("Success! Here are the first 5 rows:")
    print(df_adm[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type']].head())
except Exception as e:
    print(f"Error loading admissions: {e}")
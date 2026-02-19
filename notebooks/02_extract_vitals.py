import pandas as pd
import os

# 1. Setup our Data Lake directories
BRONZE_DIR = os.path.join("data", "bronze")
SILVER_DIR = os.path.join("data", "silver")
os.makedirs(SILVER_DIR, exist_ok=True) # This creates the silver folder automatically!

CHARTEVENTS_PATH = os.path.join(BRONZE_DIR, "icu", "chartevents.csv.gz")

# 2. Define the exact vitals we want to extract
TARGET_VITALS = {
    220045: "Heart Rate",
    220181: "Mean Blood Pressure",
    220277: "O2 Saturation",
    220210: "Respiratory Rate",
    223761: "Temperature Fahrenheit",
    223762: "Temperature Celsius"
}

print("Loading and filtering chartevents... (This might take a few seconds)")

# 3. Load the raw bronze data
df_chart = pd.read_csv(CHARTEVENTS_PATH)

# 4. Filter out everything except our target vitals
df_vitals = df_chart[df_chart['itemid'].isin(TARGET_VITALS.keys())].copy()

# 5. Map the numerical itemid to human-readable names
df_vitals['vital_name'] = df_vitals['itemid'].map(TARGET_VITALS)

# 6. Clean up: Keep only necessary columns and drop rows with missing values
columns_to_keep = ['subject_id', 'hadm_id', 'charttime', 'vital_name', 'valuenum']
df_vitals = df_vitals[columns_to_keep].dropna(subset=['valuenum'])

# 7. Save to the Silver layer as a Parquet file
output_path = os.path.join(SILVER_DIR, "filtered_vitals.parquet")
df_vitals.to_parquet(output_path, index=False)

print(f"\nSuccess! Filtered vitals saved to {output_path}")
print(f"Total vital signs extracted: {len(df_vitals)}")
print("\nSample of the cleaned data:")
print(df_vitals.head())
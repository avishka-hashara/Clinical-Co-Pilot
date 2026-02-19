import pandas as pd
import os

SILVER_DIR = os.path.join("data", "silver")
GOLD_DIR = os.path.join("data", "gold")
os.makedirs(GOLD_DIR, exist_ok=True) # Create the Gold folder

# 1. Load the Silver data
silver_path = os.path.join(SILVER_DIR, "filtered_vitals.parquet")
df_vitals = pd.read_parquet(silver_path)

# 2. Ensure charttime is treated as actual time, not just text
df_vitals['charttime'] = pd.to_datetime(df_vitals['charttime'])

print("Pivoting data from long to wide...")
# 3. Pivot the data
df_wide = df_vitals.pivot_table(
    index=['subject_id', 'hadm_id', 'charttime'],
    columns='vital_name',
    values='valuenum',
    aggfunc='mean' 
).reset_index()

# 4. Resample to 1-hour intervals and Forward-Fill
def resample_patient_data(group):
    # Set time as the index so pandas can manipulate it
    group = group.set_index('charttime')
    
    # Force the timeline into 1-Hour ('1h') buckets -> CHANGED TO LOWERCASE 'h'
    resampled = group.resample('1h').mean()
    
    # Forward-fill: carry the last known measurement forward into empty buckets
    resampled = resampled.ffill()
    
    # Backward-fill: just in case the very first hour is missing a vital, look ahead to fill it
    resampled = resampled.bfill()
    return resampled

print("Resampling to 1-hour buckets and applying forward-fill imputation. Please wait...")

# Group by each hospital admission and apply our function
# Note: In newer pandas, apply with a function returning a DataFrame drops the group keys if include_groups=False is not passed, 
# but this standard approach should still safely group your data for now.
df_gold = df_wide.groupby(['subject_id', 'hadm_id']).apply(resample_patient_data).reset_index()

# 5. Save to the Gold layer
output_path = os.path.join(GOLD_DIR, "gold_vitals_hourly.parquet")
df_gold.to_parquet(output_path, index=False)

print(f"\nSuccess! Hourly aggregated vitals saved to {output_path}")
print("\nSample of the Gold data (Notice the perfect 1-hour jumps!):")
cols_to_show = ['subject_id', 'hadm_id', 'charttime']
if 'Heart Rate' in df_gold.columns: cols_to_show.append('Heart Rate')
if 'Respiratory Rate' in df_gold.columns: cols_to_show.append('Respiratory Rate')

print(df_gold[cols_to_show].head())
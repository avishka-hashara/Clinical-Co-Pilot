import pandas as pd
import numpy as np
import os

GOLD_DIR = os.path.join("data", "gold")

print("Loading Gold layer datasets...")
df_vitals = pd.read_parquet(os.path.join(GOLD_DIR, "gold_vitals_hourly.parquet"))
df_notes = pd.read_parquet(os.path.join(GOLD_DIR, "gold_notes_embedded.parquet"))

df_notes['charttime'] = pd.to_datetime(df_notes['charttime']).astype('datetime64[us]')
df_vitals['charttime'] = pd.to_datetime(df_vitals['charttime']).astype('datetime64[us]')

# 1. Synthesize a "Deterioration" Target Label
# For this architecture build, we'll flag deterioration if Heart Rate > 100
print("Synthesizing target labels...")
if 'Heart Rate' in df_vitals.columns:
    df_vitals['target_deterioration'] = (df_vitals['Heart Rate'] > 100).astype(int)
else:
    df_vitals['target_deterioration'] = 0

# 2. Merge Vitals and Notes
# Left join ensures we keep every hour of vitals, even if there isn't a note
print("Aligning multimodal data...")
df_merged = pd.merge(df_vitals, df_notes[['subject_id', 'hadm_id', 'charttime', 'embedding']], 
                     on=['subject_id', 'hadm_id', 'charttime'], how='left')

# 3. Handle Missing Notes (Pad with zeros)
zero_embedding = np.zeros(768).tolist()
df_merged['embedding'] = df_merged['embedding'].apply(lambda x: x if isinstance(x, list) else zero_embedding)

# 4. Create Sliding Windows (Sequences)
SEQUENCE_LENGTH = 6 # Look at 6 hours of data to predict the next hour

def create_sequences(patient_data):
    patient_data = patient_data.sort_values('charttime')
    
    # Extract vitals (filling any accidental NaNs with 0)
    vital_cols = [col for col in ['Heart Rate', 'Respiratory Rate'] if col in patient_data.columns]
    vitals = patient_data[vital_cols].fillna(0).values 
    
    notes = np.stack(patient_data['embedding'].values)
    targets = patient_data['target_deterioration'].values
    
    X_vitals, X_notes, y = [], [], []
    
    # Slide the window across the patient's timeline
    for i in range(len(patient_data) - SEQUENCE_LENGTH):
        X_vitals.append(vitals[i : i + SEQUENCE_LENGTH])
        X_notes.append(notes[i : i + SEQUENCE_LENGTH])
        y.append(targets[i + SEQUENCE_LENGTH]) 
        
    return X_vitals, X_notes, y

print(f"Building {SEQUENCE_LENGTH}-hour sequence windows...")
all_X_vitals, all_X_notes, all_y = [], [], []

for (subj, hadm), group in df_merged.groupby(['subject_id', 'hadm_id']):
    if len(group) > SEQUENCE_LENGTH:
        xv, xn, y = create_sequences(group)
        all_X_vitals.extend(xv)
        all_X_notes.extend(xn)
        all_y.extend(y)

# 5. Convert to PyTorch-ready NumPy arrays
all_X_vitals = np.array(all_X_vitals, dtype=np.float32)
all_X_notes = np.array(all_X_notes, dtype=np.float32)
all_y = np.array(all_y, dtype=np.float32)

print("\n--- Final Dataset Shapes ---")
print(f"Total sequences created: {len(all_y)}")
print(f"Vitals Matrix (Batch, Sequence, Features): {all_X_vitals.shape}")
print(f"Notes Matrix (Batch, Sequence, Embeddings): {all_X_notes.shape}")
print(f"Targets Vector: {all_y.shape}")

# 6. Save the arrays
np.save(os.path.join(GOLD_DIR, "X_vitals.npy"), all_X_vitals)
np.save(os.path.join(GOLD_DIR, "X_notes.npy"), all_X_notes)
np.save(os.path.join(GOLD_DIR, "y_target.npy"), all_y)

print("\nSuccess! Sequence arrays saved to the Gold directory.")
import pandas as pd
import random
import os

GOLD_DIR = os.path.join("data", "gold")
BRONZE_NOTE_DIR = os.path.join("data", "bronze", "note")
os.makedirs(BRONZE_NOTE_DIR, exist_ok=True) # Create the note folder

print("Loading Gold vitals to map patient IDs...")
df_gold = pd.read_parquet(os.path.join(GOLD_DIR, "gold_vitals_hourly.parquet"))

# Extract unique admissions and the timestamps of their vitals
admissions = df_gold.groupby(['subject_id', 'hadm_id'])['charttime'].apply(list).reset_index()

# A dictionary of synthetic notes ranging from routine to critical
clinical_vocab = [
    "Patient is alert and oriented x3. No acute distress noted.",
    "Patient complains of mild shortness of breath (SOB). O2 saturation stable on room air.",
    "NURSE NOTE: Patient appears newly confused. Blood pressure trending downward. Notifying attending physician.",
    "Radiology report: Clear lungs, no acute cardiopulmonary process.",
    "Lab results reviewed. Lactate slightly elevated at 2.1. Will continue to monitor.",
    "Patient resting comfortably. IV fluids running at 75cc/hr.",
    "CRITICAL: Suspected sepsis protocol initiated. Patient hypotensive and tachycardic. Blood cultures drawn.",
    "Patient denies pain. Tolerating oral diet well."
]

print("Generating synthetic clinical notes...")
synthetic_notes = []

# Loop through our real patients and give them fake notes
for _, row in admissions.iterrows():
    subject_id = row['subject_id']
    hadm_id = row['hadm_id']
    times = row['charttime']
    
    # Give each patient 1 to 3 random notes during their hospital stay
    num_notes = random.randint(1, 3)
    for _ in range(num_notes):
        note_time = random.choice(times) # Pick a random time that matches their actual vitals timeline
        note_text = random.choice(clinical_vocab)
        
        synthetic_notes.append({
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'charttime': note_time,
            'text': note_text
        })

df_notes = pd.DataFrame(synthetic_notes)

# Sort them chronologically just like a real database
df_notes = df_notes.sort_values(by=['subject_id', 'hadm_id', 'charttime'])

output_path = os.path.join(BRONZE_NOTE_DIR, "synthetic_discharge.csv")
df_notes.to_csv(output_path, index=False)

print(f"\nSuccess! Generated {len(df_notes)} synthetic clinical notes.")
print(f"Saved to: {output_path}")
print("\nSample of your new text data:")
print(df_notes.head())
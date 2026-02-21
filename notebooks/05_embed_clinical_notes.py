import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

# 1. Setup paths
BRONZE_NOTE_DIR = os.path.join("data", "bronze", "note")
GOLD_DIR = os.path.join("data", "gold")
os.makedirs(GOLD_DIR, exist_ok=True)

# 2. Load the synthetic notes we just generated
input_path = os.path.join(BRONZE_NOTE_DIR, "synthetic_discharge.csv")
df_notes = pd.read_csv(input_path)

print(f"Loaded {len(df_notes)} notes. Setting up ClinicalBERT...")

# 3. Setup Device (Routing calculations to your RTX 4050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4. Load ClinicalBERT Tokenizer and Model
# We use emilyalsentzer/Bio_ClinicalBERT as it is optimized for medical text
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Put model in evaluation mode (we are using it for inference, not training right now)
model.eval()

# 5. Function to generate embeddings
def get_embedding(text):
    # Tokenize the text (convert words to BERT's internal ID numbers)
    # truncation=True and max_length=512 ensures we don't exceed the model's token limit
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad(): # Disable gradient calculation to save VRAM
        outputs = model(**inputs)
        
    # We extract the embedding of the [CLS] token (the first token), which acts as a summary of the whole sentence
    # .squeeze() removes extra dimensions, .cpu() moves it off the GPU to standard RAM, converting to a list for Parquet compatibility
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()
    return cls_embedding

print("Generating embeddings... Pushing data through the GPU.")

# 6. Apply the embedding function to all notes (using tqdm to show a progress bar)
tqdm.pandas(desc="Embedding Notes")
df_notes['embedding'] = df_notes['text'].progress_apply(get_embedding)

# 7. Save the embedded notes to the Gold layer
output_path = os.path.join(GOLD_DIR, "gold_notes_embedded.parquet")
df_notes.to_parquet(output_path, index=False)

print(f"\nSuccess! Embedded notes saved to {output_path}")
print(f"Embedding vector size: {len(df_notes['embedding'].iloc[0])} dimensions (This is the 'mathematical meaning' of the note!)")
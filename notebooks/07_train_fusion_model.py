import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

GOLD_DIR = os.path.join("data", "gold")

# 1. Setup Device (Utilizing your RTX 4050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Load the Data
print("Loading sequence arrays...")
X_vitals = np.load(os.path.join(GOLD_DIR, "X_vitals.npy"))
X_notes = np.load(os.path.join(GOLD_DIR, "X_notes.npy"))
y_target = np.load(os.path.join(GOLD_DIR, "y_target.npy"))

# Convert NumPy arrays to PyTorch Tensors
# We use unsqueeze(1) on y to change shape from [Batch] to [Batch, 1] for the loss function
tensor_vitals = torch.tensor(X_vitals, dtype=torch.float32)
tensor_notes = torch.tensor(X_notes, dtype=torch.float32)
tensor_y = torch.tensor(y_target, dtype=torch.float32).unsqueeze(1)

# Create a DataLoader to feed data to the GPU in small batches
# A batch size of 32 is very safe for 6GB VRAM
dataset = TensorDataset(tensor_vitals, tensor_notes, tensor_y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define the Multimodal Architecture
class ClinicalCoPilotModel(nn.Module):
    def __init__(self, vitals_input_size, notes_input_size, hidden_size=64):
        super(ClinicalCoPilotModel, self).__init__()
        
        # Branch A: Vitals LSTM
        self.vitals_lstm = nn.LSTM(input_size=vitals_input_size, hidden_size=hidden_size, batch_first=True)
        
        # Branch B: Notes LSTM (We reduce the 768d embedding down to 64d to save memory and match vitals)
        self.notes_reduction = nn.Linear(notes_input_size, 128)
        self.notes_lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)
        
        # Fusion Head: Takes the 64 from vitals + 64 from notes = 128
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(64, 1),
            nn.Sigmoid() # Squashes output between 0 and 1 (Probability)
        )

    def forward(self, vitals_seq, notes_seq):
        # Process Vitals (We only want the final hidden state of the sequence)
        _, (vitals_hidden, _) = self.vitals_lstm(vitals_seq)
        vitals_out = vitals_hidden[-1] # Shape: [Batch, 64]
        
        # Process Notes
        reduced_notes = torch.relu(self.notes_reduction(notes_seq))
        _, (notes_hidden, _) = self.notes_lstm(reduced_notes)
        notes_out = notes_hidden[-1] # Shape: [Batch, 64]
        
        # Fuse the modalities
        fused = torch.cat((vitals_out, notes_out), dim=1) # Shape: [Batch, 128]
        
        # Classify
        risk_score = self.classifier(fused)
        return risk_score

# 4. Initialize Model, Loss, and Optimizer
vitals_features = X_vitals.shape[2] # Usually 2 (Heart Rate, Respiratory Rate)
notes_features = X_notes.shape[2]   # 768 (BioBERT embeddings)

model = ClinicalCoPilotModel(vitals_features, notes_features).to(device)

# Binary Cross Entropy Loss (Standard for Yes/No predictions)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. The Training Loop
epochs = 10
print("\nStarting Training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_vitals, batch_notes, batch_y in dataloader:
        # Move batch to GPU
        batch_vitals = batch_vitals.to(device)
        batch_notes = batch_notes.to(device)
        batch_y = batch_y.to(device)
        
        # Forward Pass
        optimizer.zero_grad() # Clear old calculations
        predictions = model(batch_vitals, batch_notes)
        
        # Calculate Error
        loss = criterion(predictions, batch_y)
        
        # Backward Pass (Learn from the error)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# 6. Save the trained brain
torch.save(model.state_dict(), os.path.join(GOLD_DIR, "clinical_copilot_weights.pth"))
print("\nSuccess! Model trained and weights saved.")
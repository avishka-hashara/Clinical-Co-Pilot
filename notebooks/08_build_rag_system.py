import pandas as pd
import chromadb
import torch
from transformers import pipeline
import os
import warnings

# Suppress some standard HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore")

BRONZE_NOTE_DIR = os.path.join("data", "bronze", "note")

print("Initializing Vector Database (ChromaDB)...")
# Create an in-memory Chroma client for rapid prototyping
chroma_client = chromadb.Client()

# Create collections for our two distinct knowledge sources
patient_collection = chroma_client.create_collection(name="patient_history")
medical_collection = chroma_client.create_collection(name="medical_literature")

# 1. Load Patient Notes into Vector DB
print("Loading patient history into RAG...")
notes_df = pd.read_csv(os.path.join(BRONZE_NOTE_DIR, "synthetic_discharge.csv"))

# We index a sample of notes so the LLM can "read" the patient's chart
sample_notes = notes_df.head(50)
patient_collection.add(
    documents=sample_notes['text'].tolist(),
    metadatas=[{"subject_id": str(row['subject_id']), "charttime": str(row['charttime'])} for _, row in sample_notes.iterrows()],
    ids=[f"note_{i}" for i in range(len(sample_notes))]
)

# 2. Add Mock PubMed Guidelines into Vector DB
print("Loading medical literature into RAG...")
pubmed_docs = [
    "Sepsis-3 Guidelines: Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection. Key indicators include elevated lactate (>2 mmol/L) and hypotension.",
    "Hypotension Management: Initial management of hypotensive patients includes intravenous fluid resuscitation.",
    "Altered Mental Status: Acute confusion in ICU patients can be an early indicator of hypoperfusion and impending sepsis."
]
medical_collection.add(
    documents=pubmed_docs,
    metadatas=[{"source": "PubMed", "topic": "Sepsis"}, {"source": "PubMed", "topic": "Hypotension"}, {"source": "PubMed", "topic": "Neurology"}],
    ids=["doc_1", "doc_2", "doc_3"]
)

# 3. Load the Local LLM
print("\nLoading local LLM (TinyLlama)... This will download the model files the first time.")
# We use bfloat16 precision to strictly manage VRAM usage
llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# 4. The RAG Function
def generate_clinical_explanation(patient_symptoms, risk_score):
    print(f"\n--- RAG Triggered: Patient Risk Score {risk_score:.2f} ---")
    
    # Step A: Retrieve similar patient history
    patient_results = patient_collection.query(query_texts=[patient_symptoms], n_results=1)
    retrieved_history = patient_results['documents'][0][0]
    
    # Step B: Retrieve relevant medical literature (guidelines)
    medical_results = medical_collection.query(query_texts=[patient_symptoms], n_results=1)
    retrieved_literature = medical_results['documents'][0][0]
    
    # Step C: Construct the Prompt with the retrieved context
    prompt = f"""<|system|>
You are a helpful clinical AI assistant. Explain the patient's deterioration risk based on their history and medical literature. Keep it brief and professional.
<|user|>
Risk Score: {risk_score}
Current Symptoms: {patient_symptoms}
Patient History: {retrieved_history}
Medical Reference: {retrieved_literature}

Provide a 2-3 sentence explanation of why this patient is at risk.
<|assistant|>
"""
    
    # Step D: Generate the Answer
    print("Synthesizing explanation...")
    outputs = llm(prompt, max_new_tokens=100, temperature=0.3, return_full_text=False)
    explanation = outputs[0]['generated_text'].strip()
    
    return explanation

# 5. Run a Test Scenario
# Let's pretend our Neural Network from Phase 5 just flagged a patient
test_symptoms = "Patient is hypotensive and appears confused."
test_risk_score = 0.87

final_explanation = generate_clinical_explanation(test_symptoms, test_risk_score)

print("\n================ FINAL CO-PILOT OUTPUT ================")
print(f"Alert: High Risk of Deterioration ({test_risk_score * 100}%)")
print(f"Reasoning: {final_explanation}")
print("=======================================================")
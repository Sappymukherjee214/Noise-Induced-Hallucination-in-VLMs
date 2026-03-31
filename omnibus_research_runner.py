import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import pandas as pd
import albumentations as A
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# ==========================================
# 1. Omnibus Research Aggregator
# ==========================================
# Gathers P1, P2, P3 data into a final research CSV.

MODEL_ID = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "sample_input.jpg"

print(f"Initializing Omnibus Research Runner on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def calculate_token_entropy(outputs):
    logits = outputs.scores
    if not logits: return 0.0
    entropies = []
    for step_logits in logits:
        probs = F.softmax(step_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropies.append(entropy.mean().item())
    return np.mean(entropies)

def get_text_embedding(text: str):
    """Uses the BLIP text-encoder (local) to avoid external SBERT downloads."""
    inputs = processor(text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_outputs = model.text_decoder.bert(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            return_dict=True
        )
    # Mean pool the last hidden state
    embeddings = text_outputs.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_semantic_drift_local(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    cos_sim = F.cosine_similarity(emb1, emb2).item()
    return 1.0 - cos_sim # 0 = identical, 1 = complete drift

# 2. Experimental Loop (The 'Scorecard')
def perform_omnibus_study():
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    original_np = np.array(original_img)
    
    # Baseline
    inputs_clean = processor(images=original_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_clean = model.generate(**inputs_clean, max_length=50, return_dict_in_generate=True, output_scores=True)
    cap_clean = processor.decode(out_clean.sequences[0], skip_special_tokens=True)
    ent_clean = calculate_token_entropy(out_clean)
    emb_clean = get_text_embedding(cap_clean)
    
    print(f"BASELINE: {cap_clean} (H={ent_clean:.4f})")
    
    noise_types = ["gaussian", "blur", "jpeg"]
    severities = [0.2, 0.5, 0.8]
    
    records = []
    
    for nt in noise_types:
        for sev in severities:
            # Apply Noise
            if nt == "gaussian":
                transform = A.GaussNoise(var_limit=(sev*100, sev*100), p=1.0)
            elif nt == "blur":
                k = int(sev * 21) + 1
                if k % 2 == 0: k += 1
                transform = A.GaussianBlur(blur_limit=(k, k), p=1.0)
            elif nt == "jpeg":
                q = max(1, 100 - int(sev * 95))
                transform = A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
            
            noisy_np = transform(image=original_np)["image"]
            noisy_pil = Image.fromarray(noisy_np)
            
            # Predict
            inputs_noisy = processor(images=noisy_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out_noisy = model.generate(**inputs_noisy, max_length=50, return_dict_in_generate=True, output_scores=True)
            
            cap_noisy = processor.decode(out_noisy.sequences[0], skip_special_tokens=True)
            ent_noisy = calculate_token_entropy(out_noisy)
            emb_noisy = get_text_embedding(cap_noisy)
            drift = calculate_semantic_drift_local(emb_clean, emb_noisy)
            
            # 3. CHAIR-i check (Simplified)
            # Two cats are the baseline objects.
            hallucinated_obj = False
            if "cat" not in cap_noisy.lower(): 
                hallucinated_obj = True # Erasure/Swap
            if "dog" in cap_noisy.lower() or "bird" in cap_noisy.lower():
                hallucinated_obj = True # Injection
            
            rec = {
                "Noise": nt,
                "Severity": sev,
                "Caption": cap_noisy,
                "Entropy": ent_noisy,
                "Entropy_Gap": ent_noisy - ent_clean,
                "Semantic_Drift": drift,
                "Hallucination_Detected": hallucinated_obj
            }
            records.append(rec)
            print(f"[{nt} {sev}] H_Gap={rec['Entropy_Gap']:.4f} Drift={drift:.4f} -> {cap_noisy}")

    # 4. Save Final CSV
    df = pd.DataFrame(records)
    df.to_csv("expanded_research_results.csv", index=False)
    
    # 5. Result Plotting (Drift vs. Entropy Gap)
    plt.figure(figsize=(10, 6))
    for nt in noise_types:
        subset = df[df["Noise"] == nt]
        plt.scatter(subset["Entropy_Gap"], subset["Semantic_Drift"], s=100, label=f"Type: {nt}")
    
    plt.xlabel("Entropy Gap (Information Entropy Offset)")
    plt.ylabel("Semantic Drift (Cosine Distance from Source)")
    plt.title("P1: Mapping Information Entropy to Hallucination Drift")
    plt.grid(True)
    plt.legend()
    plt.savefig("global_robustness_spectrum.png")
    
    print("\n[SUCCESS] Omnibus research results aggregation complete.")
    print("CSV: 'expanded_research_results.csv'")
    print("Plot: 'global_robustness_spectrum.png'")

if __name__ == "__main__":
    perform_omnibus_study()

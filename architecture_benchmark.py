import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq
)
import albumentations as A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 2. R&D Benchmarking: BLIP vs. GIT (Microsoft)
# ==========================================
# RESEARCH GOAL: Compare 'Noise Robustness' across different architectures.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "sample_input.jpg"
MODELS = {
    "BLIP-Base (Salesforce)": "Salesforce/blip-image-captioning-base",
    "GIT-Base (Microsoft)": "microsoft/git-base"
}

print(f"Initializing Multi-Architecture Benchmark on {DEVICE}...")

def load_model(model_name: str):
    if "blip" in model_name.lower():
        # Load BLIP (already in cache probably)
        proc = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    else:
        # Load GIT
        proc = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name).to(DEVICE)
    return proc, model

def generate_caption(proc, model, image):
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return proc.decode(out[0], skip_special_tokens=True)

def run_arch_benchmark():
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    original_np = np.array(original_img)
    
    # 3. Create Noise Probe
    noise_types = ["blur", "jpeg"]
    severities = [0.2, 0.4, 0.6, 0.8]
    
    results = []
    
    for model_label, model_id in MODELS.items():
        print(f"\nProbing {model_label}...")
        proc, model = load_model(model_id)
        
        # Baseline
        clean_cap = generate_caption(proc, model, original_img)
        print(f"   [Baseline] {clean_cap}")
        
        for nt in noise_types:
            for sev in severities:
                # Apply Noise
                if nt == "blur":
                    k = int(sev * 21) + 1
                    if k % 2 == 0: k += 1
                    transform = A.GaussianBlur(blur_limit=(k, k), p=1.0)
                elif nt == "jpeg":
                    q = max(1, 100 - int(sev * 95))
                    transform = A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
                
                noisy_np = transform(image=original_np)["image"]
                noisy_img = Image.fromarray(noisy_np)
                
                # Inference
                caption = generate_caption(proc, model, noisy_img)
                print(f"      [{nt} {sev:.1f}] {caption}")
                results.append({
                    "Model": model_label,
                    "Noise": nt,
                    "Severity": sev,
                    "Caption": caption
                })
        
        # Free memory (crucial for benchmark)
        del model
        torch.cuda.empty_cache()
    
    # Analyze the benchmark
    df = pd.DataFrame(results)
    df.to_csv("architecture_robustness_benchmark.csv", index=False)
    print("\n[SUCCESS] Multi-model benchmark saved to 'architecture_robustness_benchmark.csv'.")

if __name__ == "__main__":
    run_arch_benchmark()

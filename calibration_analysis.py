import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 3. Calibration: Reliability Diagrams (ECE)
# ==========================================
# RESEARCH GOAL: Determine if the model knows WHEN it's hallucinating.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_PATH = "sample_input.jpg"
EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Initializing Calibration Unit on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
embedder = SentenceTransformer(EMBEDDER_ID).to(DEVICE)

def get_caption_with_conf(image_pil: Image.Image):
    """
    Returns the generated caption and its mean token-level confidence (softmax probability).
    """
    inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=50, 
            return_dict_in_generate=True, 
            output_scores=True
        )
    
    caption = processor.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Calculate confidence from scores
    logits = outputs.scores # Tuple of (seq_len, batch, vocab)
    confs = []
    for step_logits in logits:
        # Max probability of the selected token
        probs = F.softmax(step_logits, dim=-1)
        max_prob = torch.max(probs, dim=-1).values.item()
        confs.append(max_prob)
    
    return caption, np.mean(confs)

def run_calibration_experiment():
    import albumentations as A
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    
    # Baseline
    clean_cap, clean_conf = get_caption_with_conf(original_img)
    print(f"Baseline: {clean_cap} (Conf: {clean_conf:.4f})")
    
    # Probing across multiple noise levels to gather data points
    noise_types = ["gaussian", "blur", "jpeg"]
    severities = np.linspace(0.1, 0.9, 10)
    
    confidences = []
    accuracies = [] # Simulated 'Accuracy' via SBERT Similarity to clean baseline
    
    print("\nProbing noise-confidence spectrum...")
    for nt in noise_types:
        for sev in severities:
            # Apply Noise
            if nt == "gaussian":
                transform = A.GaussNoise(var_limit=(sev*100, sev*100), p=1.0)
            elif nt == "blur":
                k = int(sev * 31) + 1
                if k % 2 == 0: k += 1
                transform = A.GaussianBlur(blur_limit=(k, k), p=1.0)
            elif nt == "jpeg":
                q = max(1, 100 - int(sev * 98))
                transform = A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
            
            noisy_pil = Image.fromarray(transform(image=np.array(original_img))["image"])
            
            # Predict
            cap, conf = get_caption_with_conf(noisy_pil)
            
            # Simulated 'Accuracy' (0-1) using semantic similarity
            emb1 = embedder.encode(clean_cap, convert_to_tensor=True)
            emb2 = embedder.encode(cap, convert_to_tensor=True)
            accuracy = util.cos_sim(emb1, emb2).item()
            
            confidences.append(conf)
            accuracies.append(accuracy)
            
    # 4. Create Reliability Diagram
    plt.figure(figsize=(10, 10))
    plt.scatter(confidences, accuracies, alpha=0.5, label="Noise Samples")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    
    # Trend line
    z = np.polyfit(confidences, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(confidences, p(confidences), "r-", label="Model Calibration Trend")
    
    plt.xlabel("Model Confidence (Mean Token Prob)")
    plt.ylabel("Observed Semantic Accuracy (vs. Clean)")
    plt.title("Reliability Diagram: Noise-Induced Calibration Drift")
    plt.legend()
    plt.grid(True)
    plt.savefig("vlm_reliability_diagram.png")
    
    # Calculate ECE (Expected Calibration Error) approximation
    diff = np.abs(np.array(confidences) - np.array(accuracies))
    ece = np.mean(diff)
    print(f"\n[RESULTS] Approximate ECE Score (Error): {ece:.4f}")
    print("[SUCCESS] Calibration analysis saved to 'vlm_reliability_diagram.png'.")

if __name__ == "__main__":
    run_calibration_experiment()

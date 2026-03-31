import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn.functional as F
import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. Targeted Deep Probe (JPEG 0.8)
# ==========================================
MODEL_ID = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "sample_input.jpg"
JPEG_SEVERITY = 0.8

print(f"Loading Deep VLM Probing Unit on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def calculate_token_entropy(outputs):
    """Measures the model's 'Confidence vs Hallucination'."""
    logits = outputs.scores
    entropies = []
    for step_logits in logits:
        probs = F.softmax(step_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropies.append(entropy.mean().item())
    return np.mean(entropies)

def probe_deep_hallucination():
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    original_np = np.array(original_img)
    
    # 2. Targeted Corruption: Extreme JPEG artifacts
    quality = max(1, 100 - int(JPEG_SEVERITY * 95))
    transform = A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)
    noisy_np = transform(image=original_np)["image"]
    noisy_pil = Image.fromarray(noisy_np)
    
    # 3. VLM Inference with Score Output
    inputs_clean = processor(images=original_img, return_tensors="pt").to(DEVICE)
    inputs_noisy = processor(images=noisy_pil, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        out_clean = model.generate(**inputs_clean, max_length=50, return_dict_in_generate=True, output_scores=True)
        out_noisy = model.generate(**inputs_noisy, max_length=50, return_dict_in_generate=True, output_scores=True)
        
    cap_clean = processor.decode(out_clean.sequences[0], skip_special_tokens=True)
    cap_noisy = processor.decode(out_noisy.sequences[0], skip_special_tokens=True)
    
    ent_clean = calculate_token_entropy(out_clean)
    ent_noisy = calculate_token_entropy(out_noisy)
    
    print(f"\n--- [RESULTS: JPEG 0.8 SEVERITY] ---")
    print(f"BASE CAPTION: {cap_clean} (Entropy: {ent_clean:.4f})")
    print(f"NOISY CAPTION: {cap_noisy} (Entropy: {ent_noisy:.4f})")
    
    # 4. Comparative Visual Record
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(original_np)
    ax1.set_title(f"Baseline (Clean)\nCaption: {cap_clean}\nEntropy: {ent_clean:.4f}")
    ax1.axis('off')
    
    ax2.imshow(noisy_np)
    ax2.set_title(f"Targeted Probe (JPEG Sev={JPEG_SEVERITY})\nCaption: {cap_noisy}\nEntropy: {ent_noisy:.4f}")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("jpeg_detailed_probe_0_8.png")
    print(f"\n[SUCCESS] Detailed probe output image saved to 'jpeg_detailed_probe_0_8.png'.")

if __name__ == "__main__":
    probe_deep_hallucination()

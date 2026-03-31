import torch
from PIL import Image, ImageFilter
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 4. Mitigation: Adversarial Denoising Prefixes
# ==========================================
# RESEARCH GOAL: Can simple pre-processing 'Defend' against 
# noise-induced hallucinations (sitting -> sleeping)?

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_PATH = "sample_input.jpg"

print(f"Initializing Defense Engine on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def denoise_prefix_defense(noisy_np: np.ndarray):
    """
    Applies a classic R&D-grade denoising filter (Median Blur + NLMeans) 
    before the VLM to mitigate artifact-driven hallucinations.
    """
    # 1. Median Filter (removes 'salt and pepper' or sharp artifacts)
    denoised = cv2.medianBlur(noisy_np, 5)
    # 2. Gaussian Denoising
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
    return denoised

def generate_caption(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

def run_defense_experiment():
    import albumentations as A
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    original_np = np.array(original_img)
    
    # 3. Create Severe Noise (e.g., Gaussian + JPEG)
    noise_op = A.Compose([
        A.GaussNoise(var_limit=(80.0, 80.0), p=1.0),
        A.ImageCompression(quality_lower=5, quality_upper=5, p=1.0)
    ])
    noisy_np = noise_op(image=original_np)["image"]
    noisy_pil = Image.fromarray(noisy_np)
    
    # 4. Apply Defense
    print("Applying Denoising Defense prefix...")
    denoised_np = denoise_prefix_defense(noisy_np)
    denoised_pil = Image.fromarray(denoised_np)
    
    # 5. Benchmarking Accuracy
    clean_cap = generate_caption(original_img)
    hallucinated_cap = generate_caption(noisy_pil)
    defended_cap = generate_caption(denoised_pil)
    
    print(f"\n--- [RESULTS: HALLUCINATION DEFENSE] ---")
    print(f"BASELINE: {clean_cap}")
    print(f"ND-HALLUCINATION: {hallucinated_cap}")
    print(f"DEFENDED CAPTION: {defended_cap}")
    
    # Visualization: The Defense Corridor
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    axes[0].imshow(original_np)
    axes[0].set_title(f"Original (Descriptive)\n{clean_cap}")
    axes[0].axis('off')
    
    axes[1].imshow(noisy_np)
    axes[1].set_title(f"Noised (Hallucinated)\n{hallucinated_cap}")
    axes[1].axis('off')
    
    axes[2].imshow(denoised_np)
    axes[2].set_title(f"Defended (Corrected?)\n{defended_cap}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("vlm_denoising_defense_result.png")
    print("\n[SUCCESS] Defense benchmarking saved to 'vlm_denoising_defense_result.png'.")

if __name__ == "__main__":
    run_defense_experiment()

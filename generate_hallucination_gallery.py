import torch
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import albumentations as A
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Setup & Resource Fetching
# ==========================================
IMAGE_URL = "https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png" # Stable GH Link
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"

print(f"Initializing Research Demo on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def download_image(url: str, path: str):
    if not os.path.exists(path):
        print(f"Downloading sample image from {url}...")
        img_data = requests.get(url, verify=False).content # Bypass SSL for demo reliability
        with open(path, 'wb') as handler:
            handler.write(img_data)
    return Image.open(path).convert("RGB")

# ==========================================
# 2. Hallucination Probe Functions
# ==========================================
def apply_noise(image_np: np.ndarray, n_type: str, severity: float):
    if n_type == "gaussian":
        transform = A.GaussNoise(var_limit=(severity*100, severity*100), p=1.0)
    elif n_type == "blur":
        k = int(severity * 21) + 1
        if k % 2 == 0: k += 1
        transform = A.GaussianBlur(blur_limit=(k, k), p=1.0)
    elif n_type == "jpeg":
        q = max(1, int(100 - (severity * 95)))
        transform = A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
    else:
        return image_np
    return transform(image=image_np)["image"]

def generate_caption(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

# ==========================================
# 3. Main Gallery Generation
# ==========================================
def main_gallery():
    original_img = download_image(IMAGE_URL, "sample_input.jpg")
    original_np = np.array(original_img)
    
    clean_caption = generate_caption(original_img)
    print(f"Original Caption: {clean_caption}")
    
    noise_types = ["gaussian", "blur", "jpeg"]
    severities = [0.2, 0.5, 0.8]
    
    # Create the Visual Hallucination Gallery
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    plt.suptitle(f"Research: Visual Hallucination Under Semantic Degradation\nOriginal: {clean_caption}", fontsize=20)
    
    for i, nt in enumerate(noise_types):
        for j, sev in enumerate(severities):
            noisy_np = apply_noise(original_np, nt, sev)
            noisy_pil = Image.fromarray(noisy_np)
            
            caption = generate_caption(noisy_pil)
            print(f"[{nt.upper()} Sev={sev}] -> {caption}")
            
            axes[i][j].imshow(noisy_np)
            axes[i][j].set_title(f"{nt.upper()} (S={sev})\n{caption}", fontsize=10, pad=10)
            axes[i][j].axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("hallucination_gallery.png")
    print("\n[SUCCESS] Hallucination gallery saved to 'hallucination_gallery.png'.")

if __name__ == "__main__":
    main_gallery()

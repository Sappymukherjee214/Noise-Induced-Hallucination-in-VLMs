import torch
import numpy as np
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import albumentations as A
from typing import List, Tuple

# 1. Configuration & Models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"

print(f"Loading processor and model: {MODEL_ID}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

# 2. Define Noise Pipeline
def apply_noise(image_pil: Image.Image, noise_type: str, severity: float) -> Image.Image:
    """
    Apply visual noise to a PIL Image.
    severity: 0.0 to 1.0
    """
    image_np = np.array(image_pil)
    
    if noise_type == "gaussian":
        # Severity scales the variance
        var = severity * 0.1
        noise = A.GaussNoise(var_limit=(var, var), p=1.0)
        image_np = noise(image=image_np)["image"]
        
    elif noise_type == "blur":
        # Severity scales the blur limit
        ksize = int(severity * 20) + 1
        if ksize % 2 == 0: ksize += 1
        noise = A.GaussianBlur(blur_limit=(ksize, ksize), p=1.0)
        image_np = noise(image=image_np)["image"]
        
    elif noise_type == "jpeg":
        # Severity scales the quality (inverse)
        quality = int(100 - (severity * 90))
        noise = A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)
        image_np = noise(image=image_np)["image"]

    return Image.fromarray(image_np)

# 3. Caption Generation Pipeline
def generate_caption(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# 4. Example Simulation
def run_hallucination_experiment(image_path: str):
    print(f"\nProcessing: {image_path}")
    original_img = Image.open(image_path).convert("RGB")
    
    # Baseline
    clean_caption = generate_caption(original_img)
    print(f"Clean Caption: {clean_caption}")
    
    noise_types = ["gaussian", "blur", "jpeg"]
    severities = [0.2, 0.5, 0.8]
    
    results = []
    
    for nt in noise_types:
        for sev in severities:
            noisy_img = apply_noise(original_img, nt, sev)
            noisy_caption = generate_caption(noisy_img)
            
            # Save for inspection if needed
            # noisy_img.save(f"debug_{nt}_{sev}.jpg") 
            
            print(f"[{nt.upper()} Sev={sev}] -> {noisy_caption}")
            results.append({
                "noise": nt,
                "severity": sev,
                "caption": noisy_caption
            })
            
    return results

if __name__ == "__main__":
    # In a real scenario, the user provides a path to a Flickr8k or COCO image.
    # For now, this serves as a blueprint.
    print("Script initialized. Ready for experimental run.")
    # Example usage (commented out as dummy file might not exist):
    # run_hallucination_experiment("example_image.jpg")

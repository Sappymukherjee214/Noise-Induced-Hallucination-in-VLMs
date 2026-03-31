import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

# ==========================================
# 1. Advanced Vision-Encoder Attention Visualization
# ==========================================
# RESEARCH GOAL: Show 'Visual Drift' - where the ViT encoder focuses 
# when noise causes it to hallucinate.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_PATH = "sample_input.jpg"

print(f"Initializing ViT Attention Engine on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def get_vit_attention(image_pil: Image.Image):
    """
    Extracts self-attention from the BLIP Vision Encoder (last layer).
    Shows the spatial distribution of 'Vision Interest'.
    """
    inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Hook into the vision model specifically
        outputs = model.vision_model(
            pixel_values=inputs.pixel_values,
            output_attentions=True,
            return_dict=True
        )
    
    # Vision attentions: Tuple of layers (num_layers, batch, heads, 577, 577)
    attentions = outputs.attentions
    if not attentions: return None

    # Focus on the last layer, averaged across heads
    last_layer = attentions[-1] # (batch, heads, 577, 577)
    avg_heads = last_layer.mean(dim=1).squeeze() # (577, 577)
    
    # We want the attention from the CLS token (token 0) to all patches
    cls_to_patches = avg_heads[0, 1:].detach().cpu().numpy() # (576,)
    
    # Reshape into 24x24 (BLIP patches)
    heatmap = cls_to_patches.reshape(24, 24)
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize to original image
    heatmap = cv2.resize(heatmap, (image_pil.size[0], image_pil.size[1]))
    
    return heatmap

def run_attention_experiment():
    import albumentations as A
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    original_np = np.array(original_img)
    
    # Test Clean vs. High Gaussian Noise
    noise_op = A.GaussNoise(var_limit=(60.0, 60.0), p=1.0)
    noisy_np = noise_op(image=original_np)["image"]
    noisy_pil = Image.fromarray(noisy_np)
    
    print("Extracting ViT focus: Clean...")
    heat_clean = get_vit_attention(original_img)
    print("Extracting ViT focus: Noisy...")
    heat_noisy = get_vit_attention(noisy_pil)
    
    # Generate simple captions for context
    inputs_clean = processor(images=original_img, return_tensors="pt").to(DEVICE)
    inputs_noisy = processor(images=noisy_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        cap_clean = processor.decode(model.generate(**inputs_clean)[0], skip_special_tokens=True)
        cap_noisy = processor.decode(model.generate(**inputs_noisy)[0], skip_special_tokens=True)
    
    # Visualization: Side-by-Side Heatmap Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title(f"Baseline (Clean)\nCaption: {cap_clean}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_np)
    axes[0, 1].imshow(heat_clean, alpha=0.6, cmap='jet')
    axes[0, 1].set_title("ViT Spatial Focus (Clean)")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(noisy_np)
    axes[1, 0].set_title(f"Noisy (Gaussian Sev=0.6)\nCaption: {cap_noisy}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(noisy_np)
    axes[1, 1].imshow(heat_noisy, alpha=0.6, cmap='jet')
    axes[1, 1].set_title("ViT Spatial Focus (Noisy - Attention Diffusion)")
    axes[1, 1].axis('off')
    
    plt.suptitle("P2: Mapping Vision Transformer (ViT) Attention Diffusion under Sensory Noise", fontsize=18)
    plt.tight_layout()
    plt.savefig("cross_modal_attention_study.png")
    print("\n[SUCCESS] ViT Attention study results saved to 'cross_modal_attention_study.png'.")

if __name__ == "__main__":
    run_attention_experiment()

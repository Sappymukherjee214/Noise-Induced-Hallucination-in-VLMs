import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

# ==========================================
# 1. Advanced Cross-Modal Attention Visualization
# ==========================================
# RESEARCH GOAL: Show 'Visual Drift' - where the model focuses 
# when noise causes it to hallucinate.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_PATH = "sample_input.jpg"

print(f"Initializing Attention Extraction Engine on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def get_attention_heatmap(image_pil: Image.Image):
    """
    Extracts cross-attention from the BLIP text-decoder. 
    Shows which image patches inform the caption.
    """
    inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    
    # Run with output_attentions=True
    outputs = model.generate(
        **inputs, 
        max_length=50, 
        return_dict_in_generate=True, 
        output_attentions=True
    )
    
    caption = processor.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # BLIP decoder cross-attention is in outputs.cross_attentions
    # Tuple of (seq_len, num_layers, batch, num_heads, q_len, k_len)
    attentions = outputs.cross_attentions # (tgt_len, layers, batch, heads, query, key)
    
    if not attentions:
        return caption, None

    # Focus on the last layer, averaged across heads
    # key_len for BLIP ViT-base is 577 (1 cls + 24*24 patches)
    last_step_attns = attentions[-1] # (layers, batch, heads, query, key)
    last_layer_attn = last_step_attns[-1] # (batch, heads, 1, 577)
    avg_heads_attn = last_layer_attn.mean(dim=1).squeeze() # (577,)
    
    # Remove the CLS token (first token)
    patch_attn = avg_heads_attn[1:].detach().cpu().numpy()
    
    # Reshape into 24x24 (BLIP patches)
    heatmap = patch_attn.reshape(24, 24)
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (image_pil.size[0], image_pil.size[1]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return caption, heatmap

def run_attention_experiment():
    import albumentations as A
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    original_np = np.array(original_img)
    
    # Test Clean vs. High Gaussian Noise
    noise_op = A.GaussNoise(var_limit=(60.0, 60.0), p=1.0)
    noisy_np = noise_op(image=original_np)["image"]
    noisy_pil = Image.fromarray(noisy_np)
    
    print("Extracting attention: Clean...")
    cap_clean, heat_clean = get_attention_heatmap(original_img)
    print("Extracting attention: Noisy...")
    cap_noisy, heat_noisy = get_attention_heatmap(noisy_pil)
    
    # Visualization: Side-by-Side Heatmap Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title(f"Baseline (Clean)\nCaption: {cap_clean}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_np)
    axes[0, 1].imshow(heat_clean, alpha=0.5, cmap='jet')
    axes[0, 1].set_title("Cross-Modal Visual Focus (Clean)")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(noisy_np)
    axes[1, 0].set_title(f"Noisy (Gaussian Sev=0.6)\nCaption: {cap_noisy}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(noisy_np)
    axes[1, 1].imshow(heat_noisy, alpha=0.5, cmap='jet')
    axes[1, 1].set_title("Cross-Modal Visual Focus (Noisy - Attention Diffusion)")
    axes[1, 1].axis('off')
    
    plt.suptitle("P1 Explainer: Cross-Modal Attention Heatmaps under Noise", fontsize=20)
    plt.tight_layout()
    plt.savefig("cross_modal_attention_study.png")
    print("\n[SUCCESS] Attention study results saved to 'cross_modal_attention_study.png'.")

if __name__ == "__main__":
    run_attention_experiment()

import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Adversarial Probing (P2 Pillar of the Research Proposal)
# ==========================================
# We use PGD (Projected Gradient Descent) to find the smallest 
# NOISE pattern (adversarial) that induces hallucination.

MODEL_ID = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "sample_input.jpg"

print(f"Initializing Adversarial Robustness Probe on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def adversarial_pgd_attack(image_pil: Image.Image, target_text: str, epsilon: float = 8/255, alpha: float = 2/255, iters: int = 5):
    """
    Induces 'Structured Hallucination' by perturbing the pixel space.
    epsilon: max perturbation (per-pixel).
    alpha: step size.
    """
    # 1. Preprocess the original image
    inputs = processor(images=image_pil, text="", return_tensors="pt").to(DEVICE)
    pixel_values = inputs.pixel_values.clone().detach().requires_grad_(True)
    
    # 2. Define the Target (What we want the model to hallucinate)
    # We want to minimize the loss for a WRONG caption.
    target_tokens = processor(text=target_text, return_tensors="pt").to(DEVICE).input_ids
    
    # 3. PGD Loop (Adversarial Optimization)
    # We flip the gradient to minimize the loss for the target_text.
    model.eval() # VLM in eval mode
    for i in range(iters):
        outputs = model(pixel_values=pixel_values, labels=target_tokens)
        loss = outputs.loss
        loss.backward()
        
        with torch.no_grad():
            # Minimize loss for target_text (Hallucination injection)
            adv_pixels = pixel_values - alpha * pixel_values.grad.sign()
            # Project back into the epsilon ball
            eta = torch.clamp(adv_pixels - inputs.pixel_values, min=-epsilon, max=epsilon)
            pixel_values.data = torch.clamp(inputs.pixel_values + eta, min=-1, max=1)
        
        pixel_values.grad.zero_()
        print(f"   PGD Iter {i+1}: Targeted Hallucination Loss = {loss.item():.4f}")

    # 4. Convert back to Pil
    # We denormalize for visualization (mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275] for BLIP)
    adv_np = pixel_values.squeeze().detach().cpu().numpy()
    adv_np = np.transpose(adv_np, (1, 2, 0)) # CHW -> HWC
    # Simplified denorm for visualization
    adv_np = (adv_np * 0.25) + 0.45
    adv_np = np.clip(adv_np * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(adv_np)

def run_adversarial_experiment():
    print("\n--- [START] Adversarial Hallucination Experiment ---")
    print("Scenario: Natural (Two Cats) vs. Targeted Hallucination (A small bird)")
    
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    target_caption = "a photo of a small bird"
    
    # Run the attack
    adv_img = adversarial_pgd_attack(original_img, target_caption, epsilon=16/255)
    
    # Verify the Hallucination
    # Generate caption for both
    inputs_clean = processor(images=original_img, return_tensors="pt").to(DEVICE)
    inputs_adv = processor(images=adv_img, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        cap_clean = model.generate(**inputs_clean, max_length=50)
        cap_adv = model.generate(**inputs_adv, max_length=50)
        
    res_clean = processor.decode(cap_clean[0], skip_special_tokens=True)
    res_adv = processor.decode(cap_adv[0], skip_special_tokens=True)
    
    print(f"\n[RESULTS] Original Caption: {res_clean}")
    print(f"[RESULTS] Adversarial Caption: {res_adv}")
    
    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(original_img)
    ax1.set_title(f"Baseline (Clean)\nCaption: {res_clean}")
    ax2.imshow(adv_img)
    ax2.set_title(f"Adversarial (PGD)\nCaption: {res_adv}\n(Targeted at 'small bird')")
    
    plt.savefig("adversarial_hallucination_result.png")
    print("\n[SUCCESS] Adversarial study saved to 'adversarial_hallucination_result.png'.")

if __name__ == "__main__":
    run_adversarial_experiment()

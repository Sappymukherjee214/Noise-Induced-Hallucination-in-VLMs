import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Causal Probing (Modal Dominance) Logic
# ==========================================
# RESEARCH HYPOTHESIS: When visual signal is noisy, does the VLM's 
# language prior 'overpower' the visual evidence (Modal Dominance)?
# We test this by introducing a CONFLICTING prompt.

MODEL_ID = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "sample_input.jpg" # Original is 'two cats'

print(f"Initializing Modal Dominance Probing on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

def probe_modal_dominance(noise_intensity: float, conflicting_prompt: str):
    """
    Inputs:
      - noise_intensity: Level of blurring/noise on the visual input.
      - conflicting_prompt: A biased text prefix (e.g., 'A photo of a healthy dog')
    """
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    
    # 2. Add Visual Entropy (Blur)
    if noise_intensity > 0:
        import albumentations as A
        k = int(noise_intensity * 41) + 1
        if k % 2 == 0: k += 1
        transform = A.GaussianBlur(blur_limit=(k, k), p=1.0)
        img_np = np.array(original_img)
        img_noisy = transform(image=img_np)["image"]
        img_pil = Image.fromarray(img_noisy)
    else:
        img_pil = original_img

    # 3. Conflict Injection (Causal Probe)
    # We use 'conditional' generation with a conflicting text prompt.
    # We want to see if the model corrects the prompt or 'hallucinates' 
    # the dog.
    inputs = processor(images=img_pil, text=conflicting_prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # We output scores to check entropy/confidence during conflict
        outputs = model.generate(
            **inputs, 
            max_length=50, 
            return_dict_in_generate=True, 
            output_scores=True
        )
        
    caption = processor.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Calculate Token Entropy (The 'Uncertainty' of the conflict)
    logits = outputs.scores
    entropies = []
    for step_logits in logits:
        probs = F.softmax(step_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropies.append(entropy.mean().item())
    
    mean_entropy = np.mean(entropies)
    
    return caption, mean_entropy, img_pil

def run_causal_experiment():
    print("\n--- [START] Causal Modal Dominance Probe ---")
    print("Scenario: Visual info (Two Cats) vs. Textual Prior (A dog)")
    
    prompt = "A photo of a dog and its" # Leading the model to hallucinate a dog
    intensities = [0.0, 0.4, 0.8] # Increase visual degradation
    
    results = []
    for sev in intensities:
        cap, entropy, img = probe_modal_dominance(sev, prompt)
        print(f"Intensity {sev:.1f}: [CAPTION] {cap} (Entropy: {entropy:.4f})")
        results.append((sev, cap, entropy, img))
        
    # 4. Visualization for Research Report
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (sev, cap, entropy, img) in enumerate(results):
        axes[i].imshow(img)
        axes[i].set_title(f"Visual Sev={sev}\nPrompt: '{prompt}...'\nResult: '{cap}'\nEntropy: {entropy:.4f}", fontsize=9)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig("modal_dominance_probe.png")
    print("\n[SUCCESS] Causal probing result saved to 'modal_dominance_probe.png'.")

if __name__ == "__main__":
    run_causal_experiment()

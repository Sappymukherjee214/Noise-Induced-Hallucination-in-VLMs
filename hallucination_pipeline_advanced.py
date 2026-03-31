import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import albumentations as A
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import os
import cv2
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

# 1. Advanced Configuration & Model Suite
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLM_ID = "Salesforce/blip-image-captioning-base"
CLIP_ID = "openai/clip-vit-base-patch32"
EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"

print("--- INITIALIZING EXPANDED OMNIBUS RESEARCH SUITE (TIER-1+) ---")
# Pre-download NLTK data for metrics
nltk.download('wordnet')
nltk.download('punkt')

processor = BlipProcessor.from_pretrained(VLM_ID)
vlm = BlipForConditionalGeneration.from_pretrained(VLM_ID).to(DEVICE)
clip_model = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
embedder = SentenceTransformer(EMBEDDER_ID).to(DEVICE)

# 2. Comprehensive Noise Suite
class NoiseFactory:
    @staticmethod
    def get_noise(name: str, level: int):
        """Standardized severity levels 1-5 for academic reproducibility"""
        if name == "gaussian":
            return A.GaussNoise(var_limit=(0.1 * level, 0.2 * level), p=1.0)
        elif name == "blur":
            k = level * 4 + 1
            return A.GaussianBlur(blur_limit=(k, k), p=1.0)
        elif name == "jpeg":
            q = 100 - (level * 18)
            return A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
        elif name == "low_light":
            return A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.15*level, -0.15*level), contrast_limit=0, p=1.0),
                A.ISONoise(color_shift=(0.01*level, 0.05*level), intensity=(0.1*level, 0.2*level), p=1.0)
            ])
        elif name == "fog":
            return A.Fog(fog_coef_lower=0.1*level, fog_coef_upper=0.2*level, p=1.0)
        return A.NoOp()

# 3. ADVANCED VISUAL DIAGNOSTICS: SALIENCY HEATMAPS
def generate_saliency_heatmap(vlm_model, processor, original_image: Image.Image) -> np.ndarray:
    vlm_model.eval()
    inputs = processor(images=original_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = vlm_model(**inputs, labels=inputs["pixel_values"], output_attentions=True)
        cross_atn = outputs.decoder_attentions[-1].mean(dim=1).squeeze(0)
    saliency = cross_atn[0, 1:].cpu().numpy()
    dim = int(np.sqrt(saliency.shape[0]))
    heatmap = saliency.reshape(dim, dim)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
    return heatmap

# 4. ADVERSARIAL HALLUCINATION ENGINE
def generate_adversarial_hallucination(vlm_model, processor, original_image: Image.Image, epsilon: float = 0.03) -> Image.Image:
    vlm_model.eval()
    inputs = processor(images=original_image, return_tensors="pt").to(DEVICE)
    pixel_values = inputs.pixel_values.clone().detach().requires_grad_(True)
    outputs = vlm_model(pixel_values=pixel_values, labels=inputs["pixel_values"])
    entropy_loss = -torch.sum(F.softmax(outputs.logits[:, 0, :], dim=-1) * F.log_softmax(outputs.logits[:, 0, :], dim=-1), dim=-1).mean()
    vlm_model.zero_grad()
    entropy_loss.backward()
    perturbed = pixel_values + epsilon * pixel_values.grad.data.sign()
    adv_np = torch.clamp(perturbed, -1, 1).squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    adv_np = (adv_np - adv_np.min()) / (adv_np.max() - adv_np.min())
    return Image.fromarray((adv_np * 255).astype(np.uint8))

# 5. RESEARCH METRICS (EXPANDED NLP & VISION)
def calculate_nlp_drift(baseline: str, target: str) -> Dict[str, float]:
    ref = baseline.split()
    hyp = target.split()
    # Sentence BLEU (B-4)
    bleu = sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25))
    # METEOR Score
    met = meteor_score([ref], hyp)
    return {"bleu4": bleu, "meteor": met}

def get_prior_dependency(vlm_model, processor, image: Image.Image) -> float:
    ipt_real = processor(images=image, return_tensors="pt").to(DEVICE)
    null_img = Image.new('RGB', (224, 224), (0, 0, 0))
    ipt_null = processor(images=null_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        l_real = vlm_model(**ipt_real, labels=ipt_real["input_ids"]).logits[:, 0, :]
        l_null = vlm_model(**ipt_null, labels=ipt_null["input_ids"]).logits[:, 0, :]
        kl = F.kl_div(F.log_softmax(l_real, dim=-1), F.softmax(l_null, dim=-1), reduction='batchmean')
    return 1.0 / (1.0 + kl.item())

def calculate_mc_dropout_uncertainty(vlm_model, processor, image: Image.Image) -> float:
    vlm_model.train()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits_list = [vlm_model(**inputs, labels=inputs["pixel_values"]).logits for _ in range(3)]
        var = torch.var(torch.stack(logits_list), dim=0).mean().item()
    vlm_model.eval()
    return var

def get_caption_faithfulness(image: Image.Image, caption: str) -> float:
    c_in = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        score = clip_model(**c_in).logits_per_image.item() / 100.0
    return score

# 6. GLOBAL VISUALIZATION: ROBUSTNESS SPECTRUM
def plot_robustness_spectrum(df: pd.DataFrame):
    print("\n--- GENERATING GLOBAL ROBUSTNESS SPECTRUM ---")
    plt.figure(figsize=(12, 6))
    types = df["type"].unique()
    for t in types:
        sub = df[df["type"] == t]
        plt.scatter(sub["prior"], sub["faith"], alpha=0.5, label=t)
    plt.xlabel("Prior Bias Index (Hallucination Propensity)")
    plt.ylabel("CLIP Faithfulness (Visual Grounding)")
    plt.title("The Robustness Spectrum: Information Bottleneck Collapse Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig("global_robustness_spectrum.png")
    plt.close()

# 7. OMNIBUS EXPERIMENTAL ENGINE (EXPANDED)
def run_omnibus_benchmark(image_paths: List[str]):
    os.makedirs("omnibus_artifacts", exist_ok=True)
    results = []
    
    for path in tqdm(image_paths, desc="OMNIBUS_EXPANDED_RUN"):
        orig = Image.open(path).convert("RGB")
        
        # Generation Logic
        def evaluate(img_pil, type_tag, base_caption=None):
            inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = vlm.generate(**inputs, max_length=15, return_dict_in_generate=True, output_scores=True)
            caption = processor.decode(out.sequences[0], skip_special_tokens=True)
            
            # Metrics
            faith = get_caption_faithfulness(orig, caption)
            prior = get_prior_dependency(vlm, processor, img_pil)
            mc_var = calculate_mc_dropout_uncertainty(vlm, processor, img_pil)
            
            # Traditional NLP Drift (if baseline exists)
            nlp_metrics = calculate_nlp_drift(base_caption, caption) if base_caption else {"bleu4": 1.0, "meteor": 1.0}
            
            return {
                "tag": type_tag, "caption": caption, "faith": faith, 
                "prior": prior, "mc_var": mc_var, "img": img_pil,
                "bleu4": nlp_metrics["bleu4"], "meteor": nlp_metrics["meteor"]
            }

        # 1. Baseline
        b_res = evaluate(orig, "Baseline")
        
        # 2. Comparative Scenarios
        scenarios = [
            b_res,
            evaluate(Image.fromarray(NoiseFactory.get_noise("low_light", 4)(image=np.array(orig))["image"]), "Natural Noise", b_res["caption"]),
            evaluate(generate_adversarial_hallucination(vlm, processor, orig), "Adversarial Hallucination", b_res["caption"])
        ]
        
        # Save Artifacts
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))
        for i, sc in enumerate(scenarios):
            axs[i].imshow(sc["img"]); axs[i].set_title(f"{sc['tag']}\nBLEU:{sc['bleu4']:.2f}")
            results.append({
                "image": os.path.basename(path), "type": sc["tag"], "caption": sc["caption"], 
                "faith": sc["faith"], "prior": sc["prior"], "var": sc["mc_var"],
                "bleu4": sc["bleu4"], "meteor": sc["meteor"]
            })
            
        hmap = generate_saliency_heatmap(vlm, processor, scenarios[2]["img"])
        axs[3].imshow(hmap, cmap='jet'); axs[3].set_title("Saliency Profile (Adv)")
        plt.tight_layout()
        plt.savefig(f"omnibus_artifacts/{os.path.basename(path)}_expanded_profile.png")
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv("expanded_research_results.csv", index=False)
    plot_robustness_spectrum(df)
    print("\n[SUCCESS] Expanded Omnibus Complete. Results in 'expanded_research_results.csv' & 'global_robustness_spectrum.png'.")

if __name__ == "__main__":
    print("Advanced VLM Research Suite Loaded.")

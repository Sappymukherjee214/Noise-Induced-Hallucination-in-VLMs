import requests
from PIL import Image
from io import BytesIO
import os
from hallucination_pipeline_advanced import run_benchmarking_session

def download_demo_images():
    # A few COCO-like images from Wikimedia/Unsplash for demonstration
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Common_myna_at_Lalbagh_Botanical_Garden.JPG/1200px-Common_myna_at_Lalbagh_Botanical_Garden.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/b/b8/Beach_Kitchen.jpg"
    ]
    
    os.makedirs("demo_images", exist_ok=True)
    paths = []
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            path = f"demo_images/sample_{i}.jpg"
            img.save(path)
            paths.append(path)
            print(f"Downloaded: {path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return paths

if __name__ == "__main__":
    print("--- NIH-VLM DEMO RUNNER ---")
    image_paths = download_demo_images()
    
    if image_paths:
        noise_types = ["gaussian", "blur", "low_light", "fog"]
        levels = [1, 3, 5]
        
        print(f"Starting benchmark on {len(image_paths)} images...")
        run_benchmarking_session(image_paths, noise_types, levels)
    else:
        print("No images available to run demo.")

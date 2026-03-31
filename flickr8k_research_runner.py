import kagglehub
import os
import glob
from hallucination_pipeline_advanced import run_benchmarking_session

def setup_flickr8k():
    print("--- DOWNLOADING FLICKR8K DATASET ---")
    # Download latest version via kagglehub
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    print("Path to dataset files:", path)
    
    # Typically Flickr8k from this source has an 'Images' or 'images' subfolder
    image_dir = os.path.join(path, "images")
    if not os.path.exists(image_dir):
        # Check for case sensitivity or different structure
        image_dir = os.path.join(path, "Images")
        
    if not os.path.exists(image_dir):
        print(f"Warning: Could not find image directory in {path}. Searching recursively...")
        all_jpgs = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        if all_jpgs:
            image_dir = os.path.dirname(all_jpgs[0])
            print(f"Found images at: {image_dir}")
        else:
            raise FileNotFoundError("No images found in the downloaded dataset.")
            
    return image_dir

if __name__ == "__main__":
    try:
        image_folder = setup_flickr8k()
        
        # Load a subset of images for the benchmark
        # We take 20 images for a robust but relatively quick research run
        image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))[:20]
        
        if not image_paths:
            print("Error: No images found to process.")
        else:
            print(f"Starting OMNIBUS research benchmark on {len(image_paths)} Flickr8k images...")
            
            # Execute the Unified Omnibus Research Suite
            from hallucination_pipeline_advanced import run_omnibus_benchmark
            run_omnibus_benchmark(image_paths)
            
            print("\n[OMNIBUS SUCCESS] Comprehensive Research Suite Complete.")
            print("Omnibus artifacts are in 'omnibus_artifacts/'.")
            print("Full dataset report saved in 'omnibus_research_results.csv'.")
            
    except Exception as e:
        print(f"An error occurred during the Flickr8k setup: {e}")

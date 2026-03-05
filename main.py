import os
import glob
import pickle
import faiss
import numpy as np
from utils import (
    load_image, extract_clip_image_embedding, get_geometric_variations, 
    calculate_lpips_distance, get_clip_components, get_lpips_model
)

# --- Configuration ---
TARGET_IMAGE_PATH = "C:/Users/ICYICO/Desktop/fast-dataset-deduplication/unnamed.png"
SEARCH_DIRECTORY = "C:/Users/ICYICO/Desktop/fast-dataset-deduplication"

INDEX_FILE = "phyto_clip.index"
MAPPING_FILE = "phyto_filenames.pkl"

L2_THRESHOLD = 0.35 
LPIPS_THRESHOLD = 0.25 

def build_index_if_needed():
    if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
        return

    print("[INFO] Building PhytoTrace Database...")
    image_files = glob.glob(os.path.join(SEARCH_DIRECTORY, "*.png")) + \
                  glob.glob(os.path.join(SEARCH_DIRECTORY, "*.jpg"))
    
    url_file_path = os.path.join(SEARCH_DIRECTORY, "urls.txt")
    if os.path.exists(url_file_path):
        with open(url_file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip().startswith(("http://", "https://"))]
            image_files.extend(urls)

    if not image_files:
        print("[WARNING] No images found in the search directory. Index build skipped.")
        return

    print(f"      Total items to process: {len(image_files)}")

    embeddings_list = []
    valid_files = []
    
    get_clip_components()
    
    for i, fpath in enumerate(image_files):
        img = load_image(fpath)
        if img:
            embeddings_list.append(extract_clip_image_embedding(img))
            valid_files.append(fpath)
        
        if i % 10 == 0: 
            print(f"      Processed {i} items...", end="\r")
            
    if not embeddings_list:
        print("\n[WARNING] Failed to extract features from any images.")
        return

    vectors = np.vstack(embeddings_list)
    index = faiss.IndexFlatL2(512)
    index.add(vectors)
    
    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, 'wb') as f:
        pickle.dump(valid_files, f)
    print(f"\n[INFO] Index built with {index.ntotal} phytopathology samples.")

def run_batch_scan():
    print("==========================================")
    print("   PhytoTrace: Batch Evaluator            ")
    print("==========================================")
    
    build_index_if_needed()
    
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(MAPPING_FILE, 'rb') as f:
            filenames = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")
        return

    target_img = load_image(TARGET_IMAGE_PATH)
    if not target_img: 
        return
    
    # Retrieves both transformed images and their embeddings
    target_variations = get_geometric_variations(target_img)
    query_vectors = np.vstack([var[1] for var in target_variations])
    
    print(f"[INFO] Scanning against target: {os.path.basename(TARGET_IMAGE_PATH)}")
    
    D, I = index.search(query_vectors, k=10)
    
    found_candidates = set()
    
    print("-" * 65)
    print(f"{'FILE':<35} | {'L2 DIST':<7} | {'LPIPS':<6} | {'RESULT'}")
    print("-" * 65)

    get_lpips_model()

    for variant_idx, (distances, indices) in enumerate(zip(D, I)):
        # Extract the specific spatially transformed image that triggered this query
        aligned_target_img = target_variations[variant_idx][0]

        for dist, idx in zip(distances, indices):
            if idx == -1: continue
            
            candidate_file = filenames[idx]
            is_url = candidate_file.startswith(("http://", "https://"))

            if not is_url:
                if os.path.abspath(candidate_file) == os.path.abspath(TARGET_IMAGE_PATH):
                    continue
            
            if candidate_file in found_candidates:
                continue

            if dist <= L2_THRESHOLD:
                candidate_img = load_image(candidate_file)
                if candidate_img is None:
                    continue

                # Use the spatially aligned target image for accurate VGG comparison
                lpips_val = calculate_lpips_distance(aligned_target_img, candidate_img)
                
                status = "PASS"
                if lpips_val <= LPIPS_THRESHOLD:
                    status = "MEMORIZED"
                
                display_name = os.path.basename(candidate_file)
                if is_url:
                    display_name = candidate_file.split("/")[-1]
                if len(display_name) > 35:
                    display_name = display_name[:32] + "..."

                print(f"{display_name:<35} | {dist:<7.3f} | {lpips_val:.3f}   | {status}")
                
                if status == "MEMORIZED":
                    found_candidates.add(candidate_file)

    print("-" * 65)
    print(f"[REPORT] Total Verified Duplicates: {len(found_candidates)}")
    print("==========================================")

if __name__ == "__main__":
    run_batch_scan()
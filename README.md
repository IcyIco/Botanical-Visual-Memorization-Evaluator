# PhytoTrace: Botanical Visual Memorization Evaluator

## Overview

PhytoTrace is an advanced, multimodal auditing framework designed to detect data memorization and assess generative mimicry in Artificial Intelligence Generated Content (AIGC), with a specialized focus on plant pathology and botanical datasets. 

Unlike traditional heuristic or hash-based duplicate detectors, PhytoTrace leverages Vision Foundation Models (CLIP) and Learned Perceptual Neural Metrics (LPIPS) to determine if a generative model has merely "memorized" training examples rather than synthesizing novel biological features. This provides a rigorous metric for dataset traceability and intellectual property assessment in computational agriculture.

## Key Features

* Deep Semantic Representation: Utilizes OpenAI's CLIP (Contrastive Language-Image Pretraining) to extract high-dimensional (512-D) semantic embeddings, entirely replacing fragile classical computer vision techniques like Perceptual Hashing (pHash).
* Neural Perceptual Verification: Implements LPIPS (Learned Perceptual Image Patch Similarity) with a VGG backbone to calculate true human-like perceptual shifts, overcoming the rigid pixel-to-pixel limitations of SSIM.
* Multimodal Text-Image Alignment: Employs the CLIP Text Encoder to evaluate how strictly the generated botanical image adheres to the original generation prompt (e.g., text-to-image semantic similarity).
* Geometric Robustness: Automatically extracts augmented feature vectors for geometric variations (rotations at 90, 180, and 270 degrees, plus horizontal flips) to ensure high recall rates against augmented generation.
* Scalable Vector Database: Implements FAISS (Facebook AI Similarity Search) utilizing IndexFlatL2 to convert O(N) linear scanning into high-speed dense vector retrieval, capable of handling large-scale phenotyping datasets.

## Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager
* (Optional but recommended) CUDA-enabled GPU or Apple Silicon (MPS) for accelerated deep learning inference.

### Setup

1. Clone the repository:
    git clone https://github.com/your-username/PhytoTrace.git
    cd PhytoTrace

2. Install the required deep learning and processing dependencies:
    pip install -r requirements.txt
    
    (Note: Dependencies include torch, torchvision, transformers, lpips, faiss-cpu, streamlit, etc. Model weights for CLIP and VGG will be downloaded automatically upon first execution.)

## Usage

### 1. Batch Scanning (CLI)

The main.py script serves as the high-throughput batch evaluation engine. 

1. Open main.py and configure your directories:
    # Path to the specific generated botanical image to audit
    TARGET_IMAGE_PATH = "C:/path/to/generated_image.png"

    # Directory containing the original training/reference dataset
    SEARCH_DIRECTORY = "C:/path/to/training_data"

2. Execute the scanner:
    python main.py
    
    * First Run: The system will extract 512-D semantic features for all reference images and build the FAISS index (phyto_clip.index). 
    * Subsequent Runs: The system loads the pre-built dense index instantly for millisecond-level semantic search.

### 2. Multimodal Visual Analysis (GUI)

For detailed side-by-side evaluation, perceptual threshold tuning, and NLP prompt alignment:

    streamlit run app.py

## Interpretation of Results

PhytoTrace utilizes a rigorous Semantic-to-Perceptual cascading strategy. A definitive "Memorization" verdict requires passing both the semantic proximity threshold and the neural perceptual threshold.

| Metric | Threshold | Interpretation |
| :--- | :--- | :--- |
| Semantic Distance (L2) | <= 0.35 | Coarse Semantic Match: Derived from CLIP embeddings. Indicates the images share identical fundamental concepts and deep semantic traits. Lower is better (0.0 is identical). |
| Perceptual Shift (LPIPS) | <= 0.25 | Fine Perceptual Confirmation: Derived from the VGG neural network. Confirms the semantic match is also visually and structurally mimicking the source. Lower is better. |
| Text-Image Alignment | N/A | Prompt Adherence: Cosine similarity indicating how well the generated image reflects the provided textual prompt. |

### Verdict Logic

* HIGH MEMORIZATION RISK: Semantic L2 <= 0.35 AND LPIPS <= 0.25. The generative model has leaked and closely reconstructed a training sample.
* SEMANTIC RETENTION (WARNING): Semantic L2 <= 0.35 BUT LPIPS > 0.25. The image shares deep semantic traits but differs perceptually (e.g., severe style-transfer or generative hallucination based on the reference).
* GENERATIVE NOVELTY: Semantic L2 > 0.35. No significant semantic link to the training data.

## Methodology

1. Foundation Model Extraction: All reference botanical images are processed through the CLIP Vision Transformer to extract L2-normalized 512-D feature vectors.
2. Dense Vector Indexing: Vectors are aggregated into a FAISS IndexFlatL2 database for rapid Euclidean distance querying.
3. Augmented Querying: Target images are geometrically augmented; all variations are processed through CLIP and queried against the FAISS index to find the nearest semantic neighbors.
4. Deep Perceptual Verification: The top candidate matches are processed through the LPIPS neural metric to compute the true perceptual distance, delivering the final verdict.

## License

This project is licensed under the MIT License.
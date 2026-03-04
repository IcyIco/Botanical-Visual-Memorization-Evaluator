import torch
import requests
import numpy as np
import lpips
from PIL import Image, ImageOps
from io import BytesIO
from typing import Union, Optional, List
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

print("[INFO] Setting up PyTorch AI Engine...")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Lazy loading variables to prevent Streamlit memory leaks
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_LPIPS_MODEL = None

def get_clip_components():
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        print("[INFO] Loading CLIP Vision-Language Foundation Model...")
        model_id = "openai/clip-vit-base-patch32"
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_id).to(device)
        _CLIP_MODEL.eval()
    return _CLIP_PROCESSOR, _CLIP_MODEL

def get_lpips_model():
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        print("[INFO] Loading LPIPS Perceptual Metric (VGG)...")
        _LPIPS_MODEL = lpips.LPIPS(net='vgg').to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL

def load_image(source: Union[str, BytesIO]) -> Optional[Image.Image]:
    """Loads an image and ensures it is in 3-channel RGB format."""
    try:
        if hasattr(source, "read"):
            img = Image.open(source)
        elif isinstance(source, str) and source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(source)
        return img.convert('RGB')
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        return None

def extract_clip_image_embedding(image: Image.Image) -> np.ndarray:
    """Extracts 512-D visual embedding with L2 normalization."""
    processor, model = get_clip_components()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().astype('float32')[0]

def extract_clip_text_embedding(text: str) -> np.ndarray:
    """Extracts 512-D textual embedding for NLP alignment."""
    processor, model = get_clip_components()
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().astype('float32')[0]

def get_geometric_embeddings(image: Image.Image) -> List[np.ndarray]:
    """Extracts embeddings for geometric variations to boost recall robustness."""
    embeddings = [
        extract_clip_image_embedding(image),
        extract_clip_image_embedding(ImageOps.mirror(image))
    ]
    for angle in [90, 180, 270]:
        embeddings.append(extract_clip_image_embedding(image.rotate(angle, expand=True)))
    return embeddings

# Standardized PyTorch transforms for safe tensor operations
lpips_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def calculate_lpips_distance(img1: Image.Image, img2: Image.Image) -> float:
    """Calculates perceptual distance using deep neural networks."""
    lpips_model = get_lpips_model()
    t1 = lpips_transform(img1).unsqueeze(0).to(device)
    t2 = lpips_transform(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        distance = lpips_model(t1, t2)
    return float(distance.item())

def get_l2_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.linalg.norm(emb1 - emb2))
    
def get_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))
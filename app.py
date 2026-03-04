import streamlit as st
import numpy as np
from utils import (
    load_image, extract_clip_image_embedding, extract_clip_text_embedding,
    get_geometric_embeddings, calculate_lpips_distance, get_l2_distance, get_cosine_similarity,
    get_clip_components, get_lpips_model
)

st.set_page_config(page_title="PhytoTrace Eval", layout="wide")

@st.cache_resource
def init_models():
    """Initializes models once to prevent Streamlit from reloading them on every interaction."""
    get_clip_components()
    get_lpips_model()

init_models()

st.title("PhytoTrace: Botanical Visual Memorization Evaluator")
st.markdown("""
A multimodal AI framework evaluating generative mimicry in plant pathology datasets. 
Powered by **CLIP (Vision-Language Alignment)** and **LPIPS (Learned Perceptual Neural Metrics)**.
""")

st.sidebar.header("Evaluation Thresholds")
l2_thresh = st.sidebar.slider("Semantic L2 Distance (CLIP)", 0.0, 1.5, 0.35, help="Lower means tighter semantic match.")
lpips_thresh = st.sidebar.slider("Perceptual Distance (LPIPS)", 0.0, 1.0, 0.25, help="Lower means structurally and perceptually identical.")

st.subheader("NLP Modality")
text_prompt = st.text_input("Generation Prompt (e.g., 'A close-up of a wheat leaf with severe stripe rust')", "")

col1, col2 = st.columns(2)
img1, img2 = None, None

with col1:
    st.subheader("Training Reference (Botany Data)")
    f1 = st.file_uploader("Upload Reference", key="f1")
    if f1: 
        img1 = load_image(f1)
        st.image(img1, use_container_width=True)

with col2:
    st.subheader("AIGC Generated Image")
    f2 = st.file_uploader("Upload Generated", key="f2")
    if f2: 
        img2 = load_image(f2)
        st.image(img2, use_container_width=True)

if img1 and img2:
    st.markdown("---")
    st.subheader("Multimodal Analysis Results")
    
    with st.spinner("Running Deep Inference..."):
        emb1 = extract_clip_image_embedding(img1)
        variations_emb2 = get_geometric_embeddings(img2)
        
        distances = [get_l2_distance(emb1, e_var) for e_var in variations_emb2]
        best_l2 = min(distances)
        best_emb2_idx = np.argmin(distances)
        best_emb2 = variations_emb2[best_emb2_idx]
        
        lpips_val = calculate_lpips_distance(img1, img2)
        
        alignment_score = "N/A"
        if text_prompt:
            text_emb = extract_clip_text_embedding(text_prompt)
            alignment_score = f"{get_cosine_similarity(text_emb, best_emb2):.3f}"
    
    m1, m2, m3 = st.columns(3)
    with m1: 
        st.metric("Semantic Distance (L2)", f"{best_l2:.3f}")
    with m2: 
        st.metric("Perceptual Shift (LPIPS)", f"{lpips_val:.3f}")
    with m3: 
        st.metric("Text-Image Alignment", alignment_score)
        
    is_mem = (best_l2 <= l2_thresh) and (lpips_val <= lpips_thresh)
    if is_mem:
        st.error("[ALERT] High Memorization Risk: The generated image is perceptually mimicking the training data.")
    elif best_l2 <= l2_thresh:
        st.warning("[WARNING] Semantic Retention: Shares deep semantic traits but differs perceptually (Style-transfer or minor hallucination).")
    else:
        st.success("[PASS] Generative Novelty: No significant memorization detected.")
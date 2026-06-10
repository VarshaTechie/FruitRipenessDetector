import os
import json
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

# TensorFlow is optional at UI runtime; handle missing TF gracefully.
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

# ─── Page Config ───
st.set_page_config(
    page_title="Fruit Ripeness Detection",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Hide sidebar, hamburger menu, and default footer ───
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Premium CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="css"], .stMarkdown, p, span, label, div {
    font-family: 'Outfit', sans-serif !important;
}
.stApp {
    background: #f9fafb;
}

/* ── Hero ── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: #111827;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #4b5563;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 400;
}

/* ── Cards ── */
.glass-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
    margin-bottom: 1.5rem;
}

/* ── Upload Zone ── */
.upload-zone {
    border: 2.5px dashed #c8cee0;
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    background: linear-gradient(135deg, #f8f9fd 0%, #f0f2f8 100%);
    transition: all 0.3s ease;
    cursor: pointer;
}
.upload-zone:hover {
    border-color: #FF6B4B;
    background: linear-gradient(135deg, #fff5f3 0%, #fff0ec 100%);
}
.upload-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.upload-text {
    font-size: 1.1rem;
    color: #6c757d;
    font-weight: 500;
}
.upload-hint {
    font-size: 0.85rem;
    color: #a0a7b8;
    margin-top: 0.4rem;
}

/* ── Result Badge ── */
.result-card {
    text-align: center;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    color: white;
    margin-bottom: 1.2rem;
}
.result-card.ripe {
    background: linear-gradient(135deg, #00c853 0%, #2ecc71 50%, #69f0ae 100%);
    box-shadow: 0 8px 30px rgba(46,204,113,0.35);
}
.result-card.rotten {
    background: linear-gradient(135deg, #e53935 0%, #ef5350 50%, #ff8a80 100%);
    box-shadow: 0 8px 30px rgba(229,57,53,0.35);
}
.result-card.unripe {
    background: linear-gradient(135deg, #ff9800 0%, #ffb74d 50%, #ffe082 100%);
    box-shadow: 0 8px 30px rgba(255,152,0,0.35);
}
.result-label {
    font-size: 0.9rem;
    font-weight: 500;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
.result-class {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.5px;
}
.result-confidence {
    font-size: 1.3rem;
    font-weight: 600;
    opacity: 0.95;
    margin-top: 0.3rem;
}

/* ── Recommendation Card ── */
.rec-card {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1rem;
}
.rec-card.ripe {
    background: #e8f5e9;
    border-left: 4px solid #2ecc71;
}
.rec-card.rotten {
    background: #ffebee;
    border-left: 4px solid #e53935;
}
.rec-card.unripe {
    background: #fff8e1;
    border-left: 4px solid #ff9800;
}
.rec-emoji { font-size: 1.8rem; }
.rec-text {
    font-size: 1rem;
    font-weight: 500;
    color: #333;
}
.rec-sub {
    font-size: 0.85rem;
    color: #777;
    margin-top: 2px;
}

/* ── Section Titles ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #3a3f55;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Progress Bars ── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.8rem;
}
.prob-label {
    min-width: 65px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #444;
}
.prob-bar-bg {
    flex: 1;
    height: 12px;
    background: #e8eaf0;
    border-radius: 6px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}
.prob-bar-fill.ripe { background: linear-gradient(90deg, #2ecc71, #69f0ae); }
.prob-bar-fill.rotten { background: linear-gradient(90deg, #e53935, #ff8a80); }
.prob-bar-fill.unripe { background: linear-gradient(90deg, #ff9800, #ffe082); }
.prob-pct {
    min-width: 55px;
    font-weight: 700;
    font-size: 0.9rem;
    color: #333;
    text-align: right;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stButton"] > button:first-child {
    border: none !important;
}

/* ── Image Styling ── */
.uploaded-img-container {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #ddd, transparent);
    margin: 1.5rem 0;
}

/* ── Upload layout (custom drag-drop) ── */
.upload-card {
    max-width: 760px;
    margin: 0 auto;
}

/* Streamlit default file uploader is kept visible and functional */

.drag-drop-area {
    border: 2.5px dashed #c8cee0;
    border-radius: 20px;
    padding: 2.7rem 2rem;
    text-align: center;
    background: linear-gradient(135deg, #f8f9fd 0%, #f0f2f8 100%);
    transition: all 0.25s ease;
}
.drag-drop-area:hover {
    border-color: #FF6B4B;
    background: linear-gradient(135deg, #fff5f3 0%, #fff0ec 100%);
}
.drag-drop-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #3a3f55;
    margin-top: 0.8rem;
}
.drag-drop-sub {
    font-size: 0.95rem;
    color: #6c757d;
    margin-top: 0.3rem;
    font-weight: 500;
}
.drag-drop-cta {
    display: inline-block;
    margin-top: 1rem;
    font-weight: 800;
    color: #ff6b4b;
}

@media (max-width: 700px) {
    .hero-title { font-size: 2.6rem; }
    .upload-card { padding: 0 0.4rem; }
}


/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #a0a7b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading (Cached) ───
@st.cache_resource
def load_model():
    if tf is None:
        return None
    path = 'fruit_ripeness_model.keras'
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_class_names():
    path = 'model_metadata.json'
    default = ["Ripe", "Rotten", "Unripe"]
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f).get('class_names', default)
        except Exception:
            pass
    return default

model = load_model()
class_names = load_class_names()

# Recommendation data
RECOMMENDATIONS = {
    "Ripe": {
        "emoji": "✅",
        "title": "Ready to eat",
        "text": "Looks ripe! Enjoy it fresh or add it to your next snack.",
        "css": "ripe"
    },
    "Rotten": {
        "emoji": "🗑️",
        "title": "Discard fruit",
        "text": "This fruit appears spoiled. For safety, discard it.",
        "css": "rotten"
    },
    "Unripe": {
        "emoji": "⏳",
        "title": "Allow more time to ripen",
        "text": "Not quite there yet—keep it at room temperature and check again later.",
        "css": "unripe"
    }
}


# ─── Hero Section ───
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Fruit Ripeness Detection 🍎</div>
    <div class="hero-subtitle">
        Upload a fruit image and instantly detect whether it is
        <strong>Ripe</strong>, <strong>Rotten</strong>, or <strong>Unripe</strong>
        — powered by deep learning.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Session State ───
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'pred_label' not in st.session_state:
    st.session_state.pred_label = None
if 'pred_probs' not in st.session_state:
    st.session_state.pred_probs = None

# ─── Check Model ───
if tf is None:
    st.error("🚫 TensorFlow is not installed. Please install it via `pip install tensorflow`.")
    st.stop()
elif model is None:
    st.error("🚫 Model not found. Please run `train.py` first to generate `fruit_ripeness_model.keras`.")
    st.stop()


# ─── Session Helpers ───
def clear_results():
    keys_to_delete = ['prediction_done', 'pred_label', 'pred_probs', 'uploaded_file_name', 'uploaded_file_bytes']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# ─── Load Image from State ───
if "uploaded_file_bytes" in st.session_state and st.session_state.uploaded_file_bytes:
    image = Image.open(BytesIO(st.session_state.uploaded_file_bytes)).convert('RGB')
else:
    image = None

# ─── Main Interface ───
if image is None:
    # Upload Zone
    uploaded_file = st.file_uploader(
        "Upload a fruit image",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    
    if uploaded_file is not None and st.session_state.get("uploaded_file_name") != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
        st.rerun()
        
    st.markdown("<div style='text-align:center; color:#6b7280; margin: 1.5rem 0 1rem;'>Or try a sample image:</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    with c2:
        if st.button("🍎 Sample Ripe", use_container_width=True):
            with open("samples/ripe.jpg", "rb") as f:
                st.session_state.uploaded_file_bytes = f.read()
                st.session_state.uploaded_file_name = "sample_ripe.jpg"
            st.rerun()
    with c3:
        if st.button("🗑️ Sample Rotten", use_container_width=True):
            with open("samples/rotten.jpg", "rb") as f:
                st.session_state.uploaded_file_bytes = f.read()
                st.session_state.uploaded_file_name = "sample_rotten.jpg"
            st.rerun()

else:
    # Preview & Results Zone
    col_img, col_result = st.columns([1, 1], gap="large")
    
    with col_img:
        st.markdown("<div class='glass-card' style='padding: 1.5rem;'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        
        # Action Buttons
        if not st.session_state.get('prediction_done'):
            if st.button("🔍 Analyze Fruit", use_container_width=True, type="primary"):
                with st.spinner("🧠 Analyzing..."):
                    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
                    arr = np.array(resized, dtype=np.float32)
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                    batch = np.expand_dims(arr, axis=0)

                    preds = model.predict(batch, verbose=0)[0]
                    pred_idx = int(np.argmax(preds))

                    st.session_state.pred_label = class_names[pred_idx]
                    st.session_state.pred_probs = preds
                    st.session_state.prediction_done = True
                st.rerun()
            
            if st.button("🗑️ Remove Image", use_container_width=True):
                clear_results()
                st.rerun()
        else:
            if st.button("🔄 Scan Another Fruit", use_container_width=True):
                clear_results()
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_result:
        if st.session_state.get('prediction_done'):
            pred_label = st.session_state.pred_label
            pred_probs = st.session_state.pred_probs
            pred_idx = class_names.index(pred_label)
            confidence = float(pred_probs[pred_idx])
            css_class = pred_label.lower()

            # Main Result Card
            st.markdown(f"""
            <div class="result-card {css_class}" style="margin-bottom: 1rem;">
                <div class="result-label">Ripeness</div>
                <div class="result-class">{pred_label.upper()}</div>
                <div class="result-confidence">{confidence * 100:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation
            rec = RECOMMENDATIONS.get(pred_label)
            rec = rec if rec else RECOMMENDATIONS["Ripe"]
            st.markdown(f"""
            <div class="rec-card {rec['css']}" style="margin-top: 0; margin-bottom: 1rem;">
                <div class="rec-emoji">{rec['emoji']}</div>
                <div>
                    <div class="rec-text">{rec['title']}</div>
                    <div class="rec-sub">{rec['text']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Class Probabilities List
            st.markdown("<div class='glass-card' style='padding:1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title' style='font-size:0.95rem;'>Probability Breakdown</div>", unsafe_allow_html=True)
            bar_classes = ["ripe", "rotten", "unripe"]
            for i, cname in enumerate(class_names):
                prob = float(pred_probs[i])
                width_pct = prob * 100
                bc = bar_classes[i]
                st.markdown(f"""
                <div class="prob-row" style="margin-bottom: 0.4rem;">
                    <div class="prob-label" style="font-size: 0.85rem;">{cname}</div>
                    <div class="prob-bar-bg" style="height: 8px;">
                        <div class="prob-bar-fill {bc}" style="width: {width_pct:.1f}%;"></div>
                    </div>
                    <div class="prob-pct" style="font-size: 0.85rem;">{width_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Empty state before analysis
            st.markdown("""
            <div class='glass-card' style='height: 100%; display: flex; align-items: center; justify-content: center; min-height: 300px;'>
                <div style='text-align: center; color: #9ca3af;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;'>✨</div>
                    <div style='font-weight: 500;'>Image uploaded successfully!</div>
                    <div style='font-size: 0.9rem; margin-top: 0.3rem;'>Click Analyze Fruit to see results.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───
st.markdown("""
<div class="app-footer">
    Built with ❤️ using TensorFlow & Streamlit • MobileNetV2 Transfer Learning
</div>
""", unsafe_allow_html=True)

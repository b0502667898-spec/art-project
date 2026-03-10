import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="זיהוי אמנים | Art Recognition",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Goya', 'Monet', 'Raphael', 'Vincent_van_Gogh', 'William_Blake']
IMG_SIZE    = 180
MODEL_PATH  = "art_weights_v2.weights.h5"
GDRIVE_FILE_ID = "1T1qmTFfAoEhV9DcWIvkqmxaTxONglkyl"

ARTIST_HEBREW = {
    "Goya":             "פרנסיסקו גויה",
    "Monet":            "קלוד מונה",
    "Raphael":          "רפאל",
    "Vincent_van_Gogh": "וינסנט ואן גוך",
    "William_Blake":    "וויליאם בלייק",
}
ARTIST_YEARS = {
    "Goya":             "1746 – 1828",
    "Monet":            "1840 – 1926",
    "Raphael":          "1483 – 1520",
    "Vincent_van_Gogh": "1853 – 1890",
    "William_Blake":    "1757 – 1827",
}
ARTIST_STYLE = {
    "Goya":             "רומנטיציזם / ריאליזם",
    "Monet":            "אימפרסיוניזם",
    "Raphael":          "רנסנס",
    "Vincent_van_Gogh": "פוסט-אימפרסיוניזם",
    "William_Blake":    "רומנטיציזם / סימבוליזם",
}

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Raleway:wght@300;400;500&display=swap');

:root {
    --cream:  #F5F0E8;
    --gold:   #C9A84C;
    --dark:   #1A1410;
    --muted:  #8C7B6B;
    --accent: #8B2635;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--cream) !important;
    color: var(--dark);
    font-family: 'Raleway', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.gallery-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--gold);
    margin-bottom: 2rem;
    position: relative;
}
.gallery-header::before,
.gallery-header::after {
    content: '◆';
    color: var(--gold);
    font-size: 1rem;
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
}
.gallery-header::before { left: 0; }
.gallery-header::after  { right: 0; }

.gallery-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--dark);
    letter-spacing: 0.04em;
    margin: 0;
    line-height: 1.15;
}
.gallery-subtitle {
    font-family: 'Raleway', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    color: var(--muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

[data-testid="stFileUploader"] > div:first-child {
    border: 1.5px dashed var(--gold) !important;
    background: rgba(201,168,76,0.04) !important;
    border-radius: 4px !important;
    transition: background 0.2s;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    background: rgba(201,168,76,0.09) !important;
}

.result-card {
    background: var(--dark);
    color: var(--cream);
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-top: 1.4rem;
    text-align: center;
    border-top: 3px solid var(--gold);
    direction: rtl;
}
.result-card .label {
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.3rem;
    font-family: 'Raleway', sans-serif;
    font-weight: 500;
}
.result-card .artist-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.2rem 0 0.1rem;
    color: #fff;
}
.result-card .artist-years {
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    font-weight: 300;
    margin-bottom: 0.6rem;
}
.result-card .artist-style {
    font-size: 0.82rem;
    color: #C9C0B0;
    margin-bottom: 1rem;
    font-weight: 300;
}

.conf-wrap { margin-top: 0.8rem; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
    font-family: 'Raleway', sans-serif;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    height: 6px;
    width: 100%;
}
.conf-bar-fill {
    height: 6px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--gold) 0%, #E8C96A 100%);
}

.mini-bars { margin-top: 1.2rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1rem; }
.mini-row  { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.45rem; }
.mini-name { font-size: 0.72rem; width: 120px; text-align: right; color: #C9C0B0; white-space: nowrap; }
.mini-bg   { flex: 1; height: 4px; background: rgba(255,255,255,0.08); border-radius: 2px; }
.mini-fill { height: 4px; border-radius: 2px; background: rgba(201,168,76,0.55); }
.mini-pct  { font-size: 0.68rem; color: var(--muted); width: 36px; text-align: left; }

.stButton > button {
    background: var(--dark) !important;
    color: var(--cream) !important;
    border: 1.5px solid var(--gold) !important;
    border-radius: 2px !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2.2rem !important;
    transition: background 0.2s, color 0.2s !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: var(--dark) !important;
}

.ornament { text-align: center; color: var(--gold); font-size: 1.1rem; margin: 1rem 0; letter-spacing: 0.6em; }

.err-box {
    border: 1px solid var(--accent);
    background: rgba(139,38,53,0.06);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    color: var(--accent);
    font-size: 0.84rem;
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_safe(path: str):
    import os, subprocess, sys

    if not os.path.exists(path):
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            import gdown
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", path, quiet=False, fuzzy=True)

    inp = tf.keras.Input(shape=(180, 180, 3))
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inp, training=False)
    x = tf.keras.layers.RandomRotation(0.2)(x, training=False)
    x = tf.keras.layers.RandomContrast(0.2)(x, training=False)
    x = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(x)
    base = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_tensor=x)
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    model.load_weights(path)
    return model

def preprocess(image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gallery-header">
    <p class="gallery-subtitle">פרויקט בגרות · זיהוי אמנות</p>
    <h1 class="gallery-title">Art Recognition</h1>
    <p class="gallery-subtitle">העלה יצירה וגלה את האמן שמאחוריה</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
with st.spinner("טוען מודל..."):
    try:
        model = load_model_safe(MODEL_PATH)
        model_ok = True
    except Exception as e:
        model_ok = False
        st.markdown(f'<div class="err-box">⚠️ שגיאה בטעינת המודל: <code>{e}</code></div>',
                    unsafe_allow_html=True)

# ─── Upload & Predict ──────────────────────────────────────────────────────────
st.markdown('<div class="ornament">✦ ✦ ✦</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "גרור לכאן תמונת יצירת אמנות, או לחץ לבחירה",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read()))

    col_l, col_c, col_r = st.columns([1, 6, 1])
    with col_c:
        st.image(image, use_container_width=True, caption=uploaded.name)

    st.markdown("")

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        run = st.button("🎨  זיהוי אמן", use_container_width=True)

    if run:
        if not model_ok:
            st.markdown('<div class="err-box">לא ניתן לזהות — המודל לא נטען.</div>',
                        unsafe_allow_html=True)
        else:
            with st.spinner("מנתח את היצירה..."):
                inp_arr = preprocess(image)
                preds   = model.predict(inp_arr, verbose=0)[0]

            top_idx  = int(np.argmax(preds))
            top_name = CLASS_NAMES[top_idx]
            top_prob = float(preds[top_idx]) * 100
            heb_name = ARTIST_HEBREW[top_name]
            years    = ARTIST_YEARS[top_name]
            style    = ARTIST_STYLE[top_name]

            sorted_idx = np.argsort(preds)[::-1]

            mini_html = '<div class="mini-bars">'
            for i in sorted_idx:
                n    = CLASS_NAMES[i]
                p    = float(preds[i]) * 100
                hn   = ARTIST_HEBREW[n]
                bold = "color:#fff;" if i == top_idx else ""
                mini_html += f"""
                <div class="mini-row">
                    <span class="mini-name" style="{bold}">{hn}</span>
                    <div class="mini-bg"><div class="mini-fill" style="width:{p:.1f}%"></div></div>
                    <span class="mini-pct">{p:.1f}%</span>
                </div>"""
            mini_html += '</div>'

            st.markdown(f"""
            <div class="result-card">
                <div class="label">האמן שזוהה</div>
                <div class="artist-name">{heb_name}</div>
                <div class="artist-years">{years}</div>
                <div class="artist-style">{style}</div>
                <div class="conf-wrap">
                    <div class="conf-label">
                        <span>ביטחון</span>
                        <span>{top_prob:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{top_prob:.1f}%"></div>
                    </div>
                </div>
                {mini_html}
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1rem;
            border-top:1px solid #D4C5A9; color:#8C7B6B;
            font-size:0.72rem; letter-spacing:0.1em; font-family:'Raleway',sans-serif;">
    פרויקט בגרות · ResNet50V2 Transfer Learning · TensorFlow / Keras
</div>
""", unsafe_allow_html=True)

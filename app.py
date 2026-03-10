import os
# הגדרות סביבה חייבות להיות ראשונות
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import requests
from PIL import Image

# ניסיון ייבוא חסין תקלות עבור TensorFlow ו-Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        import keras
        from keras.models import load_model
    except ImportError:
        st.error("Keras/TensorFlow not found. Please check requirements.txt")
        st.stop()

# ─── פונקציית הורדה מהדרייב ──────────────────────────────────────────────────
def download_file_from_google_drive(url, destination):
    if not os.path.exists(destination):
        with st.spinner('טוען את המודל מהענן...'):
            file_id = url.split('/')[-2]
            direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            response = requests.get(direct_url, stream=True)
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

# ─── הגדרות דף ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="מזהה האמנים", page_icon="🎨")

# ─── טעינת המודל ─────────────────────────────────────────────────────────────
MODEL_PATH = "my_art_model.h5"
DRIVE_URL = "https://drive.google.com/file/d/1kIZPNmXPCGHn4IXB-nxSubwpnwnvyr2e/view?usp=sharing"

download_file_from_google_drive(DRIVE_URL, MODEL_PATH)

if 'model' not in st.session_state:
    try:
        # שימוש בטעינה ללא קומפילציה למניעת שגיאות גרסה
        st.session_state.model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {e}")
        st.stop()

model = st.session_state.model

# רשימת האמנים
ARTISTS = ["Goya", "Monet", "Raphael", "Vincent_van_Gogh", "William_Blake"]
HEB_NAMES = {
    "Goya": "פרנסיסקו גויה", "Monet": "קלוד מונה", "Raphael": "רפאל",
    "Vincent_van_Gogh": "וינסנט ואן גוך", "William_Blake": "וויליאם בלייק"
}

# ─── ממשק האתר ──────────────────────────────────────────────────────────────
st.title("🎨 מזהה האמנים הגדול")
st.write("העלה תמונה של ציור, והבינה המלאכותית תזהה מי צייר אותו!")

uploaded_file = st.file_uploader("בחר תמונה...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='התמונה שהועלתה', use_container_width=True)
    
    # עיבוד תמונה לפי הגודל שהמודל שלך מצפה לו (180x180)
    img = image.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # ביצוע חיזוי
    predictions = model.predict(img_array)
    
    # חישוב הסתברויות ב-Numpy
    exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
    score = exp_preds / exp_preds.sum()
    
    predicted_artist = ARTISTS[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"המודל מזהה שזהו ציור של: **{HEB_NAMES[predicted_artist]}**")
    st.info(f"רמת ביטחון: {confidence:.2f}%")

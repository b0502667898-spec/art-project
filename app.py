import os

# הוספת הגדרה קריטית למניעת שגיאת TypeError בטעינת המודל
# זה חייב להופיע לפני ייבוא tensorflow
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
# במקום ה-from הקודם, נשתמש בזה:
load_model = tf.keras.models.load_model
import tensorflow.keras.backend as K
from PIL import Image
import numpy as np
import requests

# ─── פונקציית הורדה מהדרייב ──────────────────────────────────────────────────
def download_file_from_google_drive(url, destination):
    if not os.path.exists(destination):
        with st.spinner('טוען את המודל מהענן... זה קורה רק בפעם הראשונה'):
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

# הורדה וטעינה
download_file_from_google_drive(DRIVE_URL, MODEL_PATH)

# טעינה באמצעות מנגנון ה-Legacy כדי למנוע קריסה
if 'model' not in st.session_state:
    K.clear_session()
    st.session_state.model = load_model(MODEL_PATH, compile=False)

model = st.session_state.model

# רשימת האמנים (לפי הסדר שהמודל אומן)
ARTISTS = ["Goya", "Monet", "Raphael", "Vincent_van_Gogh", "William_Blake"]
HEB_NAMES = {
    "Goya": "פרנסיסקו גויה",
    "Monet": "קלוד מונה",
    "Raphael": "רפאל",
    "Vincent_van_Gogh": "וינסנט ואן גוך",
    "William_Blake": "וויליאם בלייק"
}

# ─── ממשק האתר ──────────────────────────────────────────────────────────────
st.title("🎨 מזהה האמנים הגדול")
st.write("העלה תמונה של ציור, והבינה המלאכותית תזהה מי צייר אותו!")

uploaded_file = st.file_uploader("בחר תמונה...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='התמונה שהועלתה', use_container_width=True)
    
    # עיבוד התמונה למודל
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # חיזוי
    predictions = model.predict(img_array)
    # שימוש ב-Softmax רגיל של Numpy למניעת תקלות גרסה
    score = np.exp(predictions[0]) / np.sum(np.exp(predictions[0]))
    predicted_artist = ARTISTS[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"המודל מזהה שזהו ציור של: **{HEB_NAMES[predicted_artist]}**")
    st.info(f"רמת ביטחון: {confidence:.2f}%")


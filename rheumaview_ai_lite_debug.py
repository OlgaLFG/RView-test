import streamlit as st
from PIL import Image
import os
from collections import defaultdict

# Шаблонный классификатор по названию файла
def classify_region(filename):
    fname = filename.lower()
    if any(k in fname for k in ["cerv", "c-spine"]):
        return "Cervical Spine"
    elif any(k in fname for k in ["thor", "t-spine"]):
        return "Thoracic Spine"
    elif any(k in fname for k in ["lum", "l-spine"]):
        return "Lumbar Spine"
    elif "sacrum" in fname or "sacro" in fname or "si" in fname:
        return "Pelvis / SI Joints"
    elif "pelvis" in fname:
        return "Pelvis / SI Joints"
    elif "hip" in fname:
        return "Hips"
    elif "hand" in fname or "mcp" in fname:
        return "Hands"
    elif "foot" in fname or "mtp" in fname:
        return "Feet"
    elif "knee" in fname:
        return "Knees"
    elif "shoulder" in fname:
        return "Shoulders"
    else:
        return "Unknown"

st.set_page_config(page_title="RheumaView-lite", page_icon="🦴", layout="wide")
st.title("🦴 RheumaView-lite Debug App")
st.markdown("Загрузите изображения. Приложение определит предполагаемый анатомический регион и сгруппирует их.")

uploaded_files = st.file_uploader(
    "Загрузите изображения (JPG, PNG, WEBP, BMP, TIFF)", 
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Загружено файлов: {len(uploaded_files)}")

    grouped = defaultdict(list)
    for file in uploaded_files:
        region = classify_region(file.name)
        grouped[region].append(file)

    for region, files in grouped.items():
        st.subheader(f"📂 {region} – {len(files)} файл(ов)")
        cols = st.columns(3)
        for i, file in enumerate(files):
            with cols[i % 3]:
                try:
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_container_width=True)
                except:
                    st.warning(f"Не удалось отобразить: {file.name}")
else:
    st.info("Файлы не загружены.")

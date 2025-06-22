
import streamlit as st
from PIL import Image
import os
from collections import defaultdict

# Автоопределение региона по имени и размерам
def classify_region(filename, image=None):
    fname = filename.lower()

    # 1. По ключевым словам в названии
    if any(k in fname for k in ["cerv", "c-spine"]):
        return "Cervical Spine"
    elif any(k in fname for k in ["thor", "t-spine"]):
        return "Thoracic Spine"
    elif any(k in fname for k in ["lum", "l-spine"]):
        return "Lumbar Spine"
    elif "sacro" in fname or "si" in fname or "sij" in fname:
        return "SI Joints / Pelvis"
    elif "pelvis" in fname or "iliac" in fname:
        return "SI Joints / Pelvis"
    elif "hand" in fname or "mcp" in fname or "wrist" in fname:
        return "Hands"
    elif "foot" in fname or "mtp" in fname:
        return "Feet"
    elif "knee" in fname:
        return "Knees"
    elif "shoulder" in fname:
        return "Shoulders"
    elif "hip" in fname:
        return "Hips"

    # 2. По размеру изображения
    if image:
        width, height = image.size
        ratio = height / width if width else 0

        if ratio > 1.5:
            return "Spine (Tall Image)"
        elif width > 1200 and height > 900:
            return "Pelvis / SI Joints"
        elif width < 600 and height < 600:
            return "Hands or Feet"
        elif 0.8 < ratio < 1.2 and width > 800:
            return "Chest or Pelvis (Square)"
    return "Unknown"

st.set_page_config(page_title="RheumaView-lite v3", page_icon="🧠", layout="wide")
st.title("🧠 RheumaView-lite Debug App v3")
st.markdown("Загрузите изображения. Приложение попытается определить анатомический регион по названию и/или структуре изображения.")

uploaded_files = st.file_uploader(
    "Загрузите изображения (JPG, PNG, WEBP, BMP, TIFF)", 
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Загружено файлов: {len(uploaded_files)}")

    grouped = defaultdict(list)
    for file in uploaded_files:
        try:
            image = Image.open(file)
            region = classify_region(file.name, image)
            grouped[region].append((file.name, image.copy()))
        except:
            grouped["Unreadable"].append((file.name, None))

    for region, entries in grouped.items():
        st.subheader(f"📂 {region} – {len(entries)} файл(ов)")
        cols = st.columns(3)
        for i, (fname, img) in enumerate(entries):
            with cols[i % 3]:
                if img:
                    st.image(img, caption=fname, width=250)
                else:
                    st.warning(f"Не удалось открыть: {fname}")
else:
    st.info("Файлы не загружены.")

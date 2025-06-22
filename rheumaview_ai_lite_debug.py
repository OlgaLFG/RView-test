import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="RheumaView-lite", page_icon="🩻", layout="wide")

st.title("🩺 RheumaView-lite Debug App")
st.markdown("Загрузите изображения для первичной отладки интерфейса приложения RheumaView.")

uploaded_files = st.file_uploader(
    "Загрузите изображения (JPG, PNG, WEBP, BMP, TIFF)", 
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Загружено файлов: {len(uploaded_files)}")

    for idx, file in enumerate(uploaded_files):
        st.markdown(f"**Файл {idx + 1}: {file.name}**")
        try:
            image = Image.open(file)
            st.image(image, caption=file.name, use_column_width=True)
        except Exception as e:
            st.error(f"Ошибка при отображении файла {file.name}: {e}")
else:
    st.info("Файлы не загружены.")

import streamlit as st
from PIL import Image
import os
from collections import defaultdict

def classify_region(filename, image=None):
    fname = filename.lower()

    if any(k in fname for k in ["cerv", "c-spine"]):
        return "Cervical Spine"
    elif any(k in fname for k in ["thor", "t-spine"]):
        return "Thoracic Spine"
    elif any(k in fname for k in ["lum", "l-spine"]):
        return "Lumbar Spine"
    elif any(k in fname for k in ["sacro", "sij", "si", "pelvis", "iliac"]):
        return "Pelvis / SI Joints"
    elif any(k in fname for k in ["hand", "mcp", "wrist", "fingers"]):
        return "Hands"
    elif any(k in fname for k in ["foot", "mtp", "toes", "hallux"]):
        return "Feet"
    elif any(k in fname for k in ["knee"]):
        return "Knees"
    elif any(k in fname for k in ["shoulder", "ac joint"]):
        return "Shoulders"
    elif any(k in fname for k in ["hip"]):
        return "Hips"
    elif any(k in fname for k in ["elbow", "forearm", "radius", "ulna"]):
        return "Elbows / Forearms"
    elif any(k in fname for k in ["ankle"]):
        return "Ankles"
    elif "bcr-2017" in fname:
        return "Hands or Feet (from BCR-2017)"

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

st.set_page_config(page_title="RheumaView-lite v3.1", page_icon="🦴", layout="wide")
st.title("🦴 RheumaView-lite Debug App v3.1")
st.markdown("Upload radiographic images. The app will attempt to auto-classify anatomical regions using filename and image structure.")

uploaded_files = st.file_uploader(
    "Upload files (JPG, PNG, WEBP, BMP, TIFF)", 
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")

    grouped = defaultdict(list)
    for file in uploaded_files:
        try:
            image = Image.open(file)
            region = classify_region(file.name, image)
            grouped[region].append((file.name, image.copy()))
        except:
            grouped["Unreadable"].append((file.name, None))

    for region, entries in grouped.items():
        st.subheader(f"📂 {region} – {len(entries)} file(s)")
        cols = st.columns(3)
        for i, (fname, img) in enumerate(entries):
            with cols[i % 3]:
                if img:
                    st.image(img, caption=fname, width=250)
                else:
                    st.warning(f"Unreadable: {fname}")

    st.markdown("---")
    st.info("⬇️ READY button and report generation will be added in the next version.")
else:
    st.info("No files uploaded yet.")

import streamlit as st
from PIL import Image
from collections import defaultdict

REGION_LIST = [
    "Cervical Spine", "Thoracic Spine", "Lumbar Spine",
    "Pelvis / SI Joints", "Hips", "Knees", "Ankles", "Feet", "Hands",
    "Shoulders", "Elbows / Forearms", "Wrists", "Long Bones", "Unknown"
]

def predict_region(image):
    return "Unknown"  # stub for future model

st.set_page_config(page_title="RheumaView-lite v4.2-fallback-select", page_icon="🦴", layout="wide")
st.title("🦴 RheumaView-lite v4.2 – Fallback Select Enabled")
st.markdown("Upload radiographs. Manual region override is enabled if prediction fails.")

uploaded_files = st.file_uploader("Upload X-ray images", type=["jpg", "webp", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)

if "user_override" not in st.session_state:
    st.session_state.user_override = {}

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")
    grouped = defaultdict(list)

    for file in uploaded_files:
        image = Image.open(file)
        region = predict_region(image)

        # Manual override for unknown
        if region == "Unknown":
            region = st.selectbox(f"Select region for: {file.name}", options=REGION_LIST, key=file.name)

        st.session_state.user_override[file.name] = region
        grouped[region].append((file.name, image.copy()))

    for region, files in grouped.items():
        st.subheader(f"{region} — {len(files)} file(s)")
        cols = st.columns(4)
        for i, (fname, img) in enumerate(files):
            with cols[i % 4]:
                st.image(img, caption=fname, width=180)

    if st.button("✅ READY – Generate Report"):
        st.subheader("📝 Report Summary")
        for region, files in grouped.items():
            st.markdown(f"- **{region}**: {len(files)} file(s)")
            if region in st.session_state.user_override.values():
                st.markdown(f"  - AI: not available *(manual override)*")
else:
    st.info("No files uploaded.")

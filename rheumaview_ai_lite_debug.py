import streamlit as st
from PIL import Image
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

st.set_page_config(page_title="RheumaView-lite v4", page_icon="🩻", layout="wide")
st.title("🩻 RheumaView-lite Debug App v4")
st.markdown("Upload radiographic images. The app will classify regions and generate a basic report upon confirmation.")

uploaded_files = st.file_uploader(
    "Upload files (JPG, PNG, WEBP, BMP, TIFF)", 
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"], 
    accept_multiple_files=True
)

# Initialize session state
if "grouped_data" not in st.session_state:
    st.session_state.grouped_data = None
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

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

    st.session_state.grouped_data = grouped

    for region, entries in grouped.items():
        st.subheader(f"📂 {region} – {len(entries)} file(s)")
        cols = st.columns(4)
        for i, (fname, img) in enumerate(entries):
            with cols[i % 4]:
                if img:
                    st.image(img, caption=fname, width=180)
                else:
                    st.warning(f"Unreadable: {fname}")

    if st.button("✅ READY – Generate Report"):
        st.session_state.report_ready = True

    if st.session_state.report_ready:
        st.markdown("---")
        st.subheader("📝 Basic Report Summary")
        total = sum(len(v) for v in grouped.values())
        st.write(f"**Total images:** {total}")
        for region, entries in grouped.items():
            st.write(f"- **{region}**: {len(entries)} file(s)")

        st.success("Report ready. Export and AI-enhanced interpretation will be available in the next version.")
else:
    st.info("No files uploaded yet.")


import streamlit as st
from PIL import Image
from collections import defaultdict

def ai_interpret(region, count):
    region = region.lower()
    if "sij" in region or "pelvis" in region:
        if count >= 4:
            return ("Findings suggest sacroiliitis.", "High")
        elif 2 <= count < 4:
            return ("Findings may reflect early sacroiliitis.", "Moderate")
        else:
            return ("Findings may reflect degenerative change or early sacroiliitis.", "Low")
    elif "hand" in region:
        if count >= 5:
            return ("Findings suggest inflammatory arthropathy (e.g., RA).", "High")
        else:
            return ("Findings may reflect early inflammatory or mechanical arthropathy.", "Moderate")
    elif "spine" in region:
        return ("Findings may reflect degenerative spondylosis. Recommend clinical correlation.", "Moderate")
    else:
        return ("No definitive features detected.", "Low")

def classify_region(filename, image=None):
    fname = filename.lower()
    if any(k in fname for k in ["sacro", "sij", "si", "pelvis", "iliac"]):
        return "Pelvis / SI Joints"
    elif any(k in fname for k in ["hand", "mcp", "wrist", "fingers"]):
        return "Hands"
    elif any(k in fname for k in ["foot", "mtp", "toes"]):
        return "Feet"
    elif any(k in fname for k in ["cerv", "spine", "c-spine"]):
        return "Spine"
    return "Unknown"

st.set_page_config(page_title="RheumaView-lite v4.1-ai", page_icon="🧠", layout="wide")
st.title("🧠 RheumaView-lite Debug App v4.1-ai")
st.markdown("Prototype: AI-generated regional summaries with confidence-based phrasing.")

uploaded_files = st.file_uploader(
    "Upload radiographs (JPG, PNG, TIFF, etc.)", 
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"], 
    accept_multiple_files=True
)

if "grouped_data" not in st.session_state:
    st.session_state.grouped_data = None
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

if uploaded_files:
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

        # AI summary
        if region != "Unknown":
            summary, confidence = ai_interpret(region, len(entries))
            st.markdown(f"**AI Summary** ({confidence} confidence): {summary}")
        else:
            st.info("No AI summary for Unknown region.")

    if st.button("✅ READY – Generate Final Report"):
        st.session_state.report_ready = True

    if st.session_state.report_ready:
        st.markdown("---")
        st.subheader("📝 Final Draft Report")
        total = sum(len(v) for v in grouped.values())
        st.write(f"**Total images uploaded:** {total}")
        for region, entries in grouped.items():
            st.write(f"- **{region}**: {len(entries)} files")
            if region != "Unknown":
                summary, confidence = ai_interpret(region, len(entries))
                st.write(f"  - AI: {summary} *(Confidence: {confidence})*")
else:
    st.info("No files uploaded yet.")

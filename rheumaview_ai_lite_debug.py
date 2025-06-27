import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from inference_core import predict_region
from docx import Document
from datetime import datetime

# Custom class labels
CLASS_NAMES = [
    "Cervical Spine", "Thoracic Spine", "Lumbar Spine", "Pelvis/SI Joints/ Sacrum", 
    "Hips", "Knees", "Ankles/Feet", 
    "Shoulders", "Elbows", "Hands/Wrists", "Long bones"
]

st.set_page_config(page_title="RheumaView Lite", layout="centered")
st.title("🟩 RheumaView™ Lite")
st.caption("Curated by Dr. Olga Goodman • Region classifier demo")

uploaded_files = st.file_uploader("Upload X-ray images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

results = []

if uploaded_files:
    st.markdown(f"**Total files uploaded:** {len(uploaded_files)}")
    st.subheader("Image Preview, Prediction, and Manual Override")

    for file in uploaded_files:
        image = Image.open(file).convert("L")
        st.image(image, caption=f"Preview: {file.name}", width=120)

        # Convert to RGB to match model expectation
        image_rgb = image.convert("RGB")
        top3 = predict_region(image_rgb)
        top_label = CLASS_NAMES[top3[0][0]]

        st.markdown(f"**Top prediction:** {top_label}")
        st.markdown("**Confidence breakdown:**")
        for idx, prob in top3:
            st.markdown(f"- {CLASS_NAMES[idx]}: {prob:.2%}")

        manual = st.selectbox(f"Override region for {file.name}?", CLASS_NAMES, index=top3[0][0], key=file.name)
        results.append((file.name, manual))

# Generate EMR Summary
if st.button("Generate EMR Summary"):
    emr_text = []
    region_count = {}
    for _, region in results:
        region_count[region] = region_count.get(region, 0) + 1

    for region, count in region_count.items():
        emr_text.append(f"{region} – {count} view(s)")

    emr_summary = "Study includes: " + "; ".join(emr_text) + "."
    st.success("EMR Summary:")
    st.code(emr_summary, language="markdown")

# Generate report
if st.button("Generate Report (.docx)"):
    doc = Document()
    doc.add_heading("RheumaView™ Radiology Report", level=1)
    doc.add_paragraph("Curated by Dr. Olga Goodman")
    doc.add_paragraph(f"Date: {datetime.today().strftime('%Y-%m-%d')}")

    region_groups = {}
    for file_name, region in results:
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append(file_name)

    for region, files in region_groups.items():
        doc.add_heading(region, level=2)
        for fname in files:
            doc.add_paragraph(f"- {fname}")

    output_path = "/mnt/data/rheumaview_report.docx"
    doc.save(output_path)
    st.success("Report generated successfully!")
    st.download_button(label="Download Report", file_name="rheumaview_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", data=open(output_path, "rb").read())

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
from inference_core import predict_region
from docx import Document
from datetime import datetime

# Class names mapping
CLASS_NAMES = [
    "Cervical Spine", "Thoracic Spine", "Lumbar Spine", "Pelvis/SI Joints",
    "Hips", "Knees", "Ankles", "Feet",
    "Shoulders", "Elbows", "Wrists", "Hands", "Long bones"
]

st.set_page_config(page_title="RheumaView Lite", layout="centered")
st.title("🟩 RheumaView™ Lite")
st.caption("Curated by Dr. Olga Goodman • Region classifier demo")

uploaded_files = st.file_uploader("Upload X-ray images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

results = []

if uploaded_files:
    st.subheader("Image Preview and Predictions")
    for file in uploaded_files:
        image = Image.open(file).convert("L")
        st.image(image, caption=f"Preview: {file.name}", width=300)

        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor = transform(image).unsqueeze(0)

        # Prediction
        top3 = predict_region(tensor)

        top_label = CLASS_NAMES[top3[0][0]]
        st.markdown(f"**Top prediction:** {top_label}")
        st.markdown("**Confidence breakdown:**")
        for idx, prob in top3:
            st.markdown(f"- {CLASS_NAMES[idx]}: {prob:.2%}")

        results.append((file.name, top3))

# Generate EMR summary
if st.button("Generate EMR Summary"):
    emr_text = []
    region_count = {}
    for _, top3 in results:
        top_label = CLASS_NAMES[top3[0][0]]
        region_count[top_label] = region_count.get(top_label, 0) + 1

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

    for file_name, top3 in results:
        doc.add_heading(file_name, level=2)
        for idx, prob in top3:
            doc.add_paragraph(f"{CLASS_NAMES[idx]}: {prob:.2%}")

    output_path = "/mnt/data/rheumaview_report.docx"
    doc.save(output_path)
    st.success("Report generated successfully!")
    st.download_button(label="Download Report", file_name="rheumaview_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", data=open(output_path, "rb").read())

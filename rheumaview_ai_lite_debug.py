import streamlit as st
import torch
from PIL import Image
from collections import defaultdict
from inference_core import region_report

# Dummy model for grayscale images (1 channel)
class DummyRegionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(1 * 224 * 224, 10)

    def forward(self, x):
        return self.fc(self.flatten(x))

# Load model (CPU only, dummy weights)
@st.cache_resource
def load_model():
    model = DummyRegionModel()
    model.load_state_dict(torch.load("region_model.pt", map_location="cpu"))
    model.eval()
    return model

# Labels and threshold
REGION_LABELS = [
    "Cervical Spine", "Thoracic Spine", "Lumbar Spine",
    "Pelvis/SI/Sacrum", "Hands/Wrists", "Elbows", "Shoulders",
    "Hips", "Knees", "Ankles/Feet", "Feet", "Long Bones"
]
CONFIDENCE_THRESHOLD = 0.45

model = load_model()

# Upload and process files
st.title("🧠 RheumaView Region Classifier\n(Lite Debug Mode)")
st.markdown("Upload X-ray images")

uploaded_files = st.file_uploader(
    "Drag and drop files here", accept_multiple_files=True, type=["png", "jpg", "jpeg", "webp"]
)

grouped_files = defaultdict(list)

if uploaded_files:
    st.markdown("---")
    st.subheader("📁 Grouped Files by Region")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("L").resize((224, 224))
        image_tensor = torch.tensor([[[list(image.getdata())[i * 224:(i + 1) * 224] for i in range(224)]]], dtype=torch.float32) / 255.0
        outputs = model(image_tensor)
        confidences = torch.softmax(outputs, dim=1).squeeze().tolist()
        for idx, conf in enumerate(confidences):
            if conf > CONFIDENCE_THRESHOLD:
                region = REGION_LABELS[idx % len(REGION_LABELS)]
                grouped_files[region].append((uploaded_file.name, image, conf))

    displayed_files = set()
    for region, entries in grouped_files.items():
        unique_entries = [e for e in entries if e[0] not in displayed_files]
        if not unique_entries:
            continue
        st.markdown(f"**{region}** – {len(unique_entries)} file(s)**")
        cols = st.columns(3)
        for i, (fname, img, conf) in enumerate(unique_entries):
            displayed_files.add(fname)
            with cols[i % 3]:
                st.image(img, caption=f"{fname}", width=180)
                st.caption(f"{conf:.2f}")

st.markdown("---")
st.subheader("🧠 Generate Report by Region")

region_options = ["Multiple Regions"] + REGION_LABELS
selected_region = st.selectbox("Choose region to generate report for:", region_options)

def summarize_text(text, max_chars=700):
    if len(text) <= max_chars:
        return text
    paragraphs = text.split("\n")
    summary = ""
    for para in paragraphs:
        if len(summary) + len(para) + 1 <= max_chars:
            summary += para + "\n"
        else:
            break
    return summary.strip()

if st.button("Generate EMR Summary"):
    if selected_region == "Multiple Regions":
        combined_report = ""
        for region in grouped_files.keys():
            report = region_report(region)
            combined_report += f"**{region}**:\n{report}\n\n"
        summary = summarize_text(combined_report, max_chars=1000)
        st.success(f"📄 EMR Summary for multiple regions:\n\n{summary}")
    else:
        report = region_report(selected_region)
        summary = summarize_text(report, max_chars=700)
        st.success(f"📄 EMR Summary for **{selected_region}**:\n\n{summary}")
else:
    if not uploaded_files:
        st.subheader("📄 Report Generator")
        if st.button("✅ READY – Generate Report"):
            st.success("📝 Report generation coming soon.")
        else:
            st.info("No files uploaded.")

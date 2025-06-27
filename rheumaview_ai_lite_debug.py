import streamlit as st
from collections import defaultdict
from PIL import Image
import torch
import torchvision.transforms as transforms
from inference_core import predict_region

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
    "Hips", "Knees", "Ankles/Feet", "Long Bones"
]
CONFIDENCE_THRESHOLD = 0.65

# Image transform
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)

# Prediction function
model = load_model()
def predict_region(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        sorted_probs, indices = torch.topk(probs, 3)
        result = [(REGION_LABELS[i], float(sorted_probs[n])) for n, i in enumerate(indices)]
        return result

# Dummy EMR report generator
def region_report(region):
    return f"This is a brief EMR-compatible summary for the **{region}** region. Findings and impressions go here."

# Page config
st.set_page_config(page_title="RheumaView v4.2-region-ai", page_icon="🧠", layout="wide")
st.title("🧠 RheumaView-lite v4.2 Region Classifier")
st.markdown("Visual AI classifier with fallback and AI/manual reporting.")

# Reset uploaded files
st.markdown("### 📁 File Upload Control")
if st.button("🔁 Reset Uploaded Files"):
    if "upload" in st.session_state:
        del st.session_state["upload"]
    st.session_state["once_per_session_reminder"] = True
    st.rerun()

# File upload
uploaded_files = st.file_uploader(
    "Upload X-rays",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
    accept_multiple_files=True,
    key="upload"
)

grouped = defaultdict(list)
selected_region = REGION_LABELS[0]

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        predictions = predict_region(image)
        top_label, top_conf = predictions[0]

        if top_conf < CONFIDENCE_THRESHOLD:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(image, width=120)
            with col2:
                selected = st.selectbox(
                    f"Select region for {file.name} (low confidence)",
                    REGION_LABELS,
                    key=file.name
                )
            region = selected
            st.session_state.region_override = st.session_state.get("region_override", {})
            st.session_state.region_override[file.name] = "manual"
        else:
            region = top_label
            st.session_state.region_override = st.session_state.get("region_override", {})
            st.session_state.region_override[file.name] = "AI"

        grouped[region].append((file.name, image.copy(), predictions))

    displayed_files = set()
    st.markdown("---")
    st.subheader("📁 Grouped Files by Region")
    for region, entries in grouped.items():
        unique_entries = [e for e in entries if e[0] not in displayed_files]
        if not unique_entries:
            continue
        st.markdown(f"**{region}** — {len(unique_entries)} file(s)")
        cols = st.columns(3)
        for i, (fname, img, preds) in enumerate(unique_entries):
            displayed_files.add(fname)
            with cols[i % 3]:
                st.image(img, caption=f"{fname}", width=180)
                st.caption(", ".join([f"{lbl} ({conf:.2f})" for lbl, conf in preds]))

# EMR Report Generator — always visible
st.markdown("---")
st.subheader("🧠 Generate Report by Region")
selected_region = st.selectbox("Choose region to generate report for:", REGION_LABELS)

if st.button("Generate EMR Summary"):
    report = region_report(selected_region)
    st.success(f"📝 EMR Summary for **{selected_region}**:

{report}")

# General report section (optional)
if uploaded_files:
    st.subheader("🧾 Report Generator")
    if st.button("✅ READY – Generate Report"):
        st.success("📄 Report generation coming soon.")
else:
    st.info("No files uploaded.")

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
from collections import defaultdict

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

# Predict region
def predict_region(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        sorted_probs, indices = torch.topk(probs, 3)
        result = [(REGION_LABELS[i], float(sorted_probs[n])) for n, i in enumerate(indices)]
    return result

# Header
st.set_page_config(page_title="RheumaView-lite", layout="wide")
st.title("🦴 RheumaView-lite v4.2")
st.markdown("Radiologic region classifier with manual fallback and preview.")

# Reset control
st.markdown("### 🗂️ File Upload Control")
if st.button("🔄 Reset Uploaded Files"):
    if "upload" in st.session_state:
        del st.session_state["upload"]
        st.session_state["once_per_session_reminder"] = True
        st.rerun()

if st.session_state.get("once_per_session_reminder"):
    st.warning("⚠️ Please refresh the page (F5) to fully clear the upload list.")
    del st.session_state["once_per_session_reminder"]

# Upload
uploaded_files = st.file_uploader(
    "Upload X-ray files",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
    accept_multiple_files=True,
    key="upload"
)

# Load model
model = load_model()

# Classification + fallback
if uploaded_files:
    grouped = defaultdict(list)
    st.session_state.region_override = {}
    displayed_files = set()

    for file in uploaded_files:
        image_bytes = file.read()
        predictions = predict_region(image_bytes)
        top_label, top_conf = predictions[0]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
            st.session_state.region_override[file.name] = "manual"
        else:
            region = top_label
            st.session_state.region_override[file.name] = "AI"

        grouped[region].append((file.name, image.copy(), predictions))

    st.markdown("---")
    st.subheader("📊 Grouped Files by Region")
    for region, entries in grouped.items():
        unique_entries = [e for e in entries if e[0] not in displayed_files]
        if not unique_entries:
            continue
        st.markdown(f"**{region} — {len(unique_entries)} file(s)**")
        cols = st.columns(3)
        for i, (fname, img, preds) in enumerate(unique_entries):
            displayed_files.add(fname)
            with cols[i % 3]:
                st.image(img, caption=f"{fname}", width=180)
                st.caption(", ".join([f"{lbl} ({conf:.2f})" for lbl, conf in preds]))

    st.markdown("---")
    st.subheader("📝 Report Generator")
    if st.button("✅ READY – Generate Report"):
        st.success("📄 Report generation coming soon.")
else:
    st.info("No files uploaded.")
    st.markdown("### 📄 Generate Report by Region")
    selected_region = st.selectbox("Choose region to generate report for:", REGION_LABELS)

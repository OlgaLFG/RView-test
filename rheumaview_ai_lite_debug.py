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
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = image.convert("L")
    return transform(image).unsqueeze(0)

# Generate report per region
def region_report(region):
    return f"Findings consistent with standard interpretation for {region}. No acute abnormalities noted."

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🧠 RheumaView Region Classifier")
st.caption("(Lite Debug Mode)")

uploaded_files = st.file_uploader("Upload X-ray images", type=["png", "jpg", "jpeg", "webp", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    grouped = defaultdict(list)

    for file in uploaded_files:
        image = Image.open(file)
        input_tensor = preprocess(image)
        predictions = predict_region(image)  # Pass PIL image directly
        top_label, top_conf = predictions[0]

        if top_conf < CONFIDENCE_THRESHOLD:
            st.warning(f"Low confidence prediction for {file.name}. Please verify region.")
        grouped[top_label].append((file.name, image.copy(), predictions))

    st.markdown("---")
    st.subheader("📂 Grouped Files by Region")
    displayed_files = set()
    for region, entries in grouped.items():
        unique_entries = [e for e in entries if e[0] not in displayed_files]
        if not unique_entries:
            continue
        st.markdown(f"**{region}** – {len(unique_entries)} file(s)")
        cols = st.columns(3)
        for i, (fname, img, preds) in enumerate(unique_entries):
            displayed_files.add(fname)
            with cols[i % 3]:
                st.image(img, caption=f"{fname}", width=180)
                st.caption(", ".join([f"{lbl} ({conf:.2f})" for lbl, conf in preds]))

# EMR Report Generator
st.markdown("---")
st.subheader("📝 Generate Report by Region")
selected_region = st.selectbox("Choose region to generate report for:", REGION_LABELS)

if st.button("Generate EMR Summary"):
    report = region_report(selected_region)
    st.success(f"📝 EMR Summary for **{selected_region}**:

{report}")

# Optional: General report generator (REQUIRES READY flag logic to activate)
if uploaded_files:
    st.subheader("📄 Report Generator")
    if st.button("✅ READY – Generate Report"):
        st.success("📋 Report generation coming soon.")
else:
    st.info("No files uploaded.")

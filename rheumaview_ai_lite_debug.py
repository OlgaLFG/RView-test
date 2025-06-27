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
    image = image.convert("L").resize((224, 224))
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor

# Group predictions
def group_by_region(files):
    model = load_model()
    grouped = defaultdict(list)
    for file in files:
        image = Image.open(file)
        input_tensor = preprocess(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_labels = torch.topk(probs, 3)
            predictions = [(REGION_LABELS[i], float(p)) for i, p in zip(top_labels, top_probs) if float(p) >= CONFIDENCE_THRESHOLD]
        for label in predictions:
            grouped[label[0]].append((file.name, image.copy(), predictions))
    return grouped

# Region report stub
def region_report(region):
    return f"Report for {region}: radiographic findings placeholder."

# Streamlit interface
st.title("🧠 RheumaView Region Classifier")
st.caption("(Lite Debug Mode)")

st.markdown("### Upload X-ray images")
uploaded_files = st.file_uploader(
    "Drag and drop files here",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "webp", "tif", "tiff"]
)

if uploaded_files:
    grouped = group_by_region(uploaded_files)
    st.markdown("---")
    st.subheader("📁 Grouped Files by Region")
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

# EMR Report Generator – always visible
st.markdown("---")
st.subheader("📝 Generate Report by Region")
selected_region = st.selectbox("Choose region to generate report for:", REGION_LABELS)

if st.button("Generate EMR Summary"):
    report = region_report(selected_region)
    st.success(f"📋 EMR Summary for **{selected_region}**:

{report}")

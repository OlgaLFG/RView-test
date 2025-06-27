
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
CONFIDENCE_THRESHOLD = 0.65

# Image transform
def preprocess(image):
    image = image.convert("L").resize((224, 224))
    tensor = torch.tensor([[[[pixel / 255.0 for pixel in list(image.getdata())[i:i+224]] for i in range(0, 224*224, 224)]]])
    return tensor.float()

# Prediction function
def predict_region(image):
    model = load_model()
    tensor = preprocess(image)
    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    sorted_probs = sorted(zip(REGION_LABELS, probs.tolist()), key=lambda x: x[1], reverse=True)
    return sorted_probs

# EMR-style report generator
def region_report(region_name):
    templates = {
        "Cervical Spine": "Cervical lordosis preserved. No disc space narrowing or erosive changes.",
        "Thoracic Spine": "No compressions or syndesmophytes. Disc heights preserved.",
        "Lumbar Spine": "Mild facet hypertrophy. No sacroiliitis. Normal alignment.",
        "Pelvis/SI/Sacrum": "No erosions, joint space narrowing, or asymmetric sclerosis of SI joints.",
        "Hands/Wrists": "Joint spaces preserved. No marginal erosions or periarticular osteopenia.",
        "Elbows": "No effusions or cortical irregularities.",
        "Shoulders": "No erosions or joint space narrowing. AC joints intact.",
        "Hips": "No superior migration or erosions. Joint spaces maintained.",
        "Knees": "No joint effusion. Medial and lateral compartments preserved.",
        "Ankles/Feet": "No erosions, overhanging edges, or soft tissue tophi.",
        "Feet": "No joint space narrowing or erosions. Alignment preserved.",
        "Long Bones": "No periosteal reaction or cortical destruction."
    }
    return templates.get(region_name, f"No abnormalities detected in {region_name}.")

# --- Main App ---
st.title("🧠 RheumaView Region Classifier (Lite Debug Mode)")

uploaded_files = st.file_uploader("Upload X-ray images", accept_multiple_files=True)
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
    st.subheader("📂 Grouped Files by Region")
    for region, entries in grouped.items():
        unique_entries = [e for e in entries if e[0] not in displayed_files]
        if not unique_entries:
            continue
        st.markdown(f"**{region} – {len(unique_entries)} file(s)**")
        cols = st.columns(3)
        for i, (fname, img, preds) in enumerate(unique_entries):
            displayed_files.add(fname)
            with cols[i % 3]:
                st.image(img, caption=f"{fname}", width=180)
                st.caption(", ".join([f"{lbl} ({conf:.2f})" for lbl, conf in preds]))

    st.markdown("---")
    st.subheader("🧠 Generate Report by Region")
    selected_region = st.selectbox("Choose region to generate report for:", REGION_LABELS)

    if st.button("Generate EMR Summary"):
        report = region_report(selected_region)
        st.success(f"📄 EMR Summary for **{selected_region}**:\n\n{report}")

else:
    st.subheader("🧾 Report Generator")
    if st.button("✅ READY – Generate Report"):
        st.success("📄 Report generation coming soon.")
    else:
        st.info("No files uploaded.")

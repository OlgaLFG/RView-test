import streamlit as st
from PIL import Image
from collections import defaultdict
import torch
import torchvision.transforms as transforms
import torch.nn as nn

REGION_LABELS = [
    "Cervical Spine", "Thoracic Spine", "Lumbar Spine",
    "Pelvis / SI Joints", "Hips", "Knees", "Ankles", "Feet",
    "Hands", "Shoulders", "Elbows", "Wrists", "Long Bones"
]

CONFIDENCE_THRESHOLD = 0.65

class DummyRegionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1 * 224 * 224, 10)

    def forward(self, x):
        return self.fc(self.flatten(x))

@st.cache_resource
def load_model():
    model = DummyRegionModel()
    model.load_state_dict(torch.load("region_model.pt", map_location="cpu"))
    model.eval()
    return model

def region_report(region_label):
    templates = {
        "Cervical Spine": "Straightening of cervical lordosis. Degenerative spondylosis suspected.",
        "Thoracic Spine": "No acute findings in thoracic spine. Vertebral body heights preserved.",
        "Lumbar Spine": "Lumbar spine with facet sclerosis and disc space narrowing at L4-L5.",
        "Feet": "No erosions or joint space narrowing noted in forefoot views.",
        "Pelvis / SI Joints": "Sacroiliac joints are symmetric. Mild subchondral sclerosis without erosions.",
        "Hands": "No erosions or joint space narrowing. Bone mineralization is preserved.",
    }
    return templates.get(region_label, "No region-specific findings available.")

def predict_region(image):
    model = load_model()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        sorted_probs, indices = torch.topk(probs, 3)
        result = [(REGION_LABELS[i], float(sorted_probs[n])) for n, i in enumerate(indices)]
        return result

st.set_page_config(page_title="RheumaView v4.2-region-ai", page_icon="🧠", layout="wide")
st.title("🧠 RheumaView-lite v4.2 Region Classifier")
st.markdown("Visual AI classifier with fallback and AI/manual reporting.")

# Reset uploaded files button
st.markdown("### 🔁 File Upload Control")
if st.button("🔄 Reset Uploaded Files"):
    if "upload" in st.session_state:
        del st.session_state["upload"]
    st.experimental_rerun()

# Upload area
uploaded_files = st.file_uploader(
    "Upload X-rays",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
    accept_multiple_files=True,
    key="upload"
)

if "region_override" not in st.session_state:
    st.session_state.region_override = {}

if uploaded_files:
    grouped = defaultdict(list)

    for file in uploaded_files:
        image = Image.open(file)
        predictions = predict_region(image)
        top_label, top_conf = predictions[0]

        if top_conf < CONFIDENCE_THRESHOLD:
            selected = st.selectbox(f"Select region for {file.name} (low confidence)", REGION_LABELS, key=file.name)
            region = selected
            st.session_state.region_override[file.name] = "manual"
        else:
            region = top_label
            st.session_state.region_override[file.name] = "AI"

        grouped[region].append((file.name, image.copy(), predictions))

    displayed_files = set()
    for region, entries in grouped.items():
        unique_entries = [e for e in entries if e[0] not in displayed_files]
        if not unique_entries:
            continue

        st.subheader(f"{region} — {len(unique_entries)} file(s)")
        cols = st.columns(3)

        for i, (fname, img, preds) in enumerate(unique_entries):
            displayed_files.add(fname)
            with cols[i % 3]:
                st.image(img, caption=fname, width=180)
                st.markdown(f"**Top prediction:** {preds[0][0]} ({preds[0][1]*100:.1f}%)")
                st.markdown(
                    f"*Other:* {preds[1][0]} ({preds[1][1]*100:.1f}%), "
                    f"{preds[2][0]} ({preds[2][1]*100:.1f}%)"
                )

    if st.button("✅ READY – Generate Report"):
        st.subheader("📝 Report Summary")
        for region, entries in grouped.items():
            st.markdown(f"- **{region}**: {len(entries)} file(s)")
            for fname, _, _ in entries:
                src = st.session_state.region_override.get(fname, "AI")
                st.markdown(f"  - {fname} — source: {src}")
                st.markdown(f"_{region_report(region)}_")
else:
    st.info("No files uploaded.")
    st.markdown("### 📄 Generate Report by Region")
    selected_region = st.selectbox("Choose region to generate report for:", REGION_LABELS)

if st.button("Generate EMR Summary"):
    report = region_report(selected_region)
    st.success(f"📝 EMR Summary for **{selected_region}**:\n\n{report}")

import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, List

# ===== Class names — must match training label order =====
class_names = [
    "AnklesFeet",
    "CervicalSpine",
    "Elbows",
    "HandsWrists",
    "Hips",
    "Knees",
    "LongBones",
    "LumbarSpine",
    "PelvisSISacrum",
    "Shoulders",
    "ThoracicSpine"
]

# ===== Load the trained model =====
model = torch.load("region_model.pt", map_location=torch.device("cpu"))
model.eval()

# ===== Image preprocessing =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== Predict from PIL Image =====
def predict_region(image: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)

    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
    predicted_class = probabilities.argmax().item()
    predicted_label = class_names[predicted_class]
    confidence = probabilities[predicted_class].item()

    topk = torch.topk(probabilities, k=3)
    top_indices = topk.indices.tolist()
    top_probs = topk.values.tolist()
    top_predictions = [(class_names[i], round(top_probs[j], 3)) for j, i in enumerate(top_indices)]

    return predicted_label, confidence, top_predictions

# ===== Optional: Predict from file path =====
def predict_region_from_path(image_path: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    image = Image.open(image_path).convert("RGB")
    return predict_region(image)

# ===== Option

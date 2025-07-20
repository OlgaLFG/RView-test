import torch
from torchvision import transforms
from PIL import Image

import torchvision.models as models

model = torch.load("region_model.pt", map_location=torch.device('cpu'))

model.eval()

# Your class names — must match training folder structure and order
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

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Inference function

def predict_region(image: Image.Image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        print("Model output:", outputs)  # Debug line

    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()

    # Top-1 prediction
    predicted_class = probabilities.argmax().item()
    predicted_label = class_names[predicted_class]
    confidence = probabilities[predicted_class].item()

    # Top-3 predictions
    topk = torch.topk(probabilities, k=3)
    top_indices = topk.indices.tolist()
    top_probs = topk.values.tolist()
    top_predictions = [(class_names[i], round(top_probs[j], 3)) for j, i in enumerate(top_indices)]

    return predicted_label, confidence, top_predictions

def region_report(region_name):
    return f"Auto-generated EMR summary for {region_name}. [This is a placeholder.]"

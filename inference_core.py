import torch
from torchvision import transforms
from PIL import Image

# Load your own trained model from file
model = torch.load("region_model.pt", map_location=torch.device('cpu'))
model.eval()

# Your class names — must match training folder structure and order
class_names = [
    "Ankles/Feet",
    "Cervical Spine",
    "Elbows",
    "Hands/Wrists",
    "Hips",
    "Knees",
    "Long bones",
    "Lumbar Spine",
    "Pelvis/SI Joints",
    "Shoulders",
    "Thoracic Spine"
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
    predicted_class = outputs.argmax().item()
    predicted_label = class_names[predicted_class]
    return predicted_label

def region_report(region_name):
    return f"Auto-generated EMR summary for {region_name}. [This is a placeholder.]"

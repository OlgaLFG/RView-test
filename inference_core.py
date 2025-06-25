import torch
from torchvision import transforms
from PIL import Image

# Load pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Example anatomical region labels (editable)
class_names = [
    "Cervical Spine", "Thoracic Spine", "Lumbar Spine", "Pelvis/SI Joints",
    "Hips", "Knees", "Ankles", "Feet",
    "Shoulders", "Elbows", "Wrists", "Hands", "Long bones" 
]

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Main inference function
def predict_region(image: Image.Image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
    predicted_class = outputs.argmax().item()
    predicted_label = class_names[predicted_class % len(class_names)]  # Demo labeling
    return predicted_label

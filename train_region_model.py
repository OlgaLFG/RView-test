
import os
import torch
from torch import nn, optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datetime import datetime

# Paths
data_dir = "region_dataset"  # make sure this is populated
model_path = "region_model.pt"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and loader
dataset = ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.3f} | Accuracy: {100.*correct/total:.2f}%")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
}, model_path)

print(f"Model saved to {model_path} with {num_classes} classes on {datetime.now().isoformat()}")

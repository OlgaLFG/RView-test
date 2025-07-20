
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# ===== Settings =====
data_dir = 'region_dataset'         # Folder with 11 subfolders (class names)
num_classes = 11
batch_size = 16
num_epochs = 5
learning_rate = 0.001
model_output_path = 'region_model.pt'

# ===== Image Transformations =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== Dataset and Loader =====
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== Model Definition =====
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ===== Loss and Optimizer =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===== Training Loop =====
print("Training started...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")

# ===== Save Model =====
torch.save(model, model_output_path)
print(f"Model saved to {model_output_path}")

# ===== Save Class Index Mapping =====
with open('class_mapping.txt', 'w') as f:
    f.write(str(dataset.class_to_idx))

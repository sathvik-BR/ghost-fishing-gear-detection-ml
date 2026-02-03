import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Model setup (same as before)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load('sonar_model.pth'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Test data loader
test_data = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f'Test set accuracy: {100 * correct / total:.2f}%')

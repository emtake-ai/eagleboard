import torch
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

train_dataset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', download=True, train=False, transform=transform)

train_loaded = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loaded = DataLoader(test_dataset, batch_size=128, shuffle=False)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, stride=1, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, stride=1, kernel_size=5, padding=0)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 1

for epoch in range(epochs):
    model.train()
    for images, labels in train_loaded:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loaded:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "lenet5.pth")
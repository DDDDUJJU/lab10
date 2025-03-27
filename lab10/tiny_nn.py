import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class DigitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg'):
                label = int(filename.split('_')[0])
                self.labels.append(0 if label == 1 else 1)  # 1->0, 2->1
                self.image_paths.append(os.path.join(data_dir, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

transform = transforms.Compose([
    transforms.Resize((8, 8)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = DigitDataset(data_dir='./digits', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class TinyNN(nn.Module):
    def __init__(self):
        super(TinyNN, self).__init__()
        self.fc = nn.Linear(8*8, 2)
    
    def forward(self, x):
        x = x.view(-1, 8*8)
        return self.fc(x)

model = TinyNN()
print(f"Model architecture:\n{model}\n")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100

def save_model_info(model, filename):
    with open(filename, 'w') as f:
        f.write(f"Tiny Neural Network (8x8->2)\n")
        f.write(f"Total params: {sum(p.numel() for p in model.parameters())}\n\n")
        
        for name, param in model.named_parameters():
            f.write(f"{name} ({tuple(param.shape)}):\n")
            f.write(f"{param.data}\n\n")

save_model_info(model, 'tiny_model_info.txt')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")


save_model_info(model, 'tiny_model_final.txt')
torch.save(model.state_dict(), 'tiny_model_weights.pth')

print("\nTraining completed. Model info saved to:")
print("- tiny_model_info.txt (initial)")
print("- tiny_model_final.txt (trained)")
print("- tiny_model_weights.pth (weights)")
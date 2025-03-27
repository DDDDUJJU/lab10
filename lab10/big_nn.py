import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# 数据集类
class DigitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍历目录中的文件
        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg'):
                label = int(filename.split('_')[0])  # 从文件名提取数字标签
                if label == 1:
                    self.labels.append(0)  # 将1映射为0
                elif label == 2:
                    self.labels.append(1)  # 将2映射为1
                self.image_paths.append(os.path.join(data_dir, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('L')  # 转为灰度图像
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((8, 8)),  # 确保所有图像都是 8x8
    transforms.ToTensor(),      # 转换为 Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载数据集
train_dataset = DigitDataset(data_dir='./digits', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 神经网络架构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(8*8, 16)  # 输入层到隐藏层
        self.fc2 = nn.Linear(16, 2)    # 隐藏层到输出层（2个类）

    def forward(self, x):
        x = x.view(-1, 8*8)  # 将8x8图像展平
        x = torch.relu(self.fc1(x))  # 隐藏层
        x = self.fc2(x)  # 输出层
        return x

# 初始化模型，损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练过程
num_epochs = 100

# 创建文件保存网络信息
def save_model_info(model, filename='model_info.txt'):
    with open(filename, 'w') as f:
        f.write("Neural Network Architecture Information\n")
        f.write("=======================================\n\n")
        
        # 保存模型结构
        f.write("Model Structure:\n")
        f.write(str(model))
        f.write("\n\n")
        
        # 保存每层的详细信息
        f.write("Layer Details:\n")
        f.write("-------------\n")
        for name, param in model.named_parameters():
            f.write(f"Layer: {name}\n")
            f.write(f"Shape: {tuple(param.shape)}\n")
            if 'weight' in name:
                f.write("Type: Weight\n")
            else:
                f.write("Type: Bias\n")
            f.write(f"Values:\n{param.data}\n\n")
        
        # 保存参数统计信息
        f.write("Parameter Statistics:\n")
        f.write("--------------------\n")
        total_params = 0
        for name, param in model.named_parameters():
            layer_params = param.numel()
            total_params += layer_params
            f.write(f"{name}: {layer_params} parameters\n")
        f.write(f"\nTotal Parameters: {total_params}\n")

# 在训练前保存初始模型信息
save_model_info(model, 'model_info_initial.txt')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # 向前传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播并更新权重
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 打印每个epoch的损失和准确率
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# 训练后保存最终模型信息
save_model_info(model, 'model_info_final.txt')

# 保存模型权重为PyTorch格式
torch.save(model.state_dict(), 'model_weights.pth')

print("Model information and weights have been saved to files.")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1) Load data and create a parallel array of indices
digits = load_digits()
X, y = digits.images, digits.target
indices = np.arange(len(X))

# 2) Threshold data for training, flatten, etc.
X_binary = np.where(X > 8, 1.0, 0.0).reshape(-1, 64).astype(np.float32)

# 3) Split both your data and indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_binary, y, indices, test_size=0.2, random_state=42
)

# 4) Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float16)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float16)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define your minimal model with ReLU instead of sigmoid
class MinimalFNN(nn.Module):
    def __init__(self):
        super(MinimalFNN, self).__init__()
        self.hidden = nn.Linear(64, 3, bias=False)  # 3 neurons in hidden layer
        self.output = nn.Linear(3, 10, bias=False)  # 10 output classes

    def forward(self, x):
        # Replace sigmoid with ReLU
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Instantiate model
model = MinimalFNN().half()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Train
num_epochs = 70
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.half()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate accuracy
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.half()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy = {100 * correct / total:.2f}%")

# 5) Show 10 random test images with predictions
num_images = 10
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(num_images):
    # Pick random index *within the test set*
    rand_idx = np.random.randint(0, len(X_test))

    # Map to original index
    original_idx = idx_test[rand_idx]

    # Run inference
    sample = X_test[rand_idx].view(1, -1).half()
    with torch.no_grad():
        output = model(sample)
        predicted_digit = torch.argmax(output).item()

    # Display the original 8x8 grayscale image
    axes[i].imshow(digits.images[original_idx], cmap='gray')
    axes[i].set_title(f"Pred: {predicted_digit}\nActual: {digits.target[original_idx]}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# Save weights as float16
hidden_weights = model.hidden.weight.detach().cpu().half().numpy()
output_weights = model.output.weight.detach().cpu().half().numpy()

np.savetxt("hidden_weights.txt", hidden_weights, fmt="%.5f")
np.savetxt("output_weights.txt", output_weights, fmt="%.5f")

print("Hidden layer weights saved to hidden_weights.txt (float16 in text form).")
print("Output layer weights saved to output_weights.txt (float16 in text form).")
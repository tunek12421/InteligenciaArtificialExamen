"""
Ejercicio 4: CNN para clasificación MNIST con PyTorch y GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore')

print("EJERCICIO 4: CNN PARA CLASIFICACIÓN DE IMÁGENES MNIST (PyTorch + GPU)")

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Cargar datos MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Dataset: {len(train_dataset)} train, {len(test_dataset)} test")

# Definir modelo CNN
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Bloque convolucional 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Bloque convolucional 3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Capas densas
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        # Bloque 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Bloque 2
        x = torch.relu(self.bn2(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Bloque 3
        x = torch.relu(self.bn3(self.conv5(x)))
        x = self.dropout3(x)
        
        # Flatten y capas densas
        x = x.view(-1, 128 * 7 * 7)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x

# Crear modelo y moverlo a GPU
model = MNIST_CNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parámetros: {total_params:,}")

# Configurar entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# Entrenamiento
epochs = 5
print(f"\nEntrenando por {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    train_acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    
    # Validación
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += target.size(0)
    
    val_acc = 100. * val_correct / val_total
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# Evaluación final
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

final_accuracy = accuracy_score(all_targets, all_preds)
cm = confusion_matrix(all_targets, all_preds)

print(f"\n=== RESULTADOS FINALES ===")
print(f"Test Accuracy: {final_accuracy:.4f}")
print(f"Total parámetros: {total_params:,}")
print(f"Device: {device}")

# Accuracy por clase
class_accuracies = cm.diagonal() / cm.sum(axis=1)
print(f"\nAccuracy por clase:")
for i, acc in enumerate(class_accuracies):
    print(f"Dígito {i}: {acc:.4f}")

# Información de memoria GPU
if torch.cuda.is_available():
    print(f"Memoria GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    torch.cuda.empty_cache()

print("EJERCICIO 4 COMPLETADO")
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_L_32_Weights

import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys
import time

# Devce Data
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION', )
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

start_time = time.time()
batch_size = 64


# Prepare Data
transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data_path = '../VIT-dog-cat/data/dataset_treino_e_teste/train'
test_data_path = '../VIT-dog-cat/data/dataset_treino_e_teste/test'

train_dataset = ImageFolder(root=train_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Hyperparameters
num_epochs = 4
# total_steps = len(train_loader)
total_steps = 20
learning_rate = 0.001



# Import pretrained model
model = models.vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1)

in_features = model.heads[-1].in_features
model.heads[-1] = nn.Linear(in_features, 1)
model.sigmoid = nn.Sigmoid()

# Freezing weights
for name, param in model.named_parameters():
    if name.startswith('head'):
        param.requires_grad = True  
    else:
        param.requires_grad = False
        
# Device CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f"\n\n Device ---> {device} and Current Device --> {torch.cuda.current_device()}\n\n")

# Traning the model
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# Loop traning
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):

    print("Training")
  
    running_loss = 0.0
    correct_train = 0
    total_train = 0


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    # Training
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs))
        correct_train += torch.sum(predicted == labels.unsqueeze(1)).item()
        total_train += labels.size(0)
        if step == total_steps:
            print("----------------X------------------")
            break
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], Loss: {loss.item():.4f}, Accuracy: {correct_train/total_train:.4f}')
        
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation each 10 epoch
    # if (epoch + 1) % 2 == 0:
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    print("Validation")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)

          loss = criterion(outputs, labels.float().unsqueeze(1))
          running_val_loss += loss.item()

          predicted = torch.round(torch.sigmoid(outputs))

          correct_val += (predicted == labels.unsqueeze(1)).sum().item()

          total_val += labels.size(0)

          if step == total_steps:
              print("----------------X------------------")
              break
          print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(test_loader)}], Loss Validation: {loss.item():.4f}, Accuracy Validation: {correct_val/total_val:.4f}')

    val_loss = running_val_loss / len(test_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    print("Tamanho das saídas:", outputs.size())
    predicted = torch.round(torch.sigmoid(outputs))
    print("Saídas binárias:", predicted)
    print("Tamanho dos rótulos:", labels.size())


torch.save(model.state_dict(), './models/modelo_vit_gpu.pth')

print("\n\n  %s minutos" % ((time.time() - start_time) / 60 ))

# Save graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig("./graph/loss_and_accuracy_pytorch.jpg")



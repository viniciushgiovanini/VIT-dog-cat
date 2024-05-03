import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
import torchvision.models as models
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torchvision

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
# Hyperparameters
batch_size = 24
num_epochs = 10

total_steps = 50
learning_rate = 0.001


# Prepare Data
train_data_path = '../VIT-dog-cat/data/dataset_treino_e_teste/train'
test_data_path = '../VIT-dog-cat/data/dataset_treino_e_teste/test'

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Carregar os dados de treinamento e teste
train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)

num_classes = len(train_dataset.classes)
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=1, ignore_mismatched_sizes=True)


criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Device CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\n Device ---> {device} and Current Device --> {torch.cuda.current_device()}\n\n")

# Traning the model
model.to(device)
print(model)

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
    
    loss_da_epoch = 0.0
    accurracy_da_epoch = 0.0


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # total_steps = len(train_loader)
    
    # Training
    model.train()
    for step, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()

      outputs = model(images).logits 
      loss = criterion(outputs, labels.float().unsqueeze(1))

      loss.backward()
      optimizer.step()

      predicted = torch.round(torch.sigmoid(outputs))
      correct_train = (predicted == labels.unsqueeze(1)).sum().item()
      total_train = labels.size(0)
      accuracy = correct_train / total_train

      loss_da_epoch += loss.item()
      accurracy_da_epoch += accuracy
      train_losses.append(loss_da_epoch/ total_steps)
      train_accuracies.append(accurracy_da_epoch/ total_steps)
      
      
      print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
      if step == total_steps:
              print("----------------X------------------")
              break

    # Validation each 10 epoch
    # if (epoch + 1) % 2 == 0:
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    loss_val_da_epoch = 0.0
    accurracy_val_da_epoch = 0.0
    
    print("Validation")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):
          images, labels = images.to(device), labels.to(device)

          outputs = model(images).logits
          loss = criterion(outputs, labels.float().unsqueeze(1))

          predicted = torch.round(torch.sigmoid(outputs))
          correct_val = (predicted == labels.unsqueeze(1)).sum().item()
          total_val = labels.size(0)
          accuracy_val = correct_val / total_val

          loss_val_da_epoch += loss.item()
          accurracy_val_da_epoch += accuracy_val

          print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], Loss Validation: {loss.item():.4f}, Accuracy Validation: {accuracy_val:.4f}')
          if step == total_steps:
              print("----------------X------------------")
              break

    val_losses.append(loss_val_da_epoch/ total_steps)
    val_accuracies.append(accurracy_val_da_epoch/ total_steps)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_da_epoch/ total_steps:.4f}, Train Accuracy: {accurracy_da_epoch/ total_steps:.4f}, Validation Loss: {(loss_val_da_epoch/ total_steps):.4f}, Validation Accuracy: {accurracy_val_da_epoch/ total_steps:.4f}')

    # print("Tamanho das saídas:", outputs.size())
    # predicted = torch.round(torch.sigmoid(outputs))
    # print("Saídas binárias:", predicted)
    # print("Tamanho dos rótulos:", labels.size())


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



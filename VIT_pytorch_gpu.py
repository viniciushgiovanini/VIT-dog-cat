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
import shutil
import pandas as pd
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

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
learning_rate = 0.008


# Prepare Data
train_data_path = '../VIT-dog-cat/data/dataset_treino_e_teste/train'
test_data_path = '../VIT-dog-cat/data/dataset_treino_e_teste/test'

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


if os.path.exists("./lightning_logs/"):
  shutil.rmtree("./lightning_logs/")
  


# Carregar os dados de treinamento e teste
train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)

num_classes = len(train_dataset.classes)
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=1, ignore_mismatched_sizes=True)


# Device CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\n Device ---> {device} and Current Device --> {torch.cuda.current_device()}\n\n")

class Modelo(pl.LightningModule):
    def __init__(self):
        super(Modelo, self).__init__()
        # Carregar um modelo pré-treinado
        self.model = model
        
        # Congelar os pesos das camadas pré-treinadas
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Substituir a última camada para um problema binário
        self.model.classifier = nn.Linear(self.model.config.hidden_size, 1)

        # Função de perda
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).logits
        loss = self.criterion(outputs, labels.float().unsqueeze(1)) 
        predicted = torch.round(torch.sigmoid(outputs))
        accuracy = (predicted == labels.unsqueeze(1)).float().mean()
                
        self.log('train_loss', loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).logits 
        loss = self.criterion(outputs, labels.float().unsqueeze(1)) 
        predicted = torch.round(torch.sigmoid(outputs))
        accuracy = (predicted == labels.unsqueeze(1)).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Dados
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=11)
val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=11)

# Modelo
modelo = Modelo()

# Save data progress
csv_logger = CSVLogger(
    save_dir='./lightning_logs/',
    name='csv_file'
)

# Treinador
trainer = pl.Trainer(max_epochs=num_epochs,  limit_train_batches= total_steps,limit_val_batches=total_steps, log_every_n_steps=1, logger=[csv_logger, TensorBoardLogger("./lightning_logs/")])

# Treinamento
trainer.fit(modelo, train_loader, val_loader)

torch.save(modelo.state_dict(), './models/modelo_vit_gpu.pth')

df = pd.read_csv('./lightning_logs/csv_file/version_0/metrics.csv')

print("\n\n  %s minutos" % ((time.time() - start_time) / 60 ))


# Data man
epochs = []
train_accuracy_means = []
train_loss_means = []
val_accuracy_uniques = []
val_loss_uniques = []

for epoch in df['epoch'].unique():
    dados_epoca = df[df['epoch'] == epoch]
    
    train_accuracy_mean = dados_epoca['train_accuracy'].mean()
    
    train_loss_mean = dados_epoca['train_loss'].mean()
    
    val_accuracy_unique = dados_epoca['val_accuracy'].mean()
    
    val_loss_unique = dados_epoca['val_loss'].mean()
    
    epochs.append(epoch)
    train_accuracy_means.append(train_accuracy_mean)
    val_accuracy_uniques.append(val_accuracy_unique)
    train_loss_means.append(train_loss_mean)
    val_loss_uniques.append(val_loss_unique)
    
    
resultados =pd.DataFrame({'epoch': epochs,
                                'train_accuracy': train_accuracy_means,
                                'val_accuracy': val_accuracy_uniques,
                                'train_loss': train_loss_means,
                                'val_loss': val_loss_uniques,                                    
                                },
                              )

# Save graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(resultados['epoch'], resultados['train_accuracy'], label='Train Accuracy')
plt.plot(resultados['epoch'], resultados['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(resultados['epoch'], resultados['train_loss'], label='Train Loss')
plt.plot(resultados['epoch'], resultados['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig("./graph/loss_and_accuracy_pytorch.jpg")



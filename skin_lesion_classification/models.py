# Skin Lesion Classification using Deep Learning

# Folder containing all the models to be trained

# Importing all the necessary libraries
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from efficientnet_pytorch import EfficientNet

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

def train_model(model_class, num_classes, name, full_dataset, generator):
    # Settings
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)              # 80% for training
    val_size = int(0.1 * total_size)                # 10% for validation
    test_size = total_size - train_size - val_size  # Remaining 10% for testing

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    # Create DataLoaders for the training, validation, and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate the model
    model = model_class(num_classes)

    early_stop_callback = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')

    # Instantiate the trainer
    trainer = pl.Trainer(max_epochs=10,
                         accelerator='gpu' if cuda_available else None,
                         logger=TensorBoardLogger("logs", name=name),
                         callbacks=[checkpoint_callback, early_stop_callback])

    # Train the model
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    trainer.test(model, test_dataloader)

# VGG
class VGGClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()

        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze all layers except the last one
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.vgg16.classifier[6].parameters():
            param.requires_grad = True

        # Modify the classifier layer for the specified number of classes
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        # Add a scheduler
        scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

        return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

# ResNet
class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()

        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)

        # Freeze all layers except the last one
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Modify the classifier layer for the specified number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        return self.resnet(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Add a scheduler
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
# AlexNet
class AlexNetClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(AlexNetClassifier, self).__init__()

        # Load a pre-trained AlexNet model
        self.alexnet = models.alexnet(pretrained=True)

        # Freeze all layers except the last one
        for param in self.alexnet.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.alexnet.classifier[6].parameters():
            param.requires_grad = True

        # Replace the classifier layer for the specified number of classes
        num_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Add a scheduler
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
        }


    def training_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
# EfficientNet

class EfficientNetClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()

        # Load a pre-trained EfficientNet model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

        # Freeze all layers except the last one
        for name, param in self.efficientnet.named_parameters():
            if '_fc' not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.efficientnet(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Add a scheduler
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
# Inception

class InceptionClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # Load a pre-trained Inception model
        self.inception = models.inception_v3(pretrained=True)

        # Freeze all layers except the last one
        for name, param in self.inception.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

        # Replace the classifier layer for the specified number of classes
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.inception(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Add a scheduler
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x).logits                     # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch                                # batch is a tuple containing the input data (x) and the target labels (y).
        logits = self(x)                            # pass the input data to the model to get the predicted logits.
        loss = F.cross_entropy(logits, y)           # cross-entropy loss between the model's predictions (logits) and the true labels (y).
        preds = torch.argmax(logits, dim=1)         # get the predicted labels by taking the argmax of the logits.
        acc = torch.sum(preds == y).item() / len(y) # calculate the accuracy by comparing the predicted labels to the true labels.

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
"""
A generic classifier model using PyTorch Lightning.

Can be configured to use various pre-trained models like ResNet, VGG, or
Vision Transformer (ViT). The final layer of the pre-trained model is replaced
to fit the number of classes in the current task.
"""
#**************************************************************************
# IMPORTS
#**************************************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import pytorch_lightning as pl
from transformers import ViTForImageClassification, ViTMAEForPreTraining
from efficientnet_pytorch import EfficientNet

#**************************************************************************
# CLASSIFIER MODEL
#**************************************************************************
class GenericClassifier(pl.LightningModule):
    """
    A flexible classifier that can use different pre-trained models.

    Handles the logic for training, validation, and testing steps
    as required by PyTorch Lightning.
    """
    def __init__(self, model_name, num_classes, learning_rate, class_weights):
        """
        Initialize the classifier.

        Args:
            model_name (str): The name of the model to use (e.g., "resnet", "vit").
            num_classes (int): The number of output classes.
            learning_rate (float): The learning rate for the optimizer.
            class_weights (torch.Tensor): A weight for the positive class to
                                          handle imbalanced data.
        """
        super().__init__()
        self.save_hyperparameters()  # Saves args to self.hparams

        self.learning_rate = learning_rate
        self.class_weights = class_weights

        # Dictionary to easily select a model.
        model_dict = {
            'vgg': models.vgg16, 'resnet': models.resnet18, 'alexnet': models.alexnet,
            'efficientnet': EfficientNet.from_pretrained, 'inception': models.inception_v3,
            'vit' : ViTForImageClassification.from_pretrained,
            'vitmae': ViTForImageClassification.from_pretrained
        }

        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} is not supported.")

        # Load the selected model and prepare it for transfer learning.
        if model_name == 'efficientnet':
            self.model = model_dict[model_name]('efficientnet-b0', num_classes=num_classes)
            # Freeze all layers except the final classifier layer.
            for name, param in self.model.named_parameters():
                if '_fc' not in name:
                    param.requires_grad = False
        elif model_name in ['vit', 'vitmae']:
            model_id = 'google/vit-base-patch16-224-in21k' if model_name == 'vit' else 'facebook/vit-mae-base'
            self.model = model_dict[model_name](model_id, num_labels=num_classes, ignore_mismatched_sizes=True)
            # Freeze all parameters except the final classifier head.
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        else:
            # Load standard torchvision models.
            self.model = model_dict[model_name](pretrained=True)
            # Freeze all layers by default.
            for param in self.model.parameters():
                param.requires_grad = False

            # Replace the final layer and make it trainable.
            if model_name in ['vgg', 'alexnet']:
                num_features = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(num_features, num_classes)
            elif model_name == 'resnet':
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif model_name == 'inception':
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)

        # Lists to store results from each step.
        self.val_preds, self.val_true, self.val_probs = [], [], []
        self.test_preds, self.test_true, self.test_probs = [], [], []

    def forward(self, x):
        """Define the forward pass of the model."""
        if isinstance(self.model, (ViTForImageClassification, models.Inception3)):
            # These models return an object, so we extract the logits.
            output = self.model(x)
            return output.logits if hasattr(output, 'logits') else output
        else:
            return self.model(x)

    def configure_optimizers(self):
        """Set up the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Decrease learning rate every 5 epochs.
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_step(self, batch):
        """Calculate loss and predictions for a batch."""
        x, y = batch
        logits = self(x)
        # Use class weights in the loss function if they are provided.
        loss = F.binary_cross_entropy_with_logits(
            logits, y.float().unsqueeze(-1), pos_weight=self.class_weights
        )
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """Perform a single training step."""
        loss, _, _ = self._calculate_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step."""
        loss, preds, y = self._calculate_step(batch)
        acc = torch.sum(preds.view(-1) == y.long().view(-1)).item() / len(y)
        self.val_preds.extend(preds.detach().cpu().numpy())
        self.val_true.extend(y.detach().cpu().numpy())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Perform a single test step."""
        loss, preds, y = self._calculate_step(batch)
        acc = torch.sum(preds.view(-1) == y.long().view(-1)).item() / len(y)
        self.test_preds.extend(preds.detach().cpu().numpy())
        self.test_true.extend(y.detach().cpu().numpy())
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        """Clear validation results at the start of each validation epoch."""
        self.val_preds.clear()
        self.val_true.clear()

    def on_test_epoch_start(self):
        """Clear test results at the start of the test epoch."""
        self.test_preds.clear()
        self.test_true.clear()
"""
Manages the model training and evaluation process.

The ExperimentRunner class handles setting up the experiment, preparing the
data, training the model using PyTorch Lightning, and saving the results
(metrics, plots, and predictions).
"""
#**************************************************************************
# IMPORTS
#**************************************************************************
import os
import time
import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from CustomDataset import CustomDataset
from GenericClassifier import GenericClassifier

class ExperimentRunner:
    """
    Handles the setup, execution, and logging of a training experiment.
    """
    def __init__(self, args, dataset_info):
        """
        Initialize the ExperimentRunner.

        Args:
            args (Namespace): Arguments from the command line.
            dataset_info (dict): Dictionary with paths for the dataset.
        """
        self.args = args
        self.csv_file = dataset_info.get('csv_file')
        self.data_path_1 = dataset_info.get('data_path_1', None)
        self.data_path_2 = dataset_info.get('data_path_2', None)
        self.results_dir = 'results'
        self.logs_dir = 'logs'
        self.seed = 42
        self.batch_size = 64
        self.num_classes = 1  # Binary classification
        self.device = args.device
        self.df = pd.read_csv(self.csv_file)
        self.aug = False  # Becomes True if any data augmentation is used

    def set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def get_data_transforms(self):
        """
        Create and return the data transformations for training and validation.

        Returns:
            tuple: A tuple containing (train_transforms, validation_transforms).
        """
        train_transforms = []

        # Add data augmentations as specified in the arguments.
        if self.args.horizontal_flip:
            train_transforms.append(transforms.RandomHorizontalFlip())
        if self.args.vertical_flip:
            train_transforms.append(transforms.RandomVerticalFlip())
        if self.args.rotation:
            train_transforms.append(transforms.RandomRotation(self.args.rotation))
        if self.args.color_jitter:
            b, c, s, h = self.args.color_jitter
            train_transforms.append(transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))
        if self.args.crop:
            size = 224 if self.args.model in ["vit", "vitmae"] else 299
            train_transforms.append(transforms.RandomResizedCrop(size, scale=self.args.crop[1:3]))

        # Check if any augmentation was added.
        self.aug = len(train_transforms) > 0

        # Add basic resize and ToTensor if no augmentations are on.
        if not self.aug and self.args.model not in ["vit", "vitmae"]:
            train_transforms.append(transforms.Resize((299, 299)))
        if self.args.model not in ["vit", "vitmae"]:
            train_transforms.append(transforms.ToTensor())

        # Validation transforms are simpler: just resize and convert to tensor according to the model.
        val_size = (224, 224) if self.args.model in ["vit", "vitmae"] else (299, 299)
        val_transforms = [transforms.Resize(val_size)]
        if self.args.model not in ["vit", "vitmae"]:
            val_transforms.append(transforms.ToTensor())

        return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

    def prepare_data(self, transform_train, transform):
        """
        Split the dataset and create Dataset objects for train, val, and test sets.

        Args:
            transform_train (callable): Transformations for the training data.
            transform (callable): Transformations for the validation/test data.

        Returns:
            tuple: A tuple of (train_dataset, val_dataset, test_dataset).
        """
        total_samples = len(self.df)
        indices = list(range(total_samples))

        # Split data: 80% train, 10% validation, 10% test.
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        test_size = total_samples - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            indices, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Create custom dataset objects.
        train_dataset = CustomDataset(train_dataset.indices, self.csv_file, self.data_path_1, self.data_path_2, transform_train, self.args.model)
        val_dataset = CustomDataset(val_dataset.indices, self.csv_file, self.data_path_1, self.data_path_2, transform, self.args.model)
        test_dataset = CustomDataset(test_dataset.indices, self.csv_file, self.data_path_1, self.data_path_2, transform, self.args.model)

        return train_dataset, val_dataset, test_dataset

    def train_model(self, experiment_name, class_weights, train_loader, val_loader, test_loader):
        """
        Configure and run the model training process.

        Args:
            experiment_name (str): A unique name for this experiment run.
            class_weights (torch.Tensor): Weights for handling imbalanced classes.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader): DataLoader for the test set.
        """
        start_time = time.time()

        model = GenericClassifier(model_name=self.args.model, num_classes=self.num_classes, learning_rate=self.args.learning_rate, class_weights=class_weights)

        # Stop training early if validation accuracy doesn't improve.
        early_stop_callback = EarlyStopping(monitor='val_acc', patience=3, mode='max')
        # Save the best version of the model based on validation accuracy.
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')

        # Configure the trainer from PyTorch Lightning.
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            accelerator='auto',
            devices=1,
            logger=TensorBoardLogger(self.logs_dir, name=experiment_name),
            callbacks=[checkpoint_callback, early_stop_callback]
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

        end_time = time.time()
        time_taken = end_time - start_time

        # Save all metrics after training is complete.
        self.save_metrics(
            model.val_true, model.val_preds, model.test_true, model.test_preds,
            model.val_probs, model.test_probs, model.learning_rate,
            experiment_name, time_taken
        )

        print(f"Experiment {experiment_name} completed in {time_taken:.2f} seconds.")

    def save_metrics(self, val_true, val_preds, test_true, test_preds, val_probs, test_probs, learning_rate, experiment_name, time_taken):
        """
        Calculate and save performance metrics and plots.

        Args:
            val_true (list): True labels for the validation set.
            val_preds (list): Predicted labels for the validation set.
            test_true (list): True labels for the test set.
            test_preds (list): Predicted labels for the test set.
            val_probs (list): Predicted probabilities for the validation set.
            test_probs (list): Predicted probabilities for the test set.
            learning_rate (float): The learning rate used for training.
            experiment_name (str): The unique name of the experiment.
            time_taken (float): Total time for the experiment.
        """
        output_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        # Calculate all metrics.
        metrics = {
            "val_accuracy": accuracy_score(val_true, val_preds),
            "test_accuracy": accuracy_score(test_true, test_preds),
            "val_precision": precision_score(val_true, val_preds, average='binary'),
            "test_precision": precision_score(test_true, test_preds, average='binary'),
            "val_recall": recall_score(val_true, val_preds, average='binary'),
            "test_recall": recall_score(test_true, test_preds, average='binary'),
            "val_f1": f1_score(val_true, val_preds, average='binary'),
            "test_f1": f1_score(test_true, test_preds, average='binary'),
            "val_auc": roc_auc_score(val_true, val_probs),
            "test_auc": roc_auc_score(test_true, test_probs),
            "time_taken": time_taken
        }

        # Save experiment arguments to a file.
        arguments = {
            "max_epochs": self.args.max_epochs, "learning_rate": learning_rate,
            "balanced": self.args.balanced, "horizontal_flip": self.args.horizontal_flip,
            "vertical_flip": self.args.vertical_flip, "rotation": self.args.rotation,
            "resize": self.args.crop, "color_jitter": self.args.color_jitter
        }

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(arguments, f, indent=4)

        # Create and save confusion matrix plots.
        val_confusion = confusion_matrix(val_true, val_preds)
        test_confusion = confusion_matrix(test_true, test_preds)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(val_confusion, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Validation Confusion Matrix')
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
        sns.heatmap(test_confusion, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Test Confusion Matrix')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
        plt.close()

        # Save raw predictions and labels to CSV files
        np.savetxt(os.path.join(output_dir, 'val_probs.csv'), np.array(val_probs), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'val_true.csv'), np.array(val_true), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'val_preds.csv'), np.array(val_preds), delimiter= ',')
        np.savetxt(os.path.join(output_dir, 'test_probs.csv'), np.array(test_probs), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'test_true.csv'), np.array(test_true), delimiter=',')
        np.savetxt(os.path.join(output_dir, 'test_preds.csv'), np.array(test_preds), delimiter= ',')

    def run(self):
        """Run the main experiment sequence."""
        self.set_seeds()

        balanced = 'balanced' if self.args.balanced else 'notBalanced'
        transform_train, transform = self.get_data_transforms()
        augmentation = 'aug' if self.aug else 'noAug'
        experiment_name = f'{self.args.dataset}/{self.args.model}/{balanced}_{augmentation}_lr{self.args.learning_rate}_maxEpochs{self.args.max_epochs}'

        train_dataset, val_dataset, test_dataset = self.prepare_data(transform_train, transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        class_weights = None
        if self.args.balanced:
            # Calculate weights to help with imbalanced datasets.
            # Weight = (number of negative samples) / (number of positive samples)
            if "ISIC2020" in self.csv_file:
                class_counts = self.df['target'].value_counts()
            else:
                class_counts = self.df['dx_bin'].value_counts()
            count_0 = class_counts.get(0)
            count_1 = class_counts.get(1)
            class_weights = torch.tensor(count_0/count_1, dtype=torch.float).to(self.device)

        self.train_model(experiment_name, class_weights, train_loader, val_loader, test_loader)
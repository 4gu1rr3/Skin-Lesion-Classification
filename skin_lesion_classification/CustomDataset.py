"""
A custom Dataset for PyTorch to load skin lesion images.

Handles various dataset formats (HAM10000, ISIC, PH2) by reading metadata
from a CSV file and loading the corresponding images from disk.
"""
#**************************************************************************
# IMPORTS
#**************************************************************************
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from transformers import ViTImageProcessor, AutoImageProcessor

#**************************************************************************
# LOADS THE DATASET
#**************************************************************************
class CustomDataset(Dataset):
    """
    Loads skin lesion images and labels from a CSV file.

    Attributes:
        csv_file (str): Path to the CSV file with annotations.
        indices (list): A list of specific rows to use from the CSV file.
        root_dir_1 (str): Path to the main directory containing images.
        root_dir_2 (str): Path to a second directory that might contain images.
        transform (callable): Transformations to apply to the images.
        model (str): The name of the model being used (e.g., "vit").
    """
    def __init__(self, indices, csv_file, root_dir_1, root_dir_2, transform, model):
        """
        Initialize the CustomDataset object.

        Args:
            indices (list): The specific rows of the CSV to load.
            csv_file (str): The path to the CSV annotation file.
            root_dir_1 (str): The first directory where images are stored.
            root_dir_2 (str): The second directory where images are stored.
            transform (callable): The transformations to be applied to images.
            model (str): The name of the model, used to select the right processor.
        """
        self.csv_file = csv_file
        self.indices = indices
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.transform = transform
        self.annotations = pd.read_csv(csv_file).iloc[indices]
        self.model = model

        # Load a special processor if using a Vision Transformer (ViT) model.
        if model == "vit":
            self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        if model == "vitmae":
            self.processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a single sample (image and label) from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the processed image and its label.
                   Returns (None, None) if the image file cannot be found.
        """
        # Get the image ID from the correct column based on the dataset.
        if 'HAM' in self.csv_file:
            img_code = self.annotations.iloc[idx, 1]  # Column 'image_id'
        else:
            img_code = self.annotations.iloc[idx, 0]  # First column by default

        # Create the correct image file name.
        if 'PH2' in self.csv_file:
            img_name = img_code+'/'+img_code+'_Dermoscopic_Image/'+img_code+'.bmp'
        else:
            img_name = img_code + '.jpg'

        img_path = None

        # Check in the first directory.
        if os.path.exists(os.path.join(self.root_dir_1, img_name)):
            img_path = os.path.join(self.root_dir_1, img_name)
        # If not found, check in the second directory.
        elif self.root_dir_2 and os.path.exists(os.path.join(self.root_dir_2, img_name)):
            img_path = os.path.join(self.root_dir_2, img_name)

        # If the image is still not found, print an error.
        if img_path is None:
            print(f"IDX {idx}")
            print(f"File {img_name} not found in any of the specified directories.")
            return None, None

        # Open the image and make sure it's in RGB format.
        image = Image.open(img_path).convert('RGB')

        # Get the label from the correct column based on the dataset.
        if "HAM" in self.csv_file:
            label = torch.tensor(self.annotations.iloc[idx, 8])
        elif "ISIC2019" in self.csv_file:
            label = torch.tensor(self.annotations.iloc[idx, 10])
        elif "ISIC2020" in self.csv_file:
            label = torch.tensor(self.annotations.iloc[idx, 7])
        elif "PH2" in self.csv_file:
            label = torch.tensor(self.annotations.iloc[idx, 5])

        # Apply transformations if they exist.
        if self.transform:
            image = self.transform(image)
            # If it's a ViT model, apply the special processor.
            if self.model in ["vit", "vitmae"]:
                image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        return image, label
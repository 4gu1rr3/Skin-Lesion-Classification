# Skin Lesion Classification using Deep Learning

# Folder containing the dataset class

# Importing all the necessary libraries
import torch
from torch.utils.data import Dataset

from PIL import Image
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir_1, root_dir_2, transform):
        self.annotations = pd.read_csv(csv_file)                         # Path to the CSV file containing data information.
        self.root_dir_1 = root_dir_1                                     # Path to the first directory where images are stored.
        self.root_dir_2 = root_dir_2                                     # Path to the second directory where images are stored.
        self.transform = transform                                       # Transformations to be applied to the images.

    def __len__(self):
        return len(self.annotations)                                     # Return the number of samples in the dataset.

    def __getitem__(self, idx):
        img_code = self.annotations.iloc[idx, 1]                         # Extract the file code from the DataFrame
        img_name = img_code + '.jpg'                                     # Add the '.jpg' extension
        img_path = None

        # Check if image is in directory 1
        if os.path.exists(os.path.join(self.root_dir_1, img_name)):
            img_path = os.path.join(self.root_dir_1, img_name)

        # Check if image is in directory 2
        elif os.path.exists(os.path.join(self.root_dir_2, img_name)):
            img_path = os.path.join(self.root_dir_2, img_name)

        # Print an error message if image is not found in either directory
        if img_path is None:
            print("IDX ",idx )
            print(f"File {img_name} not found in any of the specified directories.")
            return None, None  # Return None for image and label

        # Open the image and convert to RGB if found
        # This operation is included as a precaution to ensure all images are treated consistently
        image = Image.open(img_path).convert('RGB')

        label = self.annotations.iloc[idx, 2]                              # Access the value in row idx and column 2

        if label == 0:
            label = torch.tensor(0)                                        # Convert to a tensor with value 0
        else:
            label = torch.tensor(1)                                        # Convert to a tensor with value 1

        if self.transform:
            image = self.transform(image)                                  # Apply transformations

        return image, label
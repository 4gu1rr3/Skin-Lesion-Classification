# Skin Lesion Classification using Deep Learning

# If you're using Tinder, all the necessary requirements are already installed in a conda environment.
# To activate the environment in the terminal, use the command: ```conda activate env```

# Accessing TensorBoard:

"""
1. Navigate to the TensorBoard logs directory:
    ```cd skin_lesion_classification/logs```

2. Start TensorBoard:
    ```tensorboard --logdir ./ --bind_all```

3. ctrl + click on the TensorBoard link.

Run the cell below if you haven't installed the requirements on your machine yet.
"""

"""
# !pip install -r requirements.txt
"""

# Importing all the necessary python libraries
import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import random

# Imports from files
from models import AlexNetClassifier, EfficientNetClassifier, InceptionClassifier, ResNetClassifier, VGGClassifier, cuda_available, train_model
from dataset import CustomDataset

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if cuda_available:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load CSV file and define paths
csv_file = '/home/ashiley/HAM10000_metadata_alterado.csv'
data_path_1 = '/home/ashiley/HAM10000_images_part_1'
data_path_2 = '/home/ashiley/HAM10000_images_part_2'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# Create custom dataset instance
custom_dataset = CustomDataset(csv_file=csv_file, root_dir_1=data_path_1, root_dir_2=data_path_2, transform=transform)

# Create a DataLoader to load data in batches during training.
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

if cuda_available:
    # Get the number of available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print("CUDA is available and {} CUDA device(s) is(are) available.".format(num_cuda_devices))
else:
    print("CUDA is not available. You are running on CPU.")

# Move a tensor to the GPU if CUDA is available
device = torch.device("cuda" if cuda_available else "cpu")

# Concatenate datasets
full_dataset = custom_dataset

# Ensure reproducibility for the dataset split
generator = torch.Generator().manual_seed(seed)

# Define
input_channels = 3  # Number of channels in the input images (RGB)

# Dicionário para mapear nomes de modelos para classes reais
MODEL_CLASSES = {
    'vgg': VGGClassifier,
    'resnet': ResNetClassifier,
    'alexnet': AlexNetClassifier,
    'efficientnet': EfficientNetClassifier,
    'inception': InceptionClassifier
}

def main():
    # Crie o parser de argumentos
    parser = argparse.ArgumentParser(description='Skin Lesion Classification using Deep Learning')
    
    # Adicione argumentos ao parser
    parser.add_argument('--model', type=str, required=True, choices=['vgg', 'resnet', 'alexnet', 'efficientnet', 'inception'], default='vgg', help='model type to be used. Options=[vgg, resnet, alexnet, efficientnet, inception].')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes.')
    parser.add_argument('--name', type=str, default='TESTING_VGGClassifier_lr1e4_freezing_all_except_last_layer', help='Name of experiment.')
    
    # Parse os argumentos
    args = parser.parse_args()

    # Obtenha a classe do modelo a partir do dicionário
    model_class = MODEL_CLASSES[args.model]
    
    # Treine o modelo com os argumentos fornecidos
    train_model(model_class=model_class, num_classes=args.num_classes, name=args.name, full_dataset=full_dataset, generator=generator)

if __name__ == '__main__':
    main()
# Skin Lesion Classification using Deep Learning

### MALTA Lab
- Escola Politécnica, PUCRS

This project is focused on classifying skin lesions using deep learning techniques. It utilizes various datasets, including ISIC 2019, ISIC 2020, HAM10000, and PH2, to train and evaluate models such as Vision Transformers (ViT) and other pre-trained convolutional neural networks (CNNs).

## Project Structure

The project consists of several scripts for preprocessing datasets, training classifiers, and running experiments:

- **Data Preprocessing Scripts**: Each dataset (ISIC 2019, ISIC 2020, HAM10000, PH2) has a corresponding script for preprocessing. These scripts filter the data, keeping only relevant skin lesion types (Melanoma and Nevus), and prepare the datasets for training.
  - `editISIC2019.py`: Preprocesses ISIC 2019 dataset.
  - `editISIC2020.py`: Preprocesses ISIC 2020 dataset.
  - `editHAM10000.py`: Preprocesses HAM10000 dataset.
  - `editPH2.py`: Preprocesses PH2 dataset.

- **Custom Dataset Class**: `CustomDataset.py` contains a custom PyTorch Dataset class to load images and their labels from the preprocessed CSV files.

- **Model and Training**: The classifier is defined in `GenericClassifier.py`, using PyTorch Lightning to handle the training, validation, and testing phases. It supports multiple pre-trained models like ResNet, VGG, EfficientNet, and Vision Transformers (ViT).

- **Experiment Runner**: `ExperimentRunner.py` is responsible for setting up the training experiments, loading datasets, applying transformations, and managing training and evaluation processes.

- **Configuration**: `datasets_config.json` contains paths to the datasets and their corresponding CSV files for training.

## Requirements

- Python 3.8 or higher
- PyTorch 1.10 or higher
- PyTorch Lightning
- Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Transformers (for Vision Transformers)
- EfficientNet (for pre-trained models)

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Setup

1. **Download Datasets**:  
   You need to download the following datasets and place them in the appropriate folders:
   - [ISIC 2019](https://challenge.isic-archive.com/data/)
   - [ISIC 2020](https://challenge.isic-archive.com/data/)
   - [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - [PH2 Dataset](https://www.fc.up.pt/addi/ph2%20database.html)

   Once downloaded, ensure that the images and CSV files are placed as described in the `datasets_config.json` file. The structure should be as follows:

   ```
   ./
     ├── CustomDataset.py
     ├── datasets_config.json
     ├── ExperimentRunner.py
     ├── GenericClassifier.py
     ├── ISIC_2020_Training_Input/
     ├── ISIC_2020_Training_JPEG/
     ├── PH2Dataset/
     ├── HAM10000_images_part_1/
     └── HAM10000_images_part_2/
   ```

2. **Run Preprocessing Scripts**:  
   After downloading the datasets, run the preprocessing scripts to filter the dataset and generate the required CSV files. For example, run:

   ```bash
   python editISIC2019.py
   python editISIC2020.py
   python editHAM10000.py
   python editPH2.py
   ```

   This will generate CSV files such as `ISIC2019_nv_mel.csv` and `HAM10000_metadata_nv_mel.csv`.
   

## Running the Program

1. **Preprocessing**:  
   Make sure to run the preprocessing scripts to filter the datasets first.

2. **Run the Experiment**:  
   Once the preprocessing is done, you can run the experiments using the `ExperimentRunner`. The following command will execute a training and evaluation session with the specified parameters:

   ```bash
   python script_args.py --dataset ISIC2019 --model vit --learning_rate 0.0001 --max_epochs 10
   ```

   This will train a model using the specified dataset and model type. The results will be logged in the `logs/` directory, and metrics will be saved in the `results/` directory.

   ### Command-Line Arguments Breakdown:

- `--dataset`: Specifies which dataset to use. Possible options are:
  - `ISIC2019`
  - `ISIC2020`
  - `HAM10000`
  - `PH2`

- `--model`: Specifies the model to use. Possible options are:
  - `resnet`
  - `vgg`
  - `alexnet`
  - `efficientnet`
  - `inception`
  - `vit`
  - `vitmae`

- `--learning_rate`: The initial learning rate for the optimizer. Example: `0.0001`

- `--max_epochs`: The maximum number of epochs to train. Example: `10`

- `--balanced`: If set, the dataset will be balanced using class weighting to handle imbalanced data.

- `--horizontal_flip`: If set, horizontal flip augmentation will be applied to the images.

- `--vertical_flip`: If set, vertical flip augmentation will be applied to the images.

- `--rotation`: Degrees of rotation for augmentation. Example: `30`

- `--crop`: Random resized crop parameters in the format `[enabled, min_scale, max_scale]`. Example: `1.0 0.8 1.0`

- `--device`: Specifies the device to run on. Possible values are:
  - `cuda:0`
  - `cuda:1`
  - `cpu`

- `--color_jitter`: Apply color jitter augmentation with the specified values for brightness, contrast, saturation, and hue. Example: `0.2 0.2 0.2 0.2`


## Outputs

- **Metrics**: After training, metrics such as accuracy, precision, recall, F1-score, and AUC will be saved in JSON format.
- **Confusion Matrices**: Plots of confusion matrices for both validation and test datasets will be generated.
- **Predictions**: Raw predictions and labels will be saved in CSV files for further analysis.

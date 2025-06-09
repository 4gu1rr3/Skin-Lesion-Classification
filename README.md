# Skin Lesion Classification using Deep Learning

## MALTA Lab

## Institution:
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
   - [ISIC 2019](https://www.isic-archive.com/)
   - [ISIC 2020](https://www.isic-archive.com/)
   - [HAM10000](https://www.kaggle.com/datasets/ryanchou/ham10000)
   - [PH2 Dataset](https://www.fc.ul.pt/en/courses/msc/Information-Systems-and-Computing/Projects/2017-2018/PH2)

   Once downloaded, ensure that the images and CSV files are placed as described in the `datasets_config.json` file. The structure should be as follows:

   ```
   datasets/
     ├── ISIC_2020_Training_Input/
     ├── ISIC_2020_Training_JPEG/
     ├── PH2Dataset/
     └── HAM10000_images_part_1/
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

3. **Configure the Experiment**:  
   Before running the experiments, configure the dataset and model by specifying the parameters in the command line. Example:

   ```bash
   python script_args.py --dataset ISIC2019 --model vit --learning_rate 0.0001 --max_epochs 10
   ```

## Running the Program

1. **Preprocessing**:  
   Make sure to run the preprocessing scripts to filter the datasets first.

2. **Run the Experiment**:  
   Once the preprocessing is done, you can run the experiments using the `ExperimentRunner`. The following command will execute a training and evaluation session with the specified parameters:

   ```bash
   python script_args.py --dataset ISIC2019 --model vit --learning_rate 0.0001 --max_epochs 10
   ```

   This will train a model using the specified dataset and model type. The results will be logged in the `logs/` directory, and metrics will be saved in the `results/` directory.

## Outputs

- **Metrics**: After training, metrics such as accuracy, precision, recall, F1-score, and AUC will be saved in JSON format.
- **Confusion Matrices**: Plots of confusion matrices for both validation and test datasets will be generated.
- **Predictions**: Raw predictions and labels will be saved in CSV files for further analysis.
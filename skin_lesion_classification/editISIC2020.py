"""
Preprocess the ISIC 2020 ground truth file.

Filters the CSV to keep only Melanoma ('melanoma') and Nevus ('nevus') samples
and prints class counts.
"""
import pandas as pd

# Load the CSV file.
file_path = 'ISIC_2020_Training_GroundTruth.csv'
data = pd.read_csv(file_path)

# Keep only the rows where the 'dx' column is either 'nevus' or 'melanoma'.
data = data[(data['dx'] == 'nevus') | (data['dx'] == 'melanoma')].copy()

#print the counts of each class
print(data['dx'].value_counts())

# Save the filtered data to a new CSV file.
data.to_csv('ISIC2020_nv_mel.csv', index=False)
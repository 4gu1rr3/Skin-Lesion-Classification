"""
Preprocess the ISIC 2019 ground truth file.

Filters the CSV to keep only Melanoma ('MEL') and Nevus ('NV') samples,
adds a new binary label column ('dx_bin'), and prints class counts.
"""
import pandas as pd

# Load the CSV file.
file_path = 'ISIC_2019_Training_GroundTruth.csv'
data = pd.read_csv(file_path)

# Filter rows where either the 'MEL' or 'NV' column is 1.0.
data = data[(data['MEL'] == 1.0) | (data['NV'] == 1.0)].copy()

# Add a new binary label column 'dx_bin' based on the 'MEL' and 'NV' columns.
data['dx_bin'] = data.apply(lambda row: 1 if row['MEL'] == 1.0 else 0, axis=1)

# Print the counts of each class.
print(data['dx_bin'].value_counts())

# Save the filtered data to a new CSV file.
data.to_csv('ISIC2019_nv_mel.csv', index=False)
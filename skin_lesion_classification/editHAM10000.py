"""
Preprocess the HAM10000 metadata file.

Filters the CSV to keep only Melanoma ('mel') and Nevus ('nv') samples
adds a new binary label column ('dx_bin'), and prints class counts.
"""
import pandas as pd

# Load the CSV file.
file_path = 'HAM10000_metadata.txt'
data = pd.read_csv(file_path)

# Keep only the rows where the 'dx' column is either 'nv' or 'mel'.
data = data[(data['dx'] == 'nv') | (data['dx'] == 'mel')].copy()

# Add a new binary label column 'dx_bin' based on the 'dx' column.
data['dx_bin'] = data['dx'].apply(lambda x: 1 if x == 'mel' else 0)

# Print the counts of each class.
print(data['dx'].value_counts())

# Save the filtered data to a new CSV file.
data.to_csv('HAM10000_metadata_nv_mel.csv', index=False)
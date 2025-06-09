"""
Preprocess the PH2 ground truth file.

Modifies the CSV to keep only Melanoma ('MEL') and Nevus ('Common Nevus' 
and 'Atypical Nevus') samples, adds a new binary label column ('dx_bin'), 
and prints class counts.
"""
import pandas as pd

# Load the CSV file.
file_path = 'PH2'
data = pd.read_csv(file_path)

# Add a new binary label column 'dx_bin' based on the 'Commom Nevus', 'Atypical Nevus' and 'Melanoma' columns.
data['dx_bin'] = data.apply(lambda row: 1 if row['Melanoma'] == 'X' else 0, axis=1)

# Print the counts of each class.
print(data['dx_bin'].value_counts())

# Save the filtered data to a new CSV file.
data.to_csv('PH2_nv_mel.csv', index=False)
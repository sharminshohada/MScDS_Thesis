"""
Author: Shohada Sharmin
Date: 2024-06-05
Description: This script processes multiple CSV files by performing the following tasks:
             1. Reads each CSV file with a multi-level header and adjusts the column names.
             2. Increments the first column's values by 1 and converts the 'Label' column to numeric.
             3. Filters the data to include only specific labels (1, 2, 3, or 4) and removes columns ending with '_likelihood'.
             4. Saves the processed data to new CSV files with '_Filtered' appended to the original filenames.
             5. Counts and displays the occurrences of each label in the 'Label' column for each processed file.

Dependencies:
- pandas: Used for data manipulation and analysis.
"""

# Function to process the data from a CSV file. Adjusts column names, filters data, and removes unnecessary columns.
import pandas as pd

def process_data(file_path):
    data = pd.read_csv(file_path, header=[0, 1])
    new_columns = [f"{x}_{y}" if y is not None else x for x, y in data.columns[:-1]] + ['Label']
    data.columns = new_columns
    data[data.columns[0]] = data[data.columns[0]] + 1
    data['Label'] = pd.to_numeric(data['Label'], errors='coerce')
    filtered_data = data.dropna(subset=['Label'])
    filtered_data = filtered_data[filtered_data['Label'].isin([1, 2, 3, 4])]
    # Remove columns that end with '_likelihood'
    filtered_data = filtered_data[[col for col in filtered_data.columns if not col.endswith('_likelihood')]]
    return filtered_data

# List of file paths (adjust these paths to match where your files are located)
file_paths = [
    'D:\\Research\\2024\\2finalprep\\Analysis\\P1_Cheerio_NV.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P1_Cheerio_V.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P2_Cheerio_NV.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P2_Cheerio_V.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P3_Cheerio_NV.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P3_Cheerio_V.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P4_Cheerio_NV.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P4_Cheerio_V.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P5_Cheerio_NV.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P5_Cheerio_V.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P6_Cheerio_NV.csv',
    'D:\\Research\\2024\\2finalprep\\Analysis\\P6_Cheerio_V.csv',
    # Add your other file paths here
]

# Loop through each file, process the data, save the filtered results, and count labels
for file_path in file_paths:
    try:
        filtered_data = process_data(file_path)
        output_file_path = file_path.replace('.csv', '_Filtered.csv')  # Generates a new file name
        filtered_data.to_csv(output_file_path, index=False)
        print(f"Data processed and saved successfully for {file_path}")
        
        # Count and display the number of each label
        label_counts = filtered_data['Label'].value_counts().sort_index()
        print(f"Label counts for {file_path}:")
        print(label_counts)
        
    except Exception as e:
        print(f"Failed to process and save data for {file_path}: {e}")

import os
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np

# Define the paths
csv_file = 'SIREN_test_sets/the_rock/the_rock_metadata.csv'
destination_dir = 'SIREN_test_sets/the_rock/the_rock_ALL_SPECTRA'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Read the CSV file containing the file names
file_metadata = pd.read_csv(csv_file)

# Assuming the file names are in a column named 'File Name'
file_names = file_metadata['File Name'].tolist()

# Get the current working directory
search_dir = os.getcwd()

# Define the possible file endings
file_endings = [
    ".dat", ".txt", ".csv", ".fits", ".asc", ".spec", ".flm", 
    ".dat.gz", ".txt.gz", ".csv.gz", ".fits.gz", ".asc.gz", 
    ".spec.gz", ".flm.gz"
]

# List to keep track of moved files
moved_files = []

# Loop through the file names and move them if found
for file_name in tqdm(file_names, desc="Processing Files", ncols=100):
    found = False
    original_file_name = file_name
    
    # Try all file endings
    for ending in file_endings:
        if found:
            break
        search_file_name = original_file_name + ending
        
        # Use os.walk to recursively search for the file in the current directory
        for root, dirs, files in os.walk(search_dir):
            if search_file_name in files:
                # Construct full file path
                source_path = os.path.join(root, search_file_name)
                # Move the file to the destination directory
                shutil.move(source_path, os.path.join(destination_dir, search_file_name))
                print(f'Moved {search_file_name} to {destination_dir}')
                moved_files.append(original_file_name)
                found = True
                break  # Stop searching once the file is found

    if not found:
        # If the file was not found, print a message
        print(f'File {original_file_name} not found with any of the expected extensions.')

# List all files in the destination directory (base names only)
files_in_destination = [os.path.splitext(f)[0] for f in os.listdir(destination_dir)]

# Check if all files from the metadata have been moved
missing_files = [file for file in file_names if file not in files_in_destination]

# Output results
if missing_files:
    print(f'\n{len(missing_files)} files are missing and were not moved:')
    for missing_file in missing_files:
        print(missing_file)
else:
    print(f'\nAll files from the metadata have been successfully moved.')

# Print a summary
print(f'\nSummary:')
print(f'Files in metadata: {len(file_names)}')
print(f'Files found and moved: {len(moved_files)}')
print(f'Files in destination directory: {len(files_in_destination)}')

# Check the metadata file for any duplicate names
duplicate_files = file_metadata[file_metadata.duplicated(subset=['File Name'], keep=False)]

if not duplicate_files.empty:
    print(f'\n{len(duplicate_files)} duplicate file names found in the metadata:')
    print(duplicate_files)
else:
    print('\nNo duplicate file names found in the metadata.')

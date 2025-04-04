import os
import shutil
import pandas as pd

# Load the metadata
metadata_path = r'C:\Users\gabv1\Downloads\karen_imagenes\metadata.csv'
metadata_df = pd.read_csv(metadata_path)

# Define a function to classify lesions as cancerous or not
def classify_lesion(diagnostic):
    if diagnostic in ['BCC', 'MEL', 'SCC']:
        return True
    else:
        return False

# Apply the function to the 'diagnostic' column to create a new column 'cancer_status'
metadata_df['cancer_status'] = metadata_df['diagnostic'].apply(classify_lesion)

# Paths to the image directories
image_directory = r'C:\Users\gabv1\Downloads\karen_imagenes\images\todas'
cancer_directory = r'C:\Users\gabv1\Downloads\karen_imagenes\cancer_images'
not_cancer_directory = r'C:\Users\gabv1\Downloads\karen_imagenes\not_cancer_images'

# Create the directories if they don't exist
os.makedirs(cancer_directory, exist_ok=True)
os.makedirs(not_cancer_directory, exist_ok=True)

# Function to move images based on cancer status
def move_images(row):
    img_id = row['img_id']
    cancer_status = row['cancer_status']
    src_path = os.path.join(image_directory, img_id)
    
    if cancer_status:
        dst_path = os.path.join(cancer_directory, img_id)
    else:
        dst_path = os.path.join(not_cancer_directory, img_id)
    
    # Check if the source file exists
    if os.path.exists(src_path):
        print(f"Moving {src_path} to {dst_path}")
        shutil.move(src_path, dst_path)
    else:
        print(f"File not found: {src_path}")

# Apply the function to each row in the dataframe
metadata_df.apply(move_images, axis=1)

print("Images have been moved successfully.")
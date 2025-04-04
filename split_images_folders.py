# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:07:49 2024

@author: gabv1
"""

import os
import random
import shutil

def split_images(source_folder, dest_folder_20, dest_folder_80, split_ratio=0.2):
    # Ensure destination folders exist
    os.makedirs(dest_folder_20, exist_ok=True)
    os.makedirs(dest_folder_80, exist_ok=True)
    
    # Get list of all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Shuffle and split the list of files
    random.shuffle(all_files)
    split_point = int(len(all_files) * split_ratio)
    
    # Split the files into two groups
    files_20 = all_files[:split_point]
    files_80 = all_files[split_point:]
    
    # Move files to their respective folders
    for f in files_20:
        shutil.move(os.path.join(source_folder, f), os.path.join(dest_folder_20, f))
        
    for f in files_80:
        shutil.move(os.path.join(source_folder, f), os.path.join(dest_folder_80, f))

if __name__ == "__main__":
    source_folder = r"C:\Users\gabv1\Downloads\karen_imagenes\not_cancer_images"
    dest_folder_20 = r"C:\Users\gabv1\Downloads\karen_imagenes\not_cancer_images_20"
    dest_folder_80 = r"C:\Users\gabv1\Downloads\karen_imagenes\not_cancer_images_80"
    
    split_images(source_folder, dest_folder_20, dest_folder_80)
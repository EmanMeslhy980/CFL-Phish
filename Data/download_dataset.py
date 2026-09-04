"""
Phish360 Dataset Downloader and Extractor
-----------------------------------------
This script downloads the official Phish360 dataset and extracts it 
using the provided password. It is designed to run in a Google Colab 
environment and save the data directly to Google Drive.
"""

import os
import requests
from tqdm import tqdm
import zipfile

# 1. Setup paths and password
# Modify this path if you wish to save the dataset in a different Google Drive directory
drive_path = "/content/drive/MyDrive/Phish360_Dataset"
zip_filename = "phish360.zip"
zip_filepath = os.path.join(drive_path, zip_filename)
extract_path = os.path.join(drive_path, "phish360_extracted")

# The password must be encoded to bytes for the zipfile library
password = "phish360_trt".encode('utf-8')

# Official download URL
url = "https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/data/phish360.zip"

# 2. Ensure the target directory exists and create it if necessary
os.makedirs(drive_path, exist_ok=True)

# 3. Download the dataset (with a progress bar)
if not os.path.exists(zip_filepath):
    print(f"⬇️ Starting download from: {url}")
    print("⚠️ This may take a few minutes depending on your internet speed (~5 GB)...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status() # Ensure the connection was successful
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_filepath, 'wb') as file, tqdm(
        desc=zip_filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
            size = file.write(data)
            progress_bar.update(size)
            
    print("✅ Download completed successfully!")
else:
    print("⚠️ Zip file already exists. Skipping download...")

# 4. Extract the dataset using the password
if not os.path.exists(extract_path):
    print(f"📂 Extracting files to: {extract_path}")
    print("⚠️ Extraction may take a few minutes...")
    
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path, pwd=password)
        print("✅ Extraction completed successfully!")
    except RuntimeError as e:
        if "Bad password" in str(e):
            print("❌ Error: Incorrect password!")
        else:
            print(f"❌ Extraction error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
else:
    print("✅ Extracted folder already exists. Skipping extraction...")

print(f"\n🎉 Process finished! You can find your data at: {extract_path}")

import os
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import pandas as pd
from tqdm import tqdm  # Import tqdm
from io import StringIO

# Base URL of the 2D directory
BASE_URL = "https://files.docking.org/2D/"

# List of pages to download (for example, AA to KK)
pages = [chr(i) + chr(j) for i in range(65, 75) for j in range(65, 75)]  # 'AA' to 'KK'

# Shared list to store the downloaded SMILES data
smiles_list = pd.DataFrame(columns=['smiles', 'zinc_id'])

# Lock to ensure thread-safe access to smiles_list
smiles_lock = threading.Lock()

output_file = 'data/smiles.csv'

# Function to download a single SMILES file
def download_smi_file(page, file_name):
    try:
        url = f"{BASE_URL}{page}/{file_name}"
        response = requests.get(url)
        if response.status_code == 200:        
            content_str = response.content.decode('utf-8')
            # Use StringIO to simulate a file object
            content_io = StringIO(content_str)
            # Read the content into a DataFrame (with whitespace separation)
            data = pd.read_csv(content_io, sep=' ', header=0, names=['smiles', 'zinc_id'])
            # Append the SMILES file to the shared list
            with smiles_lock:  # Acquire the lock to safely append to the list
                pd.concat((smiles_list, data), axis=0, ignore_index=True)
            print(f"Downloaded {file_name} from {url}")
    except Exception as e:
        print(f"Failed to download {file_name} from {url}: {e}")
# Function to download all SMILES files in a page
def download_page_files(page):
    try:
        response = requests.get(f"{BASE_URL}{page}/")
        response.raise_for_status()  # Check if the page is available
        file_list = response.text.split('\n')  # Extract file names from the page (adjust based on the HTML structure)
        #print(file_list)
        smi_files = [file for file in file_list if '.smi' in file]  # Filter only SMILES files
        smile_files = []
        for content in smi_files:
            idx = content.index('.smi')
            file_name = content[idx-4:idx+4]
            smile_files.append(file_name)
        # Use tqdm to display a progress bar for downloading each file
        with ThreadPoolExecutor(max_workers=10) as executor:
            for smile_file in tqdm(smile_files, desc=f"Downloading {page}", unit="file", leave=True):
                executor.submit(download_smi_file, page, smile_file)
    except requests.RequestException as e:
        print(f"Error downloading files for page {page}: {e}")

# Start downloading for all pages
if __name__ == "__main__":
    # Use tqdm for progress on pages download
    for page in tqdm(pages, desc="Downloading Pages", unit="page", leave=True):
        download_page_files(page)
    
    # After all downloads, print the total number of files downloaded
    print(f"Total SMILES files downloaded: {len(smiles_list)}")

    smiles_list = smiles_list.drop(columns=['zinc_id'])

    pd.to_csv(smiles_list, index=False)

    


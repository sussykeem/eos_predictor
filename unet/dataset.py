import deepchem as dc
import pandas as pd
import requests
import time

# Function to convert ZINC ID to SMILES via ZINC15 API
def zinc_id_to_smiles(zinc_id):
    url = f"https://zinc15.docking.org/substances/{zinc_id}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("smiles")
        else:
            print(f"Failed to get SMILES for {zinc_id}")
            return None
    except Exception as e:
        print(f"Error with {zinc_id}: {e}")
        return None

# Step 1: Load the ZINC15 dataset
zinc_tasks, zinc_datasets, transformers = dc.molnet.load_zinc15()
train_dataset, valid_dataset, test_dataset = zinc_datasets

# Step 2: Extract ZINC IDs (not SMILES) from each dataset
train_ids = train_dataset.ids
valid_ids = valid_dataset.ids
test_ids = test_dataset.ids

# Step 3: Convert ZINC IDs to SMILES
def convert_ids_to_smiles(zinc_ids, sleep_time=0.2):
    smiles = []
    for zid in zinc_ids:
        try:
            s = zinc_id_to_smiles(zid)
            if s:
                smiles.append(s)
            else:
                print(f"Failed to convert {zid} to SMILES")
                smiles.append("FAILED")
            time.sleep(sleep_time)  # To avoid overwhelming the server
        except Exception as e:
            print(f"Error converting {zid}: {e}")
            smiles.append("ERROR")
    return smiles

print("Converting Train ZINC IDs to SMILES...")
train_smiles = convert_ids_to_smiles(train_ids)

print("Converting Validation ZINC IDs to SMILES...")
valid_smiles = convert_ids_to_smiles(valid_ids)

print("Converting Test ZINC IDs to SMILES...")
test_smiles = convert_ids_to_smiles(test_ids)

# Step 4: Save to CSV
pd.DataFrame({'SMILES': train_smiles}).to_csv('zinc_train_smiles.csv', index=False)
pd.DataFrame({'SMILES': valid_smiles}).to_csv('zinc_valid_smiles.csv', index=False)
pd.DataFrame({'SMILES': test_smiles}).to_csv('zinc_test_smiles.csv', index=False)

print("SMILES strings saved to CSVs successfully!")

# Step 5: Verification
for split, path in [("Train", "zinc_train_smiles.csv"), 
                    ("Validation", "zinc_valid_smiles.csv"), 
                    ("Test", "zinc_test_smiles.csv")]:
    df = pd.read_csv(path)
    print(f"First 5 {split} SMILES: {df['SMILES'].head().tolist()}")
    print(f"Size of {split} SMILES: {len(df)}")

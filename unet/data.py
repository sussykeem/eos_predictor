import deepchem as dc
import smilite as sm
from tqdm import tqdm
import pandas as pd
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Load ZINC15 data
tasks, datasets, _ = dc.molnet.load_zinc15()
train_dataset, valid_dataset, test_dataset = datasets

train_zinc_list, valid_zinc_list, test_zinc_list = train_dataset.ids, valid_dataset.ids, test_dataset.ids

def get_smile(zinc_id):
    """Helper function to get single SMILES string"""
    try:
        smile = sm.get_zinc_smile(zinc_id, 'zinc15')
        return zinc_id, smile, None
    except Exception as e:
        return zinc_id, None, str(e)

def parallel_get_smiles(zinc_list, output_file, num_workers=32):
    """Parallel processing of SMILES retrieval"""
    smile_data = []
    failed_data = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_id = {executor.submit(get_smile, zinc_id): zinc_id for zinc_id in zinc_list}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_id), total=len(zinc_list), desc="Processing"):
            zinc_id = future_to_id[future]
            try:
                zinc_id, smile, error = future.result()
                if smile:
                    smile_data.append(smile)
                elif error:
                    failed_data.append((zinc_id, error))
            except Exception as e:
                failed_data.append((zinc_id, str(e)))
    
    # Save results
    if smile_data:
        pd.DataFrame(smile_data, columns=['smiles']).to_csv(output_file, index=False)
    
    if failed_data:
        pd.DataFrame(failed_data, columns=['zinc_id', 'error']).to_csv(
            output_file.replace('.csv', '_failed.csv'), index=False)
    
    return len(smile_data), len(failed_data)

if __name__ == '__main__':
    # Set number of workers based on your system (typically 2-4x CPU cores)
    NUM_WORKERS = min(32, (os.cpu_count() or 1) * 4)
    
    print(f"Using {NUM_WORKERS} workers for parallel processing")
    
    print("Processing training set...")
    train_success, train_failed = parallel_get_smiles(train_zinc_list, 'train_smiles.csv', NUM_WORKERS)
    
    print("Processing validation set...")
    valid_success, valid_failed = parallel_get_smiles(valid_zinc_list, 'valid_smiles.csv', NUM_WORKERS)
    
    print("Processing test set...")
    test_success, test_failed = parallel_get_smiles(test_zinc_list, 'test_smiles.csv', NUM_WORKERS)
    
    print("\nProcessing complete!")
    print(f"Training set: {train_success} succeeded, {train_failed} failed")
    print(f"Validation set: {valid_success} succeeded, {valid_failed} failed")
    print(f"Test set: {test_success} succeeded, {test_failed} failed")
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem

urls = ['https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/test_data.csv',
                'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/train_data.csv']
train = pd.read_csv(urls[0])
test = pd.read_csv(urls[1])

data = pd.concat((train, test), axis=0, ignore_index=True)

X_cols = ['smile']
y_cols = ['a', 'b']

X = data[X_cols].copy()
y = data[y_cols]

        # Ensure the first row is not a header issue
X = X.reset_index(drop=True)

element_counts = {}

for smiles in X['smile']:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            # Count the atom itself
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
            # Count implicit hydrogens
            num_h = atom.GetTotalNumHs()
            if num_h > 0:
                element_counts['H'] = element_counts.get('H', 0) + num_h

# Get atomic weights
ptable = Chem.GetPeriodicTable()
element_weights = {el: ptable.GetAtomicWeight(el) for el in element_counts}

# Sort by atomic weight
sorted_elements = sorted(element_counts.keys(), key=lambda el: element_weights[el])
sorted_counts = [element_counts[el] for el in sorted_elements]

# Plot
plt.figure(figsize=(8, 6))
plt.bar(sorted_elements, sorted_counts, color='orchid', edgecolor='black')
plt.title('Element Frequency in Van der Waals Dataset')
plt.xlabel('Element (sorted by weight)')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
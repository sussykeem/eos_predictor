from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger

# data cols
# ['sci_name','name','cid','smile','Molecular Weight','LogP','TPSA','Rotatable Bonds','H Bond Donors','H Bond Acceptors','Aromatic Rings','Num Rings','Atom Count','coulomb_matrix','embeddings']

class MoleculeVisualizer():

    def visualize_molecule_2D(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        RDLogger.DisableLog('rdApp.*')
        # mol = Chem.AddHs(mol) # add implicit hydrogens
        if mol is None:
            print("Invalid SMILES string.")
            return

        img = Draw.MolToImage(mol, size=(224, 224))

        return img

class EOS_Dataset(Dataset):

    def __init__(self, train=True):
        
        self.smiles = self.load_data(train)

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))], p=0.3),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform = transform_train if train else transform_test

    def load_data(self, train):
        urls = ['https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/test_data.csv',
                'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/train_data.csv']

        data = pd.read_csv(urls[train])

        X_cols = ['smile']

        X = data[X_cols].copy()

        # Ensure the first row is not a header issue
        X = X.reset_index(drop=True)

        return X['smile']


    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        X = self.smiles[idx]
        img = MoleculeVisualizer().visualize_molecule_2D(X)
        img = self.transform(img)

        label = img # for training on reconstructed images

        return img, label

    def transform(self, img):
        return self.transform(img)            

class EOS_Dataloader():

    def __init__(self):

        train = EOS_Dataset(True)
        test = EOS_Dataset(False)

        self.train_loader = DataLoader(train, batch_size=32, shuffle=True)
        self.train_smiles = train.smiles
        self.test_loader = DataLoader(test, batch_size=32, shuffle=True)
        self.test_smiles = test.smiles
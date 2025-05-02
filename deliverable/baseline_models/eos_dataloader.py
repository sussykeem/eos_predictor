from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger
import random

# data cols
# ['sci_name','name','cid','smile','Molecular Weight','LogP','TPSA','Rotatable Bonds','H Bond Donors','H Bond Acceptors','Aromatic Rings','Num Rings','Atom Count','coulomb_matrix','embeddings']

class MoleculeVisualizer():

    def visualize_molecule_2D(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        RDLogger.DisableLog('rdApp.*')
        if mol is None:
            print("Invalid SMILES string.")
            return

        img = Draw.MolToImage(mol, size=(256, 256))

        return img

class EOS_Dataset(Dataset):

    def __init__(self, train=True, mode='reconstruct', scaler=None, num=None):
        self.mode = mode
        self.train = train
        self.scaler = scaler
        self.visualizer = MoleculeVisualizer()
        self.smiles, self.targets = self.load_data(train, mode, num)

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Lambda(lambda img: F.invert(img)),
            #transforms.Lambda(lambda img: self.safe_invert(img)),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
            #transforms.RandomApply([AddRandomNoise(noise_range=0.25)], p=1),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Lambda(lambda img: self.safe_invert(img)),
            #transforms.Lambda(lambda img: F.invert(img)),
            #transforms.RandomApply([AddRandomNoise(noise_range=0.25)], p=1),
        ])

        self.transform_fn = transform_train if train else transform_test

    def load_data(self, train, mode, num=None):

        if mode == 'reconstruct':
            urls = [
                'dataset/test_smiles.csv',
                'dataset/train_smiles.csv',
            ]
        elif mode == 'predict':
            urls = [
                'eos_dataset/test_data.csv',
                'eos_dataset/train_data.csv'
            ]

        df = pd.read_csv(urls[train])
        df = df.reset_index(drop=True)

        if num is not None:
            smiles = df['smile'][:num]
            if not train:
                smiles = df['smile'][:int(num/5)]
        else:
            smiles = df['smile']

        if mode == 'reconstruct':
            targets = smiles  # We'll generate the same image again
        elif mode == 'predict':
            targets = df[['a', 'b']].values.astype(np.float32)
            if self.scaler is not None:
                if train:
                    targets = self.scaler.fit_transform(targets)
                else:
                    targets = self.scaler.transform(targets)

        return smiles, targets

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        img = self.visualizer.visualize_molecule_2D(smile)
        img = self.transform_fn(img)

        if self.mode == 'reconstruct':
            label = img  # autoencoder-style
        elif self.mode == 'predict':
            label = torch.tensor(self.targets[idx])  # a, b values

        return img, label
    
    def safe_invert(self, img):

        summed_img = torch.mean(img, axis=0)

        mask = (summed_img == 1)
        mask = mask.unsqueeze(0).repeat(3, 1, 1)
        img[mask] = 0.5
        
        mask = (summed_img < .5)
        mask = mask.unsqueeze(0).repeat(3, 1, 1)
        img[mask] = 1

        return img

class AddRandomNoise:
    def __init__(self, noise_range=0.05):
        self.noise_range = noise_range  # Max magnitude of noise

    def __call__(self, img):
        # img: Tensor [C, H, W] with values in [0, 1]
        noise = torch.empty_like(img).uniform_(-self.noise_range, self.noise_range)
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0.0, 1.0)


class EOS_Dataloader():


    def __init__(self, batch_size=32, mode='reconstruct', num=None):

        assert mode in ['reconstruct', 'predict'], "Mode must be 'reconstruct' or 'predict'"

        self.mode = mode

        self.batch_size = batch_size
        
        self.scaler = StandardScaler()

        train = EOS_Dataset(train=True, mode=self.mode, scaler=self.scaler, num=num)
        test = EOS_Dataset(train=False, mode=self.mode, scaler=self.scaler, num=num)

        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.train_smiles = train.smiles
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
        self.test_smiles = test.smiles
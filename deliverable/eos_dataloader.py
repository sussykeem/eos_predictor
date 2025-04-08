from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# data cols
# ['sci_name','name','cid','smile','Molecular Weight','LogP','TPSA','Rotatable Bonds','H Bond Donors','H Bond Acceptors','Aromatic Rings','Num Rings','Atom Count','coulomb_matrix','embeddings']

class MoleculeVisualizer():

    def visualize_molecule_2D(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) # add implicit hydrogens
        if mol is None:
            print("Invalid SMILES string.")
            return

        img = Draw.MolToImage(mol, size=(300, 300))

        return img

class EOS_Dataset(Dataset):

    def __init__(self, scale=True, train=True):
        
        self.imgs, self.y, self.smiles = self.load_data(train)
        self.y = self.y.values.astype(np.float32)

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

        # if k_augs:
        #     assert k_augs > 0
        #     self.imgs, self.y = self.data_aug(k_augs)

        # Separate scalers for a and b
        self.scaler_a = StandardScaler()
        self.scaler_b = StandardScaler()

        # Fit scalers on the respective columns
        if scale:
          self.y[:, 0] = self.scaler_a.fit_transform(self.y[:, 0].reshape(-1, 1)).reshape(-1)
          self.y[:, 1] = self.scaler_b.fit_transform(self.y[:, 1].reshape(-1, 1)).reshape(-1)

    def load_data(self, train):
        urls = ['https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/test_data.csv',
                'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/train_data.csv']
        url = urls[int(train)]

        imgs = []

        data = pd.read_csv(url)

        X_cols = ['smile']
        y_cols = ['a', 'b']

        X = data[X_cols].copy()
        y = data[y_cols]

        # Ensure the first row is not a header issue
        X = X.reset_index(drop=True)

        for smile in X['smile']:
            img = MoleculeVisualizer().visualize_molecule_2D(smile)
            imgs.append(np.array(img, dtype=np.float32))
        imgs = np.array(imgs, dtype=np.uint8)

        return imgs, y, X['smile']


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.y[idx]

        img = self.transform(img)

        return img, label

    def transform(self, img):
        return self.transform(img)
    
    def inverse_transform(self, labels):
        labels_unscaled = np.zeros_like(labels)
        labels_unscaled[:, 0] = self.scaler_a.inverse_transform(labels[:,0].reshape(-1,1)).reshape(-1)
        labels_unscaled[:, 1] = self.scaler_b.inverse_transform(labels[:,1].reshape(-1,1)).reshape(-1)
        return labels_unscaled
    
    # def data_aug(self, k_augs):

    #     aug_imgs = np.repeat(self.imgs, k_augs, axis=0)
    #     factor_y = np.repeat(self.y, k_augs, axis=0)

    #     transform = transforms.Compose([
            
    #     ])

    #     for i, img in enumerate(aug_imgs):
    #         img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    #         if i % k_augs == 0:
    #             c_img = img
    #         else:
    #             c_img = transform(img)
    #         c_img = c_img.permute(1, 2, 0).numpy()
    #         aug_imgs[i] = c_img
    #     return aug_imgs, factor_y               

class EOS_Dataloader():

    def __init__(self, train, test):
        self.train_loader = DataLoader(train, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=32, shuffle=False)
        self.train_smiles = train.smiles
        self.test_smiles = test.smiles
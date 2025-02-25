from rdkit import Chem
from rdkit.Chem import Draw

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class MoleculeVisualizer():

    def visualize_molecule_2D(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES string.")
            return

        img = Draw.MolToImage(mol, size=(300, 300))

        return img

class EOS_Dataset(Dataset):

    def __init__(self, imgs, y, scale=True):
        self.imgs = imgs
        self.y = y.values.astype(np.float32)
        self.transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Separate scalers for a and b
        self.scaler_a = StandardScaler()
        self.scaler_b = StandardScaler()

        # Fit scalers on the respective columns
        if scale:
          self.y[:, 0] = self.scaler_a.fit_transform(self.y[:, 0].reshape(-1, 1)).reshape(-1)
          self.y[:, 1] = self.scaler_b.fit_transform(self.y[:, 1].reshape(-1, 1)).reshape(-1)


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




class EOS_Dataloader():

    def __init__(self):

        train_url = 'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/train_data.csv'
        test_url = 'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/test_data.csv'

        train_imgs = []
        test_imgs = []

        train_data = pd.read_csv(train_url)
        test_data = pd.read_csv(test_url)

        X_cols = ['sci_name','name','cid','smile','Molecular Weight','LogP','TPSA','Rotatable Bonds','H Bond Donors','H Bond Acceptors','Aromatic Rings','Num Rings','Atom Count','coulomb_matrix','embeddings']
        y_cols = ['a', 'b']

        X_train = train_data[X_cols]
        y_train = train_data[y_cols]
        X_test = test_data[X_cols]
        y_test = test_data[y_cols]

        for smile in X_train['smile']:
            img = MoleculeVisualizer().visualize_molecule_2D(smile)
            train_imgs.append(np.array(img, dtype=np.float32))

        for smile in X_test['smile']:
            img = MoleculeVisualizer().visualize_molecule_2D(smile)
            test_imgs.append(np.array(img, dtype=np.float32))

        train_imgs = np.array(train_imgs, dtype=np.uint8)
        test_imgs = np.array(test_imgs, dtype=np.uint8)

        train_data = EOS_Dataset(train_imgs, y_train)
        test_data = EOS_Dataset(test_data, y_test)
        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
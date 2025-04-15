import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class EOS_Features_Dataset(Dataset):
    def __init__(self, train=True, in_scale=None, t_scale=None):
        super().__init__()

        urls = [
            'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/test_data.csv',
            'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/train_data.csv'
        ]
    
        # Use 1 for train, 0 for test
        df = pd.read_csv(urls[train])
        df = df.reset_index(drop=True)
        inputs = df[['Molecular Weight','LogP','TPSA','Rotatable Bonds','H Bond Donors','H Bond Acceptors','Aromatic Rings','Num Rings','Atom Count']]
        targets = df[['a', 'b']].values.astype(np.float32)
        
        if in_scale is not None and t_scale is not None:
            self.inputs = in_scale.fit_transform(inputs) if train else in_scale.transform(inputs)
            self.targets = t_scale.fit_transform(targets) if train else t_scale.transform(targets)
        else:
            self.inputs = inputs.values.astype(np.float32)
            self.targets = targets
            
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class EOS_Features_Dataloader():
    def __init__(self, batch_size=32, scale=True):
        super().__init__()


        if scale:
            self.in_scaler = StandardScaler()
            self.t_scaler = StandardScaler()
        else:
            self.in_scaler = None
            self.t_scaler = None

        train_data = EOS_Features_Dataset(train=True, in_scale=self.in_scaler, t_scale=self.t_scaler)
        test_data = EOS_Features_Dataset(train=False, in_scale=self.in_scaler, t_scale=self.t_scaler)

        self.train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(test_data, batch_size=batch_size, shuffle=True)


features_data = EOS_Features_Dataloader()
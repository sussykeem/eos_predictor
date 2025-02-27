import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from cnn import CNN
from eos_dataloader import EOS_Dataloader, EOS_Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = EOS_Dataset(scale=True, train=True)
test_data = EOS_Dataset(scale=True, train=False)

eos_dataloader = EOS_Dataloader(train_data, test_data)

class PKAN_Data(Dataset):

    def __init__(self, data_loader, train=True):
        self.data_loader = data_loader
        self.smiles = data_loader.train_smiles if train else data_loader.test_smiles
        self.encoder = self.load_encoder()
        data = self.generate_encoding_set(train)
        del self.encoder # free up GPU resources
        self.X = [copy.copy(x) for x in data['X']]
        self.y = [copy.copy(y) for y in data['y']]

        # Compute molecular weights for all SMILES in dataset 
        self.molecular_weights = self.compute_molecular_weights()

    def compute_molecular_weights(self):
        """Computes molecular weights for all SMILES strings in dataset."""
        mol_weights = []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw = Descriptors.MolWt(mol)  # Compute molecular weight
                mol_weights.append(mw)
            else:
                print(f"Invalid SMILES: {smi}")  # Debugging output
                mol_weights.append(0.0)  # Assign a default valid value (e.g., 0)
        return torch.tensor(mol_weights, dtype=torch.float32, device=device)

    
    def load_encoder(self, path='cnn_model.pth', input_dim=300):
        model = CNN(self.data_loader, input_dim)
        # strict=False allows us to use the model with the last layer dropped
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
        model.to(device)
        model.eval()
        return model

    def generate_encoding_set(self, train=True):
        data = {'X': [], 'y': []}

        loader = self.data_loader.train_loader if train else self.data_loader.test_loader
        encodings = []
        labels = []
        for img, label in loader:
            img = img.to(device)
            encodings.append(self.encoder(img))
            labels.append(label)

        for i, e in enumerate(encodings):
            for j in range(e.shape[0]):
                data['X'].append(e[j])
                data['y'].append(labels[i][j])
        return data

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.molecular_weights[idx]


class KANLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_kernels=10):

        super(KANLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_kernels = num_kernels

        self.weights = nn.Parameter(torch.randn(self.output_dim, self.num_kernels))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

        self.centers = nn.Parameter(torch.linspace(-1, 1, self.num_kernels))
        self.widths = nn.Parameter(torch.ones(self.num_kernels) * 0.1)

    def forward(self, x):

        kernels = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) / (2 * self.widths ** 2))
        activation = torch.sum(torch.matmul(kernels, self.weights.T), dim=-1)  + self.bias

        return activation

class PKAN(nn.Module):

    def __init__(self, kan_layer, train_data, test_data, feature_vec_size=64):
        super(PKAN, self).__init__()

        self.train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

        self.KAN_layer = kan_layer

        self.fc_pipeline = nn.Sequential(
            nn.Linear(feature_vec_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout after activation
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            self.KAN_layer,
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Slightly lower dropout for later layers
            nn.Linear(32, 2)
        )       

    def forward(self, x):
        x = self.fc_pipeline(x)
        return x
    
    def loss(self, pred, label, mol_weights, factor=0.1): # change when implementing physics loss
        criterion = nn.SmoothL1Loss()
        base_loss = criterion(pred, label)
        phys_loss = self.physics_loss(pred, mol_weights)
        return base_loss + factor * phys_loss
    
    def physics_loss(self, pred, mol_weights):
        phys_loss = 0  # Initialize physics loss

        # 1. Positivity Constraint (Ensure a and b are non-negative)
        phys_loss += torch.mean(F.relu(-pred[:, 0]))  # Ensure a >= 0
        phys_loss += torch.mean(F.relu(-pred[:, 1]))  # Ensure b >= 0

        # 2. Approximate Physical Relationship
        c = 0.1  # Empirical constant
        phys_loss += torch.mean(F.relu(pred[:, 1] - c * (torch.abs(pred[:, 0]) ** (1/3))))  # Avoid negative roots

        # 3. Enforce Monotonicity of b with Molecular Weight
        sorted_indices = mol_weights.argsort()
        sorted_b = pred[:, 1][sorted_indices]
        sorted_molecular_weights = mol_weights[sorted_indices]

        diffs = sorted_b[1:] - sorted_b[:-1]  # b_{i+1} - b_i
        mw_diffs = sorted_molecular_weights[1:] - sorted_molecular_weights[:-1]  # M_{i+1} - M_i

        valid_mask = mw_diffs > 1e-6  # Avoid division by zero or indexing errors
        if valid_mask.any():  # Check if valid values exist
            phys_loss += torch.mean(F.relu(-diffs[valid_mask]))

        return phys_loss
    
    def train_pkan(self, epochs=100, learning_rate=1e-4, patience=10, min_delta=0.001):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        self.to(device)  # Move model to GPU if available
        loss_history = []
        val_history = []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_val = 0.0

            for inputs, labels, mol_weights in self.train_loader:
                inputs, labels, mol_weights = inputs.to(device), labels.to(device), mol_weights.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.loss(outputs, labels, mol_weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Clip gradients
                optimizer.step()

                running_loss += loss.item()
                loss_history.append(loss.item())
                
                val_loss = self.train_val()
                running_val += val_loss.item()
                val_history.append(val_loss.item())

            avg_val_loss = running_val / len(self.test_loader)
            scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(self.train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"\tValidation Loss: {running_val / len(self.test_loader):.4f}")

               # Early Stopping Logic
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_weights = self.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        # Restore best model before returning
        if best_model_weights:
            self.load_state_dict(best_model_weights)
            print("Restored best model weights.")

        plt.figure()
        plt.plot(loss_history)
        plt.plot(val_history, c='r')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('PKAN Training Loss')
        plt.show()

        print("PKAN Training Complete!")

        return loss_history, val_history
    
    def train_val(self):
        self.eval()
        with torch.no_grad():
            imgs, labels, mol_weights = next(iter(self.test_loader))
            imgs, labels, mol_weights = imgs.to(device), labels.to(device), mol_weights.to(device)

            outputs = self(imgs)
            loss = self.loss(outputs, labels, mol_weights)
        self.train()
        return loss
    
    def validate_pkan(self, unscale_loader):
        self.eval()  # Set model to evaluation mode
        total_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():  # Disable gradient computation
            for data in self.test_loader:
                inputs, labels, mol_weights = data
                inputs, labels, mol_weights = inputs.to(device), labels.to(device), mol_weights.to(device)

                outputs = self(inputs)  # Forward pass through PKAN
                loss = self.loss(outputs, labels, mol_weights)
                total_loss += loss.item()

                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        print(f"Validation Loss: {avg_loss:.4f}")

        # Convert lists to numpy arrays for further evaluation
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Unscale predictions and labels 

        # fix this, dataloader issues, scalers are not shared between dataset and dataloader
        unscaled_outputs = unscale_loader.inverse_transform(torch.tensor(all_outputs))
        unscaled_labels = unscale_loader.inverse_transform(torch.tensor(all_labels))

        # Compute evaluation metrics
        mae = np.mean(np.abs(unscaled_outputs - unscaled_labels), axis=0)
        mse = np.mean((unscaled_outputs - unscaled_labels) ** 2, axis=0)

        print(f"Mean Absolute Error (MAE): a={mae[0]:.4f}, b={mae[1]:.4f}")
        print(f"Mean Squared Error (MSE): a={mse[0]:.4f}, b={mse[1]:.4f}")

        return avg_loss, mae, mse
    
    def save_model(self, file_path="pkan_model.pth"):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

# implement physics loss, replace L1.
# figure out if we use trend loss
# consider other physical constraints
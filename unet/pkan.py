import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset, DataLoader

#maybe delete these later
from unet_2 import Unet
from eos_dataloader import EOS_Dataloader
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PKAN_Data(Dataset):

    def __init__(self, data_loader, encoder, train=True):
        self.data_loader = data_loader
        self.smiles = data_loader.train_smiles if train else data_loader.test_smiles
        self.encoder = encoder
        data = self.generate_encoding_set(train)
        # del self.encoder # free up GPU resources
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

# KAN Layer class
class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_kernels):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_kernels = num_kernels

        # Learnable parameters for KAN layer
        self.weights = nn.Parameter(torch.randn(self.output_dim, self.num_kernels))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.centers = nn.Parameter(torch.linspace(-1, 1, self.num_kernels))
        self.widths = nn.Parameter(torch.ones(self.num_kernels) * 0.1)

    def forward(self, x):
        # Gaussian kernel function (RBF)
        kernels = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) / (2 * self.widths ** 2))
        activation = torch.sum(torch.matmul(kernels, self.weights.T), dim=-1) + self.bias
        return activation

# PKAN class
class PKAN(nn.Module):
    def __init__(self, config, dataloader, encoder):
        super(PKAN, self).__init__()

        self.config = config

        train_data = PKAN_Data(dataloader, encoder)
        test_data = PKAN_Data(dataloader, encoder, False)
        self.train_loader = EOS_Dataloader(train_data, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

        self.activations = {
            'ReLU': nn.ReLU(),
            "Sigmoid": nn.Sigmoid(),
        }
        
        # Fully connected layers (including KAN layer)
        self.fc_pipeline = nn.Sequential()
        prev_units = None  # Initialize prev_units for fully connected layers

        for i, layer_config in enumerate(config['layers']):
            if layer_config['type'] == 'KAN':
                # Initialize KAN Layer and add it to the pipeline
                kan_layer = KANLayer(
                    input_dim=prev_units if prev_units else config['network']['input_dim'],
                    output_dim=layer_config['output'],
                    num_kernels=layer_config['num_kernels']
                )
                self.fc_pipeline.add_module(f"kan_{i+1}", kan_layer)
                prev_units = layer_config['output']  # Output of KAN layer
                if 'activation' in layer_config:
                    self.fc_pipeline.add_module(f"relu_{i+1}", self.activations[layer_config['activation']])
                if 'dropout' in layer_config:
                    self.fc_pipeline.add_module(f"dropout_{i+1}", nn.Dropout(layer_config['dropout']))
            else:
                # Add Fully Connected layers
                self.fc_pipeline.add_module(f"fc_{i+1}", nn.Linear(prev_units if prev_units else config['network']['input_dim'], layer_config['units']))
                if 'activation' in layer_config:
                    self.fc_pipeline.add_module(f"relu_{i+1}", self.activations[layer_config['activation']])
                if 'dropout' in layer_config:
                    self.fc_pipeline.add_module(f"dropout_{i+1}", nn.Dropout(layer_config['dropout']))
                prev_units = layer_config['units']
        
        # Output layer
        self.fc_pipeline.add_module("output", nn.Linear(prev_units, 2))  # Assuming binary classification (2 output classes)

        self._initialize_weights()

        self.train_pkan()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
    
    def train_pkan(self, epochs=100, learning_rate=1e-4, patience=10, min_delta=0.001, accuracy_threshold=0.05):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        self.to(device)  # Move model to GPU if available
        loss_history = []
        val_history = []
        accuracy_history = []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_val = 0.0
            running_accuracy = 0.0
            num_batches = 0

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

                # Calculate accuracy based on the threshold
                accuracy = self.calculate_accuracy(outputs, labels, threshold=accuracy_threshold)
                running_accuracy += accuracy
                num_batches += 1
                accuracy_history.append(accuracy)

                # Validation loss
                val_loss = self.train_val()
                running_val += val_loss.item()
                val_history.append(val_loss.item())

            avg_loss = running_loss / num_batches
            avg_accuracy = running_accuracy / num_batches
            avg_val_loss = running_val / len(self.train_loader)

            scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"\tValidation Loss: {avg_val_loss:.4f}")

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

        # Plotting loss and accuracy
        plt.figure(figsize=(12, 6))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, label="Training Loss")
        plt.plot(val_history, c='r', label="Validation Loss")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('PKAN Training Loss')
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_history, label="Training Accuracy")
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('PKAN Training Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print("PKAN Training Complete!")

        return loss_history, val_history, accuracy_history

    def calculate_accuracy(self, outputs, labels, threshold=0.5):
        """
        Calculate the accuracy by checking how many predictions are within the threshold.
        Args:
            outputs (Tensor): The predicted values.
            labels (Tensor): The true labels.
            threshold (float): The error threshold for considering a prediction correct.
        Returns:
            accuracy (float): The percentage of correct predictions.
        """
        # Calculate absolute errors for both 'a' and 'b'
        abs_error_a = torch.abs(outputs[:, 0] - labels[:, 0])
        abs_error_b = torch.abs(outputs[:, 1] - labels[:, 1])

        # Calculate accuracy: predictions are considered accurate if the error is below the threshold
        accuracy_a = (abs_error_a < threshold).float()
        accuracy_b = (abs_error_b < threshold).float()

        # Average accuracy across both 'a' and 'b'
        accuracy = (accuracy_a + accuracy_b) / 2
        return accuracy.mean().item()  # Return as a scalar value
    
    def train_val(self):
        self.eval()
        with torch.no_grad():
            imgs, labels, mol_weights = next(iter(self.test_loader))
            imgs, labels, mol_weights = imgs.to(device), labels.to(device), mol_weights.to(device)

            outputs = self(imgs)
            loss = self.loss(outputs, labels, mol_weights)
        self.train()
        return loss
    
    def validate_pkan(self, unscale_dataset):
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
        unscaled_outputs = unscale_dataset.inverse_transform(torch.tensor(all_outputs))
        unscaled_labels = unscale_dataset.inverse_transform(torch.tensor(all_labels))

        # Compute evaluation metrics
        mae = np.mean(np.abs(unscaled_outputs - unscaled_labels), axis=0)
        mse = np.mean((unscaled_outputs - unscaled_labels) ** 2, axis=0)

        print(f"Mean Absolute Error (MAE): a={mae[0]:.4f}, b={mae[1]:.4f}")
        print(f"Mean Squared Error (MSE): a={mse[0]:.4f}, b={mse[1]:.4f}")

        return avg_loss, mae, mse
    
    def save_model(self, file_path="pkan_model.pth"):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

DataLoader = EOS_Dataloader()
encoder = Unet(DataLoader)

# Load the pre-trained weights from the file
model_path = './unet2_model.pth'  # Adjust the path if needed
encoder.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

# Move the model to the correct device (GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder.to(device)

# Freeze the U-Net encoder's weights to prevent training
for param in encoder.parameters():
    param.requires_grad = False

# Set the model to evaluation mode
encoder.eval()

print("U-Net model loaded successfully!")

# Load the YAML config
with open('base_pkan.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

pkan = PKAN(config, DataLoader, encoder)
pkan.train_pkan()
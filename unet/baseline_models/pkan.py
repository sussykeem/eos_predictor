import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import product
import copy
import matplotlib.pyplot as plt

from eos_features import EOS_Features_Dataloader

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

class PKAN(nn.Module):
    def __init__(self, data = None, num_kernels=5, dropout=[0.5, 0.3, 0.2], input_size=8, output_size=2):
        super(PKAN, self).__init__()

        self.data = data

        layers = [input_size, 64, 32, 16 ,output_size]

        self.kan_layer = KANLayer(layers[2], layers[2], num_kernels=num_kernels)

        self.mlp = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.Dropout(dropout[0]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.Dropout(dropout[1]),
            nn.ReLU(),
            self.kan_layer,
            nn.Linear(layers[2], layers[3]),
            nn.Dropout(dropout[2]),
            nn.ReLU(),
            nn.Linear(layers[3], layers[4]),
        )
        
    def forward(self, x):
        return self.mlp(x)
    
    def train_model(self, epochs=100, learning_rate=0.01, phys_factor=0.1, betas=(0.9, 0.999), weight_decay=0.0, patience=10, min_delta=0.001,):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], betas=self.config['betas'], weight_decay=self.config['weight_decay'])
        self.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in self.data.train:
                inputs = inputs.float()
                targets = targets.float()

                mol_weights = inputs[:, 0]
                x = inputs[:, 1:]

                outputs = self(x)
                loss = self.loss(outputs, targets, mol_weights, factor=phys_factor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_val_loss = self.evaluate()

            # Early Stopping Logic
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_weights = self.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(self.data.train):.4f}')

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        # Restore best model before returning
        if best_model_weights:
            self.load_state_dict(best_model_weights)
            print("Restored best model weights.")



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
    
    def evaluate(self):
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in self.data.test:
                inputs = inputs.float()
                targets = targets.float()
                # Split off molecular weights
                molecular_weights = inputs[:, 0]                  # shape: (batch_size,)
                inputs = inputs[:, 1:]                            # shape: (batch_size, 8)

                # Forward pass
                outputs = self(inputs)                            # shape: (batch_size, 2)
                loss = self.loss(outputs, targets, molecular_weights)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.data.test)
        print(f'Evaluation Loss: {avg_loss:.4f}')
        return avg_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
                x = x[:, 1:]
            return self(x).numpy()
        
    def save_model(self, file_path="base_model_weights/pkan.pth"):
        """ Save the model state dictionary to a file """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def test_model(self):

        self.eval()
            
        with torch.no_grad():

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

            for inputs, targets in self.data.test:
                inputs = inputs.float()
                targets = targets.float()
                inputs = inputs[:, 1:]                            # shape: (batch_size, 8)
                # Forward pass
                preds = self(inputs)
                outputs = self.data.t_scaler.inverse_transform(preds)
                targets = self.data.t_scaler.inverse_transform(targets)

                a_pred = outputs[:,0]
                b_pred = outputs[:,1]

                ax[0].scatter(targets[:,0], a_pred, color='r', alpha=.5)
                ax[0].set_title('A predicted vs A actual')
                ax[0].set_ylabel('A pred')
                ax[0].set_xlabel('A actual')
                ax[1].scatter(targets[:,1], b_pred, color='b', alpha=.5)
                ax[1].set_title('B predicted vs B actual')
                ax[1].set_ylabel('B pred')
                ax[1].set_xlabel('B actual')
            
            fig.suptitle('PKAN Test Prediction Results')
            plt.tight_layout()
            plt.show()

def run_experiment(config, eos_data):
    model = PKAN(eos_data, config['num_kernels'], config['dropout'])
    model.train_model(
        epochs=config['epochs'],
        learning_rate=config['lr'],
        phys_factor=config['phys_factor'],
        betas=config['betas'],
        weight_decay=config['weight_decay']
    )
    loss = model.evaluate()
    return model, loss

def grid_search(param_grid, data):
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in product(*values)]

    best_loss = float('inf')
    best_model = None
    best_config = None

    for config in configs:
        print(f"\n🔧 Running config: {config}")
        model, loss = run_experiment(config, data)
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)
            best_config = config

    print(f"\n✅ Best Config: {best_config}, Eval Loss: {best_loss:.4f}")
    return best_model, best_config



# features_data = EOS_Features_Dataloader()

# param_grid = {
#     'lr': [0.001, 0.0005],
#     'epochs': [100, 200],
#     'phys_factor': [0.05, 0.1],
#     'num_kernels': [5,10],
#     'betas': [(0.9, 0.999), (0.95, 0.999)],
#     'weight_decay': [0.0, 1e-5],
#     'dropout': [[0.5,0.3,0.2], [.4,.2,.2], [.2,.1,0]]
# }

# best_model, best_config = grid_search(param_grid, features_data)

# test_model(best_model, features_data)

# best_model.save_model()
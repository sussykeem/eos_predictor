import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import product
import copy
import matplotlib.pyplot as plt

#from baseline_models.eos_features import EOS_Features_Dataloader

class PINN(nn.Module):
    def __init__(self, data=None, dropout=[0.5, 0.3, 0.2], input_size=8, output_size=2):
        super(PINN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = data

        layers = [input_size, 64, 32, 16 ,output_size]

        self.mlp = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.Dropout(dropout[0]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.Dropout(dropout[1]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3]),
            nn.Dropout(dropout[2]),
            nn.ReLU(),
            nn.Linear(layers[3], layers[4]),
        )

        self.to(self.device)
        
    def forward(self, x):
        return self.mlp(x)
    
    def train_model(self, epochs=100, learning_rate=0.01, phys_factor=0.1, betas=(0.9, 0.999), weight_decay=0.0, patience=10, min_delta=0.001,):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        self.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in self.data.train:
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)

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
        base_loss = self.weighted_loss(pred, label)
        phys_loss = self.physics_loss(pred, mol_weights)
        return base_loss + factor * phys_loss
    
    def weighted_loss(self, outputs, labels):
        criterion = nn.SmoothL1Loss()
        loss = criterion(outputs, labels)
        weight = torch.tensor([1.0, 80], device=self.device)  # Adjust if needed
        return (loss * weight).mean()
    
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
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
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
                x = torch.from_numpy(x).float().to(self.device)
                x = x[:, 1:]
            return self.data.t_scaler.inverse_transform(self(x).cpu().numpy())
        
    def save_model(self, file_path="base_model_weights/pinn.pth"):
        """ Save the model state dictionary to a file """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def test_model(self):

        self.eval()
            
        with torch.no_grad():

            #fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

            a_data = np.empty((2, 0))
            b_data = np.empty((2, 0))

            for inputs, targets in self.data.test:
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                # Forward pass
                x = inputs[:, 1:]
                preds = self(x)
                outputs = self.data.t_scaler.inverse_transform(preds.cpu().numpy())
                targets = self.data.t_scaler.inverse_transform(targets.cpu().numpy())

                a_pred = outputs[:,0]
                b_pred = outputs[:,1]

                a_stack = np.stack([targets[:,0], a_pred])
                b_stack = np.stack([targets[:,1], b_pred])

                a_data = np.concatenate([a_data, a_stack], axis=1)
                b_data = np.concatenate([b_data, b_stack], axis=1)

            return a_data, b_data

def run_experiment(config, eos_data):
    model = PINN(eos_data, config['dropout'])
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
#     'betas': [(0.9, 0.999), (0.95, 0.999)],
#     'weight_decay': [0.0, 1e-5],
#     'dropout': [[0.5,0.3,0.2], [.4,.2,.2], [.2,.1,0]]
# }

# best_model, best_config = grid_search(param_grid, features_data)

# best_model.test_model()

# best_model.save_model()
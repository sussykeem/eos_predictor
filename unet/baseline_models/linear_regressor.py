import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import product
import copy
import matplotlib.pyplot as plt

from eos_features import EOS_Features_Dataloader

class LinearRegressor(nn.Module):
    def __init__(self, data=None, input_size=9, output_size=2):
        super(LinearRegressor, self).__init__()

        self.data = data

        layers = [input_size,output_size]

        self.mlp = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
        )
        
    def forward(self, x):
        return self.mlp(x)
    
    def train_model(self, epochs=100, learning_rate=0.01, betas=(0.9, 0.999), weight_decay=0.0, patience=10, min_delta=0.001,):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        self.train()
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in self.data.train:
                inputs = inputs.float()
                targets = targets.float()

                outputs = self(inputs)
                loss = self.loss(outputs, targets)

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



    def loss(self, pred, label): # change when implementing physics loss
        criterion = nn.SmoothL1Loss()
        base_loss = criterion(pred, label)
        return base_loss
    
    def evaluate(self):
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in self.data.test:
                inputs = inputs.float()
                targets = targets.float()                     # shape: (batch_size, 8)

                # Forward pass
                outputs = self(inputs)                            # shape: (batch_size, 2)
                loss = self.loss(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.data.test)
        print(f'Evaluation Loss: {avg_loss:.4f}')
        return avg_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            return self(x).numpy()
        
    def save_model(self, file_path="base_model_weights/linear.pth"):
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
            
            fig.suptitle('Linear Regressor Test Prediction Results')
            plt.tight_layout()
            plt.show()

def run_experiment(config, eos_data):
    model = LinearRegressor(eos_data)
    model.train_model(
        epochs=config['epochs'],
        learning_rate=config['lr'],
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
#     'betas': [(0.9, 0.999), (0.95, 0.999)],
#     'weight_decay': [0.0, 1e-5],
# }

# best_model, best_config = grid_search(param_grid, features_data)

# test_model(best_model, features_data)

# best_model.save_model()
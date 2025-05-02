import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
from scipy import stats
import seaborn as sns

from eos_dataloader import EOS_Dataloader
from unet import Unet, Encoder

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_kernels):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_kernels = num_kernels

        # KAN parameters
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim, num_kernels))  # per-input-dim kernels
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.centers = nn.Parameter(torch.linspace(-1, 1, num_kernels))  # [num_kernels]
        self.widths = nn.Parameter(torch.ones(num_kernels) * 0.1)        # [num_kernels]

    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.size(0)

        # Compute RBF kernels per input feature: [batch, input_dim, num_kernels]
        x_expanded = x.unsqueeze(-1)  # [batch, input_dim, 1]
        centers = self.centers.view(1, 1, -1)  # [1, 1, num_kernels]
        widths = self.widths.view(1, 1, -1)    # [1, 1, num_kernels]
        kernels = torch.exp(-((x_expanded - centers) ** 2) / (2 * widths ** 2))  # [batch, input_dim, num_kernels]

        # Weighted sum of kernels: [batch, output_dim]
        # weights: [output_dim, input_dim, num_kernels]
        # kernel features: [batch, input_dim, num_kernels]
        weighted = (kernels.unsqueeze(1) * self.weights.unsqueeze(0)).sum(dim=(2, 3))  # [batch, output_dim]

        return weighted + self.bias  # [batch, output_dim]


class Predictor(nn.Module):

    def __init__(self, config, dataloader=None):
        super(Predictor, self).__init__()
        self.config = self.load_config(config)
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = self.load_encoder()

        self.decoder = self.build_decoder()

    def forward(self, x):

        enc = self.encoder(x)
        enc = F.normalize(enc, p=2, dim=1)
        # # Convert encoding to numpy for visualization
        # encoding_numpy = enc.detach().cpu().numpy()
        # # Visualize it
        # plt.figure(figsize=(10, 8))
        # plt.imshow(encoding_numpy)  # Example for 2D encoding visualization
        # plt.colorbar()
        # plt.title("Encoding Visualization")
        # plt.show()
        pred = self.decoder(enc)

        return pred

    def load_config(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_encoder(self):
        unet_data = EOS_Dataloader(mode='reconstruct')
        full_unet = Unet(unet_data)
        full_unet.load_state_dict(torch.load('unet2_sa_model.pth', weights_only=True))
        full_unet.eval()

        encoder = Encoder(full_unet)
        encoder.to(self.device)
        encoder.eval()

        # Freeze only shallow layers — e.g., first two downsampling blocks
        for name, param in encoder.named_parameters():
            if any(layer in name for layer in ["channel1", "encoder1"]):
                param.requires_grad = False
            else:
                param.requires_grad = True  # Fine-tune deeper layers
            

        return encoder

    def build_decoder(self):
        activations = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'LeakyReLU': nn.LeakyReLU(),
            # Add more if needed
        }
        
        fc_pipeline = nn.Sequential()
        prev_units = self.config['network']['input_dim']  # Start with the input dim

        for i, layer_config in enumerate(self.config['layers']):
            layer_type = layer_config['type']

            if layer_type == 'KAN':
                # Placeholder for when you implement KANLayer
                kan_layer = KANLayer(
                    input_dim=prev_units,
                    output_dim=layer_config['output'],
                    num_kernels=layer_config.get('num_kernels', 4)
                )
                fc_pipeline.add_module(f"kan_{i+1}", kan_layer)
                prev_units = layer_config['output']

                if 'activation' in layer_config:
                    act = activations.get(layer_config['activation'], nn.ReLU())
                    fc_pipeline.add_module(f"act_kan_{i+1}", act)

                if 'dropout' in layer_config:
                    fc_pipeline.add_module(f"dropout_kan_{i+1}", nn.Dropout(layer_config['dropout']))

            elif layer_type == 'FC':
                out_units = layer_config['units']
                fc_pipeline.add_module(f"fc_{i+1}", nn.Linear(prev_units, out_units))
                prev_units = out_units

                if 'activation' in layer_config:
                    act = activations.get(layer_config['activation'], nn.ReLU())
                    fc_pipeline.add_module(f"act_fc_{i+1}", act)

                if 'dropout' in layer_config:
                    fc_pipeline.add_module(f"dropout_fc_{i+1}", nn.Dropout(layer_config['dropout']))

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Final output layer
        fc_pipeline.add_module("output", nn.Linear(prev_units, 2))  # 2 outputs for binary classification

        self.initialize_weights(fc_pipeline)
        return fc_pipeline.to(self.device)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        # Using a small constant to push towards positive values and avoid zero predictions
        phys_loss += torch.mean(F.relu(-pred[:, 0] + 1e-6))  # Ensure a >= 0
        phys_loss += torch.mean(F.relu(-pred[:, 1] + 1e-6))  # Ensure b >= 0

        # 2. Approximate Physical Relationship between a and b
        c = 0.1  # Empirical constant
        # Applying a minimum threshold to avoid extreme values of a
        min_a = 1e-3  # Minimum value for a to prevent small values
        phys_loss += torch.mean(F.relu(pred[:, 1] - c * (torch.abs(pred[:, 0]).clamp(min=min_a)) ** (1/3)))  # Physical relationship

        # 3. Enforce Monotonicity of b with Molecular Weight (b should increase with molecular weight)
        # Sort the b values and molecular weights
        sorted_indices = mol_weights.argsort()
        sorted_b = pred[:, 1][sorted_indices]
        sorted_molecular_weights = mol_weights[sorted_indices]

        # Calculate differences between consecutive b values and molecular weights
        diffs = sorted_b[1:] - sorted_b[:-1]  # b_{i+1} - b_i
        mw_diffs = sorted_molecular_weights[1:] - sorted_molecular_weights[:-1]  # M_{i+1} - M_i

        # Valid mask to avoid division by zero or indexing errors
        valid_mask = mw_diffs > 1e-6
        if valid_mask.any():  # Check if there are valid differences
            # Penalize negative differences between b values to ensure monotonicity
            phys_loss += torch.mean(F.relu(torch.abs(diffs[valid_mask])))  # Penalize negative b differences

        return phys_loss
    
    def predict(self, img):
        scaler = self.dataloader.train_loader.dataset.scaler
        with torch.no_grad():
            pred = self(img)
            pred = pred.cpu().numpy()
            return scaler.inverse_transform(pred)

    def train_predictor(self, epochs=100, patience=20, min_delta=0.001):
        if self.config['sgd']:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),lr=self.config['learning_rate'], momentum=self.config.get('momentum', 0.9), weight_decay=self.config.get('weight_decay', 0.0), nesterov=self.config.get('nesterov', False))  
        else:
             optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),  lr=self.config['learning_rate'], betas=self.config['betas'], weight_decay=self.config['weight_decay'])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):

            self.train()
            running_loss = 0.0
            num_batches = 0

            for images, labels, weights in self.dataloader.train_loader:

                images, labels, weights= images.to(self.device), labels.to(self.device), weights.to(self.device)
                optimizer.zero_grad()

                predictions = self(images)
                loss = self.loss(predictions, labels, weights, factor=self.config['factor'])
                #loss = self.loss(predictions, labels, weights)
                loss.backward()
                if self.config['grad_clip']: torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()), max_norm=1.0)  # Clip gradients
                optimizer.step()
                running_loss += loss.item()
                loss_history.append(loss.item())

                num_batches += 1

            avg_loss = running_loss / num_batches

            avg_val_loss = self.eval_predictor()

            scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_loss:.4f} | Test Loss: {avg_val_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_weights = self.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break
        
        if best_model_weights:
            self.load_state_dict(best_model_weights)
            print("Restored best model weights.")

        plt.figure()
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Predictor Training Loss')
        plt.legend()
        plt.show()

    def eval_predictor(self):
        self.eval()

        with torch.no_grad():
            total_loss = 0.0
            num_batches = len(self.dataloader.test_loader)
            for imgs, labels, weights in self.dataloader.test_loader:
                imgs, labels, weights = imgs.to(self.device), labels.to(self.device), weights.to(self.device)
                predictions = self(imgs)
                loss = self.loss(predictions, labels, weights, factor=self.config['factor'])
                total_loss += loss.item()
        return total_loss / num_batches
    
    def test_model(self):

        self.eval()
            
        with torch.no_grad():

            a_data = np.empty((2, 0))
            b_data = np.empty((2, 0))

            for inputs, targets, _ in self.dataloader.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Forward pass
                preds = self(inputs)
                outputs = self.dataloader.scaler.inverse_transform(preds.cpu().numpy())
                targets = self.dataloader.scaler.inverse_transform(targets.cpu().numpy())

                a_pred = outputs[:,0]
                b_pred = outputs[:,1]

                a_stack = np.stack([targets[:,0], a_pred])
                b_stack = np.stack([targets[:,1], b_pred])

                a_data = np.concatenate([a_data, a_stack], axis=1)
                b_data = np.concatenate([b_data, b_stack], axis=1)

            return a_data, b_data

    def save_model(self, file_path="predictor.pth"):
        """ Save the model state dictionary to a file """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

def random_search_tuning(dataloader, config_file, num_trials=10):
    """
    Performs random search hyperparameter tuning for the given predictor class.
    Randomly samples hyperparameters and trains the model for each combination.
    """
    best_params = None
    best_val_loss = float('inf')

    # Define hyperparameter search space
    learning_rates = [1e-3, 1e-4, 1e-5]
    num_kernels = [4, 8, 16]
    dropouts = [0.2, 0.3, 0.5]
    factors = [0.5, 1.0, 1.5, 5, 10]
    weight_decays = [0.0, 1e-4, 1e-5]
    betas = [[0.9, 0.999], [0.95, 0.999], [0.9, 0.98]]
    momentum = [0.9, 0.99]
    bools = [True, False]

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}...")
        
        # Sample hyperparameters randomly
        lr = random.choice(learning_rates)
        num_kernel = random.choice(num_kernels)
        dropout = random.choice(dropouts)
        factor = random.choice(factors)

        # Modify the config file for this trial (assuming config is a yaml file)
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Update config with sampled hyperparameters
        config['learning_rate'] = lr
        config['layers'][0]['num_kernels'] = num_kernel  # Update KAN layer
        config['layers'][0]['dropout'] = dropout  # Update dropout for the first layer
        config['factor'] = factor
        config['weight_decay'] = random.choice(weight_decays)
        config['betas'] = random.choice(betas)
        config['sgd'] = random.choice(bools)
        config['grad_clip'] = random.choice(bools)
        config['nesterov'] = random.choice(bools)
        config['momentum'] = random.choice(momentum)

        # You can add logic to modify other layers similarly if needed
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Create a new model instance with the updated config
        model = Predictor(config=config_file, dataloader=dataloader)

        # Train the model
        model.train_predictor()

        # Evaluate on validation set
        val_loss = model.eval_predictor()

        print(f"Validation Loss for trial {trial + 1}: {val_loss:.4f}")

        # Save the best hyperparameters and model weights based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {
                "learning_rate": lr,
                "num_kernels": num_kernel,
                "dropout": dropout,
                "factor": factor,
                "weight_decay": config['weight_decay'],
                "betas": config['betas'],
                "sgd": config['sgd'],
                "grad_clip": config['grad_clip'], 
                "nesterov": config['nesterov'],
                "momentum": config['momentum'],
            }
            best_model_weights = model.state_dict()

    print(f"Best Hyperparameters: {best_params}")
    return best_params, best_model_weights

# # Example usage:

def plot_distribution(preds, title, model_name):

        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        median_pred = np.median(preds)
        mode_pred = stats.mode(preds, keepdims=True)[0][0]
        lower_ci = np.percentile(preds, 2.5)
        upper_ci = np.percentile(preds, 97.5)
        dist = [mean_pred, std_pred, median_pred, mode_pred, lower_ci, upper_ci]

        plt.figure(figsize=(8, 5))
        sns.histplot(preds, bins=50, kde=True, color="blue", alpha=0.6, label="Prediction Distribution")

        # Mark mean, standard deviation, and confidence interval
        plt.axvline(mean_pred, color='red', linestyle='--', label=f"Mean: {mean_pred:.4f}")
        plt.axvline(median_pred, color='purple', linestyle='-.', label=f"Median: {median_pred:.4f}")
        plt.axvline(mode_pred, color='brown', linestyle=':', label=f"Mode: {mode_pred:.4f}")
        plt.axvline(lower_ci, color='green', linestyle='--', label=f"95% CI Lower: {lower_ci:.4f}")
        plt.axvline(upper_ci, color='green', linestyle='--', label=f"95% CI Upper: {upper_ci:.4f}")

        # Highlight ±1 std deviation
        plt.axvline(mean_pred - std_pred, color='orange', linestyle='-.', label=f"-1 Std Dev: {mean_pred - std_pred:.4f}")
        plt.axvline(mean_pred + std_pred, color='orange', linestyle='-.', label=f"+1 Std Dev: {mean_pred + std_pred:.4f}")

        plt.xlabel("Predicted Value")
        plt.ylabel("Frequency")
        plt.title(f"Prediction Distribution - {model_name} - {title}")
        plt.legend()
        plt.savefig(f'data/{model_name}_{title}.png')
        plt.close()
        return dist
    
# data = EOS_Dataloader(mode='predict')
# # # predictor = Predictor(config='base_decoder.yaml', dataloader=data)

# # #predictor.train_predictor()

# # best_params, best_weights = random_search_tuning(data, 'base_decoder.yaml', num_trials=100)

# # Load the best model weights
# predictor = Predictor(config='base_decoder.yaml', dataloader=data)
# #predictor.load_state_dict(best_weights)
# predictor.train_predictor()
# predictor.eval()


# x, y, _= next(iter(data.train_loader))
# # x, y = x[0], y[0]
# x_t, y_t, _ = next(iter(data.test_loader))
# # x_t, y_t = x_t[0], y_t[0]
# x = x.to(predictor.device)
# x_t = x_t.to(predictor.device)

# x_p, y_p = predictor.predict(x, y)
# x_t_p, y_t_p = predictor.predict(x_t, y_t)

# x_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in x_p.tolist()]
# y_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in y_p.tolist()]
# x_t_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in x_t_p.tolist()]
# y_t_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in y_t_p.tolist()]


# print(f'train predictions: {x_p[0]}\ntrain actual {y_p[0]}')
# print(f'test predictions: {x_t_p[0]}\ntest actual {y_t_p[0]}')

# predictor.test_model()
# predictor.save_model()

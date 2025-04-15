import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import yaml

from eos_dataloader import EOS_Dataloader
from unet import Unet, Encoder

class Predictor(nn.Module):

    def __init__(self, config, dataloader):
        super(Predictor, self).__init__()
        self.config = self.load_config(config)
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = self.load_encoder()

        self.decoder = self.build_decoder()

    def forward(self, x):

        with torch.no_grad():
            enc = self.encoder(x)

        pred = self.decoder(enc)

        return pred

    def load_config(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_encoder(self):
        unet_data = EOS_Dataloader('reconstruct')
        full_unet = Unet(unet_data)
        full_unet.load_state_dict(torch.load('unet_model.pth', weights_only=True))
        full_unet.eval()

        encoder = Encoder(full_unet)
        encoder.to(self.device)
        encoder.eval()

        for param in encoder.parameters():
            param.requires_grad = False

        return encoder

    def build_decoder(self):

        activations = {
            'ReLU': nn.ReLU(),
            "Sigmoid": nn.Sigmoid(),
        }
        fc_pipeline = nn.Sequential()
        prev_units = None  # Initialize prev_units for fully connected layers

        for i, layer_config in enumerate(self.config['layers']):
            if layer_config['type'] == 'KAN':
                pass
                # # Initialize KAN Layer and add it to the pipeline
                # kan_layer = KANLayer(
                #     input_dim=prev_units if prev_units else self.config['network']['input_dim'],
                #     output_dim=layer_config['output'],
                #     num_kernels=layer_config['num_kernels']
                # )
                # fc_pipeline.add_module(f"kan_{i+1}", kan_layer)
                # prev_units = layer_config['output']  # Output of KAN layer
                # if 'activation' in layer_config:
                #     fc_pipeline.add_module(f"relu_{i+1}", activations[layer_config['activation']])
                # if 'dropout' in layer_config:
                #     fc_pipeline.add_module(f"dropout_{i+1}", nn.Dropout(layer_config['dropout']))
            elif layer_config['type'] == 'FC':
                # Add Fully Connected layers
                fc_pipeline.add_module(f"fc_{i+1}", nn.Linear(prev_units if prev_units else self.config['network']['input_dim'], layer_config['units']))
                if 'activation' in layer_config:
                    fc_pipeline.add_module(f"relu_{i+1}", activations[layer_config['activation']])
                if 'dropout' in layer_config:
                    fc_pipeline.add_module(f"dropout_{i+1}", nn.Dropout(layer_config['dropout']))
                prev_units = layer_config['units']
        
        # Output layer
        fc_pipeline.add_module("output", nn.Linear(prev_units, 2))  # Assuming binary classification (2 output classes)

        self.initialize_weights(fc_pipeline)
        
        return fc_pipeline.to(self.device)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loss(self, pred, label): # change when implementing physics loss
        criterion = nn.MSELoss()
        base_loss = criterion(pred, label)
        return base_loss
    
    def predict(self, img, y=None):

        self.eval()
        with torch.no_grad():
            pred = self(img)
            pred = pred.cpu().numpy()
            return self.dataloader.scaler.inverse_transform(pred), self.dataloader.scaler.inverse_transform(y) if y is not None else None

    def train_predictor(self, epochs=100, learning_rate=1e-3, patience=10, min_delta=0.001, accuracy_threshold=0.05):

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        loss_history = []

        for epoch in range(epochs):

            self.train()
            running_loss = 0.0
            num_batches = 0

            for images, labels in self.dataloader.train_loader:

                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                predictions = self(images)
                loss = self.loss(predictions, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()), max_norm=1.0)  # Clip gradients
                optimizer.step()
                running_loss += loss.item()
                loss_history.append(loss.item())

                num_batches += 1
            avg_loss = running_loss / num_batches

            scheduler.step(avg_loss)

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                
data = EOS_Dataloader('predict')
predictor = Predictor(config='base_decoder.yaml', dataloader=data)

predictor.train_predictor()

x, y = next(iter(data.train_loader))
# x, y = x[0], y[0]
x_t, y_t = next(iter(data.test_loader))
# x_t, y_t = x_t[0], y_t[0]
x = x.to(predictor.device)
x_t = x_t.to(predictor.device)

x_p, y_p = predictor.predict(x, y)
x_t_p, y_t_p = predictor.predict(x_t, y_t)

x_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in x_p.tolist()]
y_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in y_p.tolist()]
x_t_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in x_t_p.tolist()]
y_t_p = [f'[{v[0]:.2f}, {v[1]:.6f}]' for v in y_t_p.tolist()]


print(f'train predictions: {x_p[0]}\ntrain actual {y_p[0]}')
print(f'test predictions: {x_t_p[0]}\ntest actual {y_t_p[0]}')
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import numpy as np
from pkan import PKAN, KANLayer, generate_encoding_set
from eos_dataloader import EOS_Dataloader

eos_dataloader = EOS_Dataloader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = generate_encoding_set(eos_dataloader, train=True)
test_data = generate_encoding_set(eos_dataloader, train=False)

class PKAN_Data(Dataset):

    def __init__(self, data):
        self.X = [copy.copy(x) for x in data['X']]
        self.y = [copy.copy(y) for y in data['y']]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train_pkan(model, train_loader, epochs=20, learning_rate=1e-3, step_size=10, gamma=0.8):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.SmoothL1Loss()  # Huber loss

    model.to(device)  # Move model to GPU if available
    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_history.append(loss.item())

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('PKAN Training Loss')
    plt.show()

    print("PKAN Training Complete!")

    return model

def validate_pkan(model, loader, unscale_loader):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_outputs = []
    all_labels = []

    criterion = nn.SmoothL1Loss()  # Huber loss
    with torch.no_grad():  # Disable gradient computation
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass through PKAN
            loss = criterion(outputs, labels)  # Use CNN's loss function
            total_loss += loss.item()

            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.test_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

    # Convert lists to numpy arrays for further evaluation
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Unscale predictions and labels 

    # fix this, dataloader issues, scalers are not shared between dataset and dataloader
    unscaled_outputs = unscale_loader.inverse_scaler(torch.tensor(all_outputs), train=False)
    unscaled_labels = unscale_loader.inverse_scaler(torch.tensor(all_labels), train=False)

    # Compute evaluation metrics
    mae = np.mean(np.abs(unscaled_outputs - unscaled_labels), axis=0)
    mse = np.mean((unscaled_outputs - unscaled_labels) ** 2, axis=0)

    print(f"Mean Absolute Error (MAE): a={mae[0]:.4f}, b={mae[1]:.4f}")
    print(f"Mean Squared Error (MSE): a={mse[0]:.4f}, b={mse[1]:.4f}")

    return avg_loss, mae, mse

pkan_train_data = PKAN_Data(train_data)
pkan_test_data = PKAN_Data(test_data)

train_loader = DataLoader(pkan_train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(pkan_test_data, batch_size=32, shuffle=True)

kan_layer = KANLayer(64, 64)

pkan = PKAN(kan_layer)

pkan = train_pkan(pkan, train_loader, epochs=100)

validate_pkan(pkan, test_loader, eos_dataloader)
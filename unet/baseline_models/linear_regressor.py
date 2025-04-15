import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from eos_features import EOS_Features_Dataloader

class LinearRegressor(nn.Module):
    def __init__(self, input_size=9, output_size=2):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)
    
    def train_model(self, dataloader, epochs=100, learning_rate=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.train()  # Set model to training mode
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                # Convert to float32 if needed
                inputs = inputs.float()
                targets = targets.float()
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def evaluate(self, dataloader):
        self.eval()  # Set model to evaluation mode
        criterion = nn.MSELoss()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.float()
                targets = targets.float()
                
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Evaluation Loss: {avg_loss:.4f}')
        return avg_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            return self(x).numpy()
        

features_data = EOS_Features_Dataloader()

# Initialize the model
model = LinearRegressor()

# Train the model
model.train_model(features_data.train)

# Evaluate on test set
model.evaluate(features_data.test)

atom_name = 'Thiophene'

features = np.array([84.143, 1.7481, 0.0, 0, 0, 1, 1, 1, 5])
const = [17.21, 0.1058]

input_scaled = features_data.in_scaler.transform(features.reshape(1,-1))

pred = model.predict(input_scaled)

const_pred = features_data.t_scaler.inverse_transform(pred)

print(f'{atom_name} Prediction: a: {const_pred[0][0]:.4f}, b: {const_pred[0][1]:.4f}')
print(f'{atom_name} Actual: a: {const[0]:.4f}, b: {const[1]:.4f}')
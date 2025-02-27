import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# combine with train script and add validation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):

    def __init__(self, dataloader, input_dim):
        super(CNN, self).__init__()

        self.dataloader = dataloader

        self.cnn_pipeline = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros((1, 3, input_dim, input_dim), dtype=torch.float32)
            dummy_output = self.cnn_pipeline(dummy_input)
            flatten_size = dummy_output.view(1,-1).shape[1]

        self.fc_pipeline = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(64, 2)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, predict=False):
        x = self.cnn_pipeline(x)
        x = torch.flatten(x, 1)
        x = self.fc_pipeline(x)
        if predict:
            x = self.output_layer(x)
        return x
    
    def weighted_loss(self, outputs, labels):
        criterion = nn.SmoothL1Loss()
        loss = criterion(outputs, labels)
        weight = torch.tensor([1.0, 80], device=device)  # Adjust if needed
        return (loss * weight).mean()

    def train_model(self, epochs=100, learning_rate=0.0001, patience=10, min_delta=0.001):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        self.to(device)  # Move the model to GPU if available

        loss_history = []
        val_history = []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            running_loss = 0.0
            running_val = 0.0

            for data in self.dataloader.train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs, predict=True)
                loss = self.weighted_loss(outputs, labels)
                loss_history.append(loss.item())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                val_loss = self.train_val()
                val_history.append(val_loss.item())
                running_val += val_loss.item()

            avg_val_loss = running_val / len(self.dataloader.test_loader)
            scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(self.dataloader.train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
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

        plt.figure()
        plt.plot(loss_history, label='Training Loss')
        plt.plot(val_history, label='Validation Loss', color='r')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('CNN Training Loss')
        plt.legend()
        plt.show()

        print("CNN Training Complete!")

        return loss_history, val_history

    
    def train_val(self):

        with torch.no_grad():
            imgs, labels = next(iter(self.dataloader.test_loader))
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = self(imgs, predict=True)
            loss = self.weighted_loss(outputs, labels)
        return loss
    
    def save_model(self, file_path="cnn_model.pth"):

        del self.output_layer

        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
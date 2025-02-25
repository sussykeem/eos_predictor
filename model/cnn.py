import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):

    def __init__(self, input_dim):
        super(CNN, self).__init__()

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
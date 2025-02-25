import torch.nn as nn
import torch
from cnn import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_encoder(path='cnn_model.pth', input_dim=300):
    model = CNN(input_dim)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()
    return model

def generate_encoding_set(data_loader, train=True):
    encoder = load_encoder()
    data = {'X': [], 'y': []}

    loader = data_loader.train_loader if train else data_loader.test_loader
    encodings = []
    labels = []
    for img, label in loader:
        img = img.to(device)
        encodings.append(encoder(img))
        labels.append(label)

    for i, e in enumerate(encodings):
        for j in range(e.shape[0]):
            data['X'].append(e[j])
            data['y'].append(labels[i][j])
    return data

class KANLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_kernels=10):

        super(KANLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_kernels = num_kernels

        self.weights = nn.Parameter(torch.randn(self.output_dim, self.num_kernels))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

        self.centers = nn.Parameter(torch.linspace(-1, 1, self.num_kernels))
        self.widths = nn.Parameter(torch.ones(self.num_kernels) * 0.1)

    def forward(self, x):

        kernels = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) / (2 * self.widths ** 2))
        activation = torch.sum(torch.matmul(kernels, self.weights.T), dim=-1)  + self.bias

        return activation

class PKAN(nn.Module):

    def __init__(self, kan_layer, feature_vec_size=64):
        super(PKAN, self).__init__()

        self.KAN_layer = kan_layer

        self.fc_pipeline = nn.Sequential(
            nn.Linear(feature_vec_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            self.KAN_layer,
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.fc_pipeline(x)
        return x
    

# implement physics loss, replace L1.
# figure out if we use trend loss
# consider other physical constraints
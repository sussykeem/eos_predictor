import sys
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
from eos_dataloader import EOS_Dataset
from cnn import CNN
from pkan import PKAN, KANLayer
import matplotlib.pyplot as plt

# Set random seeds
seed = 42
torch.manual_seed(seed)

class MolPredictor():

    def __init__(self):
        sys.stdout.write('Loading encoder\n')
        self.encoder = self.load_encoder()
        sys.stdout.write('Encoder loaded, loading pkan\n')
        self.pkan = self.load_pkan()
        sys.stdout.write('pkan loaded\n')
        self.train = EOS_Dataset()

    def load_encoder(self, path='mvp/cnn_model.pth', input_dim=300):
        try:
            model = CNN(input_dim=input_dim)
            # strict=False allows us to use the model with the last layer dropped
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f'Failed {e}')
        return model
    
    def load_pkan(self, path='mvp/pkan_model.pth'):
        try:
            model = PKAN(kan_layer=KANLayer(64, 64))
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Failed {e}")
        return model

    def predict(self, im_path):
        with torch.no_grad():
            image = Image.open(im_path).convert("RGB")
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            try:
                image_tensor = transform(image).unsqueeze(0).to(device)
                preds = self.encoder(image_tensor, predict=True)
                #preds = self.pkan(encoding)
                preds_unscaled = self.train.inverse_transform(preds.cpu())
            except Exception as e:
                print(f"Failed {e}")

        return preds_unscaled[0][0] , preds_unscaled[0][1]

def main(x):
    predictor = MolPredictor()
    a, b = predictor.predict(x)
    print("Prediction\n")
    sys.stdout.write(f"a: {a:.4f}\n")
    sys.stdout.write(f"b: {b:.4f}\n")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    argv = parser.parse_args()
    main(argv.file_path)
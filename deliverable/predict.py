import sys
import argparse
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from eos_dataloader import EOS_Dataset
from build_model import Model
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set random seeds
seed = 42
torch.manual_seed(seed)

class MolPredictor():

    def __init__(self, encoder_path, decoder_path):
        sys.stdout.write('Loading encoder\n')
        self.encoder = self.load_encoder(path=encoder_path)
        sys.stdout.write('Encoder loaded\n')

        if 'pth' in decoder_path:
            sys.stdout.write('Loading pkan\n')
            self.pkan = self.load_decoder(path=decoder_path)
            sys.stdout.write('pkan loaded\n')
            self.e_pred = False
        else:
            self.e_pred = True
        self.train = EOS_Dataset()

    def load_encoder(self, path):
        try:

            weight_path = path[0]
            model_path = path[1]

            model = Model(model_path)
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
            model.to(device)
            model.train()
        except Exception as e:
            print(f'Failed {e}')
        return model
    
    def load_decoder(self, path):
        try:

            weight_path = path[0]
            model_path = path[1]

            model = Model(model_path)
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
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
                a_preds = []
                b_preds = []
                for i in range(1000): # num MC prediction
                    preds = self.encoder(image_tensor, predict=self.e_pred)
                    if not self.e_pred:
                        preds = self.pkan(preds)
                    preds_unscaled = self.train.inverse_transform(preds.cpu())
                    a_preds.append(preds_unscaled[0][0])
                    b_preds.append(preds_unscaled[0][1])

                a = self.plot_distribution(a_preds, 'a')
                b = self.plot_distribution(b_preds, 'b')
                
            except Exception as e:
                print(f"Failed {e}")

        return a, b
    
    def plot_distribution(self, preds, title):

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
        plt.title(f"MC Dropout Prediction Distribution - {title}")
        plt.legend()
        plt.show()
        return dist

def main(x, e_path, d_path):
    predictor = MolPredictor(e_path, d_path)
    a, b = predictor.predict(x)
    print("Prediction\n")
    sys.stdout.write(f"a:\n")
    sys.stdout.write(f"\tmean: {a[0]:.4f}, std: {a[1]:.4f}\n")
    sys.stdout.write(f'\tmedian: {a[2]:.4f}, mode: {a[3]:.4f}\n')
    sys.stdout.write(f"\tci: ({a[4]:.4f}, {a[5]:.4f})\n")
    sys.stdout.write(f"b:\n")
    sys.stdout.write(f"\tmean: {b[0]:.4f}, std: {b[1]:.4f}\n")
    sys.stdout.write(f'\tmedian: {b[2]:.4f}, mode: {b[3]:.4f}\n')
    sys.stdout.write(f"\tci: ({b[4]:.4f}, {b[5]:.4f})\n")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('encoder_path', type=str)
    parser.add_argument('decoder_path', type=str)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    argv = parser.parse_args()
    main(argv.file_path, argv.encoder_path, argv.decoder_path)
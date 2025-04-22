import sys
import argparse
from PIL import Image
import torch
from joblib import load
import numpy as np
import torchvision.transforms as transforms
from eos_dataloader import EOS_Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from predictor import Predictor
from baseline_models.cnn import CNN
from baseline_models.random_forest import RandomForestModel
from baseline_models.linear_regressor import LinearRegressor
from baseline_models.pinn import PINN
from baseline_models.pkan import PKAN
from baseline_models.mlp import MLP
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from baseline_models.eos_features import EOS_Features_Dataset
from sklearn.preprocessing import StandardScaler

# Set random seeds
seed = 42
torch.manual_seed(seed)

class AddRandomNoise:
    def __init__(self, noise_range=0.05):
        self.noise_range = noise_range  # Max magnitude of noise

    def __call__(self, img):
        # img: Tensor [C, H, W] with values in [0, 1]
        noise = torch.empty_like(img).uniform_(-self.noise_range, self.noise_range)
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0.0, 1.0)

class MolPredictor():

    def __init__(self):
        self.types = ['random_forest', 'linear', 'mlp', 'pinn', 'pkan', 'cnn']
        self.models = [RandomForestModel, LinearRegressor, MLP, PINN, PKAN, CNN, Predictor]
        self.models_path = ['base_model_weights/random_forest.pth', 'base_model_weights/linear.pth', 'base_model_weights/mlp.pth', 'base_model_weights/pinn.pth', 'base_model_weights/pkan.pth', 'base_model_weights/cnn.pth']
        self.in_scaler = EOS_Features_Dataset(in_scale=StandardScaler()).in_scale
        sys.stdout.write('Loading Baselines\n')
        self.models_init = self.load_baselines()
        sys.stdout.write('Baselines Loaded\n')
        sys.stdout.write('Loading Predictor\n')
        self.predictor = self.load_predictor()
        sys.stdout.write('Predictor Loaded\n')

        self.train = EOS_Dataset()

    def load_predictor(self, model_path='base_decoder.yaml', weight_path='../predictor.pth'):
        try:
            model = Predictor(model_path)
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f'Failed {e}')
        return model
    
    def load_baseline(self, type='', data=None):
        assert type in self.types, 'Please input a valid model type'

        model_idx = self.types.index(type)

        self.model = self.models[model_idx](data)

        if type != 'random_forest':
            self.model.load_state_dict(torch.load(self.models_path[model_idx], weights_only=True))
        else:
            self.model = load(self.models_path[model_idx])
    
    def load_baselines(self):
        models_init = []
        try:
            for type in self.types:
                if type == 'cnn':
                    model = self.load_baseline(type, self.i_data)
                elif type == 'predictor':
                    model = self.load_baseline(type, self.p_data)
                else:    
                    model = self.load_baseline(type, self.f_data)
                models_init.append(model)
                return models_init
        except Exception as e:
            sys.stderr.write("Error loading baselines: {e}")  

    def extract_molecular_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        features = {
            'MolecularWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'RotatableBonds': Lipinski.NumRotatableBonds(mol),
            'HBondDonors': Lipinski.NumHDonors(mol),
            'HBondAcceptors': Lipinski.NumHAcceptors(mol),
            'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'NumRings': rdMolDescriptors.CalcNumRings(mol),
            'AtomCount': mol.GetNumAtoms()
        }

        return list(features.values())  # return as a feature vector

    def predict(self, model, input):
        
        with torch.no_grad():
            try:
                a_preds = []
                b_preds = []
                for i in range(1000): # num MC prediction
                    preds = model.predict(input)
                    a_preds.append(preds[0][0])
                    b_preds.append(preds[0][1])
                
            except Exception as e:
                print(f"Failed {e}")

        return a_preds, b_preds
    
    def run_models(self, img_path, smi_path):
        image = Image.open(img_path).convert("RGB")
        smile_file = open(smi_path, 'r')
        smile = smile_file.read()
        mol_features = self.extract_molecular_features(smile)
        mol_features = self.in_scaler.transform(mol_features)

        transform_p = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda img: F.invert(img)),
                transforms.RandomApply([AddRandomNoise(noise_range=0.25)], p=1),
        ])
        transform_c = transforms.Compose([
                transforms.ToTensor(),
        ])

        output_data = {}

        for model in self.models_init:
            if type(model) == type(CNN):
                transform = transform_c
                input = transform(image).unsqueeze(0).to(device)
            elif type(model) == type(Predictor):
                transform = transform_p
                input = transform(image).unsqueeze(0).to(device)
            else:
                input = mol_features
            a_preds, b_preds = self.predict(model, input)
            a = self.plot_distribution(a_preds, 'a', model.__class__.__name__)
            b = self.plot_distribution(b_preds, 'b', model.__class__.__name__)

            preds = {'a': a, 'b': b}

            output_data[model.__class__.__name__] = preds
        
        return output_data
    
    def plot_distribution(self, preds, title, model_name):

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

def main(im_path, smi_path):

    predictor = MolPredictor()
    a, b = predictor.run_models(im_path, smi_path)
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
    print('run')
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('encoder_m_path', type=str)
    parser.add_argument('encoder_w_path', type=str)
    parser.add_argument('decoder_path', type=str)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    argv = parser.parse_args()
    encoder_path = (argv.encoder_w_path, argv.encoder_m_path)
    main(argv.file_path, encoder_path, argv.decoder_path)
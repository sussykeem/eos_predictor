from mlp import MLP
from pinn import PINN
from pkan import PKAN
from linear_regressor import LinearRegressor
from random_forest import RandomForestModel
from cnn import CNN
from deliverable.predictor import Predictor
import torch
import numpy as np
from eos_features import EOS_Features_Dataloader
from eos_dataloader import EOS_Dataloader
from deliverable.eos_dataloader import EOS_Dataloader as Eos_Predictloader
from joblib import load
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
import os

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(sys.prefix, "Library", "plugins", "platforms")

class Baseline_Model():

    def __init__(self, type, data):

        self.types = ['random_forest', 'linear', 'mlp', 'pinn', 'pkan', 'cnn']

        self.models = [RandomForestModel, LinearRegressor, MLP, PINN, PKAN, CNN, Predictor]
        self.models_path = ['base_model_weights/random_forest.pth', 'base_model_weights/linear.pth', 'base_model_weights/mlp.pth', 'base_model_weights/pinn.pth', 'base_model_weights/pkan.pth', 'base_model_weights/cnn.pth', '../predictor.pth']

        assert type in self.types, 'Please input a valid model type'

        model_idx = self.types.index(type)

        self.model = self.models[model_idx](data)

        if type != 'random_forest':
            self.model.load_state_dict(torch.load(self.models_path[model_idx], weights_only=True))
        else:
            self.model = load(self.models_path[model_idx])

    
class Baselines():

    def __init__(self, f_data, i_data, p_data):
        self.types = ['random_forest', 'linear', 'mlp', 'pinn', 'pkan', 'cnn', 'predictor']
        # self.types = ['linear']
        self.models = []
        self.f_data = f_data
        self.i_data = i_data
        self.p_data = p_data

        for type in self.types:
            if type == 'cnn':
                model = Baseline_Model(type, self.i_data)
            elif type == 'predictor':
                model = Baseline_Model(type, self.p_data)
            else:    
                model = Baseline_Model(type, self.f_data)
            self.models.append(model)
            

    def test_models(self):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

        for i in range(len(self.types)):

            print(f'{self.types[i]}:')

            model = self.models[i].model

            a_data, b_data = model.test_model()

            a_avg = np.mean(np.abs(a_data[1] - a_data[0]))
            b_avg = np.mean(np.abs(b_data[1] - b_data[0]))

            print(f'\tA MAE: {a_avg:.4f}, B MAE: {b_avg:.4f}')

            a_x = np.linspace(np.min(a_data[0]), np.max(a_data[0]), a_data.shape[1])
            b_x = np.linspace(np.min(b_data[0]), np.max(b_data[0]), b_data.shape[1])

            a_slope, a_intercept, _, _, _ = linregress(a_data[0], a_data[1])
            b_slope, b_intercept, _, _, _ = linregress(b_data[0], b_data[1])
            a_line = a_slope * a_x + a_intercept
            b_line = b_slope * b_x + b_intercept

            ax[0].plot(a_x, a_line, label=f'{self.types[i]}')

            ax[1].plot(b_x, b_line, label=f'{self.types[i]}')

        ax[0].plot(a_x, a_x, label=f'True A', color='green')

        ax[1].plot(b_x, b_x, label=f'True B', color='green')

        
        ax[0].set_title('A Prediction Accuracy in Baseline Models')
        ax[0].set_ylabel('A Prediction')
        ax[0].set_xlabel('A Actual')
        ax[0].legend()
        ax[1].set_title('B Prediction Accuracy in Baseline Models')
        ax[1].set_ylabel('B Prediction')
        ax[1].set_xlabel('B Actual')
        ax[1].legend()

        plt.tight_layout()
        plt.show()



f_data = EOS_Features_Dataloader()
i_data = EOS_Dataloader(mode='predict')
p_data = Eos_Predictloader(mode='predict')

baselines = Baselines(f_data, i_data, p_data)

baselines.test_models()


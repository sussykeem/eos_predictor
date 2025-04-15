from mlp import MLP
from pinn import PINN
from pkan import PKAN
from linear_regressor import LinearRegressor
from random_forest import RandomForestModel
import torch
import numpy as np
from eos_features import EOS_Features_Dataloader
from joblib import load
import matplotlib.pyplot as plt

class Baseline_Model():

    def __init__(self, type, data):

        self.types = ['random_forest', 'linear', 'mlp', 'pinn', 'pkan']

        self.models = [RandomForestModel, LinearRegressor, MLP, PINN, PKAN]
        self.models_path = ['base_model_weights/random_forest.pth', 'base_model_weights/linear.pth', 'base_model_weights/mlp.pth', 'base_model_weights/pinn.pth', 'base_model_weights/pkan.pth']

        assert type in self.types, 'Please input a valid model type'

        model_idx = self.types.index(type)

        self.model = self.models[model_idx](data)

        if type != 'random_forest':
            self.model.load_state_dict(torch.load(self.models_path[model_idx], weights_only=True))
        else:
            self.model = load(self.models_path[model_idx])

    
class Baselines():

    def __init__(self, data):
        self.types = ['random_forest', 'linear', 'mlp', 'pinn', 'pkan']

        self.models = []
        self.data = data

        for type in self.types:
            model = Baseline_Model(type, self.data)
            self.models.append(model)

    def test_models(self):
        for i in range(len(self.types)):

            print(f'{self.types[i]}:')

            model = self.models[i].model

            model.test_model()
        


data = EOS_Features_Dataloader()

baselines = Baselines(data)

baselines.test_models()


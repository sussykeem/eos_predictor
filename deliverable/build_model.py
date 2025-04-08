import yaml
from model_classes.cnn import CNN
from model_classes.pkan import PKAN
from eos_dataloader import EOS_Dataloader, EOS_Dataset
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model():

    def __init__(self, config_path, encoder=None):
        self.config_path = config_path
        self.encoder = encoder
        
        train_data = EOS_Dataset(scale=True, train=True)
        test_data = EOS_Dataset(scale=True, train=False)

        self.eos_dataloader = EOS_Dataloader(train_data, test_data)
        
        with open(self.config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.model_type = self.config['network']['type']

        if self.model_type == 'CNN':
            self.build_cnn()
        elif self.model_type == 'PKAN':
            self.build_pkan()
        
        self.network.to(device)
        print(self.network)
    
    def build_cnn(self):
        self.network = CNN(self.config, self.eos_dataloader)

    def build_pkan(self):
        if self.encoder is None:
            print('Encoder not found')
            return
        self.network = PKAN(self.config, self.eos_dataloader, self.encoder)


def main():

    encoder = Model('model_config/base_cnn.yaml')
    decoder = Model('model_config/base_pkan.yaml', encoder.network)

if __name__ == "__main__":
    main()
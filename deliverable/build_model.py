import yaml
from model_classes.cnn import CNN
from model_classes.pkan import PKAN
from eos_dataloader import EOS_Dataloader

class Model():

    def __init__(self, config_path):
        self.config_path = config_path

        self.eos_dataloader = EOS_Dataloader()
        
        with open(self.config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.model_type = self.config['network']['type']

        if self.model_type == 'CNN':
            self.build_cnn()
        elif self.model_type == 'PKAN':
            self.build_pkan()
        print(self.network)
    
    def build_cnn(self):
        self.network = CNN(self.config)

    def build_pkan(self):
        self.network = PKAN(self.config)


def main():

    model = Model('model_config/base_cnn.yaml')
    model = Model('model_config/base_pkan.yaml')

if __name__ == main():
    main()
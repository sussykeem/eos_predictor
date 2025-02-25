from eos_dataloader import EOS_Dataloader
from cnn import CNN
from pkan import PKAN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_encoder(path='cnn_model.pth', input_dim=300):
    model = CNN(input_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_pkan(path='pkan_model.pth')
    model = PKAN()


def generate_img_set(url):

    return

class EOS_Predictor():


eos = EOS_Dataloader()
print(eos.train_loader)
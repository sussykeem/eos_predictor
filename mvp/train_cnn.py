import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from eos_dataloader import EOS_Dataloader, EOS_Dataset
from cnn import CNN


train_data = EOS_Dataset(scale=True, train=True)
test_data = EOS_Dataset(scale=True, train=False)

eos_dataloader = EOS_Dataloader(train_data, test_data)

cnn = CNN(eos_dataloader, 300)

cnn.train_model()

cnn.save_model()
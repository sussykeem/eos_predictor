from pkan import PKAN, KANLayer, PKAN_Data
from eos_dataloader import EOS_Dataloader, EOS_Dataset

train_data = EOS_Dataset(scale=True, train=True)
test_data = EOS_Dataset(scale=True, train=False)

eos_dataloader = EOS_Dataloader(train_data, test_data)

pkan_train_data = PKAN_Data(eos_dataloader)
pkan_test_data = PKAN_Data(eos_dataloader, False)

kan_layer = KANLayer(64, 64)

pkan = PKAN(kan_layer, pkan_train_data, pkan_test_data)

pkan.train_pkan()

pkan.validate_pkan(test_data)

pkan.save_model()
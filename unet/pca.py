from sklearn.decomposition import PCA
import torch
import numpy as np
from eos_dataloader import EOS_Dataloader
from unet_2 import Unet, Encoder

def extract_data_from_dataloader(dataloader, max_batches=None):
    all_images = []
    
    for i, (images, _) in enumerate(dataloader):
        # Move to CPU and convert to numpy
        images_np = images.view(images.size(0), -1).cpu().numpy()
        all_images.append(images_np)

        if max_batches and i + 1 >= max_batches:
            break

    X = np.concatenate(all_images, axis=0)
    return X

# Example: assume X has shape (num_samples, height, width, channels)
def run_pca_on_images(X, explained_variance=0.99):
    # Reshape images to flat vectors: (num_samples, height * width * channels)
    num_samples = X.shape[0]
    flat_X = X.reshape(num_samples, -1)

    # Fit PCA
    pca = PCA(n_components=explained_variance)
    X_reduced = pca.fit_transform(flat_X)

    # Number of components to retain 95% variance
    n_components = pca.n_components_
    explained = np.sum(pca.explained_variance_ratio_)

    print(f"Selected {n_components} components to preserve {explained:.2%} variance")
    return X_reduced, n_components, pca

# unet_data = EOS_Dataloader(mode='predict')
# full_unet = Unet(unet_data)
# full_unet.load_state_dict(torch.load('unet2_model.pth', weights_only=True))
# full_unet.eval()
# encoder = Encoder(full_unet)
# encoder.to(full_unet.device)
# encoder.eval()
# # Freeze only shallow layers — e.g., first two downsampling blocks
# for name, param in encoder.named_parameters():
#     #if any(layer in name for layer in [""]):
#     #    param.requires_grad = False
#     #else:
#     #    param.requires_grad = True  # Fine-tune deeper layers
#     param.requires_grad = False  # Freeze all layer

# features = []

# for input, target, _ in unet_data.train_loader:
#     input = input.to(full_unet.device)
#     target = target.to(full_unet.device)
#     with torch.no_grad():
#         embedding = encoder(input).view(input.size(0), -1).cpu().numpy()
#         features.append(embedding)

# X = np.concatenate(features, axis=0) 

data = EOS_Dataloader(mode='reconstruct', num=5000)
X = extract_data_from_dataloader(data.train_loader)
run_pca_on_images(X)
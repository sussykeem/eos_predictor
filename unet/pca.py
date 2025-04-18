from sklearn.decomposition import PCA
import torch
import numpy as np
from eos_dataloader import EOS_Dataloader

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
def run_pca_on_images(X, explained_variance=0.95):
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


data = EOS_Dataloader()

X = extract_data_from_dataloader(data.train_loader)

run_pca_on_images(X)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision.transforms import CenterCrop
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torch.optim as optim
import matplotlib.pyplot as plt

from eos_dataloader import EOS_Dataloader

class UNet(nn.Module):
    def __init__(self, dataloader, patch_size=16, mask_ratio=0.5):
        super(UNet, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.dataloader = dataloader
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Encoder block (downsampling)
        self.encoder1 = self.conv_block(3, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        
        # Decoder block (upsampling)
        self.upscale4 = self.upscale_block(512, 256)
        self.decoder4 = self.upconv_block(512, 256)
        self.upscale3 = self.upscale_block(256, 128)
        self.decoder3 = self.upconv_block(256, 128)
        self.upscale2 = self.upscale_block(128, 64)
        self.decoder2 = self.upconv_block(128, 64)
        self.upscale1 = self.upscale_block(64, 32)
        self.decoder1 = self.upconv_block(64, 32)
        
        # Final output layer
        self.upscale0 = self.upscale_block(32, 3)
        self.final_conv = nn.Conv2d(6, 3, kernel_size=1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def upscale_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )
    
    def mask_input(self, x):
        """ Masks random patches of the input tensor (x) """
        batch_size, channels, height, width = x.size()
        
        # Create grid of patches
        patch_h = self.patch_size
        patch_w = self.patch_size
        
        patches_h = height // patch_h
        patches_w = width // patch_w
        
        # Create mask for patches
        mask = torch.ones(batch_size, channels, height, width).to(x.device)
        
        # Randomly mask patches
        num_patches_to_mask = int(patches_h * patches_w * self.mask_ratio)
        mask_positions = random.sample(range(patches_h * patches_w), num_patches_to_mask)
        
        # Iterate over the patches and mask them
        for pos in mask_positions:
            row = pos // patches_w
            col = pos % patches_w
            start_h, end_h = row * patch_h, (row + 1) * patch_h
            start_w, end_w = col * patch_w, (col + 1) * patch_w
            mask[:, :, start_h:end_h, start_w:end_w] = 0  # Mask the patch

        # Apply mask to the input image
        masked_x = x * mask
        
        return masked_x, mask

    def forward(self, x, enc=False):
        # Encoder path (down-sampling)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        bottleneck_flat = self.flatten(bottleneck)
        encoded = self.fc(bottleneck_flat)
        
        # Decoder path (up-sampling)
        
        up4  = self.upscale4(bottleneck)
        up4  = self.crop(enc4, up4)
        up4  = torch.cat([up4, enc4], dim=1)
        dec4 = self.decoder4(up4)
        up3  = self.upscale3(dec4)
        up3  = self.crop(enc3, up3)
        up3  = torch.cat([up3, enc3], dim=1)
        dec3 = self.decoder3(up3)
        up2  = self.upscale2(dec3)
        up2  = self.crop(enc2, up2)
        up2  = torch.cat([up2, enc2], dim=1)
        dec2 = self.decoder2(up2)
        up1  = self.upscale1(dec2)
        up1  = self.crop(enc1, up1)
        up1  = torch.cat([up1, enc1], dim=1)
        dec1 = self.decoder1(up1)
        
        # Final output layer
        up0  = self.upscale0(dec1)
        up0  = self.crop(x, up0)
        up0  = torch.cat([up0, x], dim=1)
        output = self.final_conv(up0)
        
        if enc:
            return encoded

        return output
    
    def compute_loss(self, output, target, mask):
        """Compute the loss only on the masked patches"""
        masked_output = output * mask.float()
        masked_target = target * mask.float()
        
        # Calculate Mean Squared Error loss only on the masked patches
        loss = F.mse_loss(masked_output, masked_target, reduction='mean')
        
        return loss
    
    def compute_ssim(self, pred, target):
        if pred.max() > 1.0:
            pred = pred / 255.0
        if target.max() > 1.0:
            target = target / 255.0

        return ssim_metric(pred, target)
    
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures
    
    def train_unet(self, epochs=10, learning_rate=0.001, patience=10, min_delta=0.001):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        self.to(self.device)  # Move the model to GPU if available

        loss_history = []
        val_history = []
        train_acc_history = []
        val_acc_history = []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            running_loss = 0.0
            running_val = 0.0
            running_train_acc = 0.0
            running_val_acc = 0.0

            for data in self.dataloader.train_loader:
                input, labels = data
                # Mask the input
                masked_input, mask = self.mask_input(input)
                masked_input, labels = masked_input.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(masked_input)
                loss = self.compute_loss(outputs, labels, mask)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_train_acc += np.sqrt(loss.item())

            # Validation
            val_loss, val_acc = self.train_val()
            val_history.append(val_loss)
            val_acc_history.append(val_acc)

            # Compute average losses and accuracy
            avg_train_loss = running_loss / len(self.dataloader.train_loader)
            avg_train_acc = running_train_acc / len(self.dataloader.train_loader)
            avg_val_loss = val_loss
            avg_val_acc = val_acc

            loss_history.append(avg_train_loss)
            train_acc_history.append(avg_train_acc)

            #scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_train_loss:.4f} | Accuracy: {avg_train_acc:.4f}")
            print(f"\tValidation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_acc:.4f}")

            # Early Stopping Logic
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_weights = self.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        # Restore best model before returning
        if best_model_weights:
            self.load_state_dict(best_model_weights)
            print("Restored best model weights.")

        plt.figure()
        plt.plot(loss_history, label='Training Loss')
        plt.plot(val_history, label='Validation Loss', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('CNN Training Loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_acc_history, label='Training Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (Lower is Better)')
        plt.title('CNN Accuracy')
        plt.legend()
        plt.show()

        print("CNN Training Complete!")

        return loss_history, val_history, train_acc_history, val_acc_history

    
    def train_val(self):
        with torch.no_grad():
            total_loss = 0.0
            total_acc = 0.0
            num_batches = len(self.dataloader.test_loader)

            for imgs, labels in self.dataloader.test_loader:
                masked_imgs, mask = self.mask_input(imgs)
                masked_imgs, labels = masked_imgs.to(self.device), labels.to(self.device)

                outputs = self(imgs)
                loss = self.compute_loss(outputs, labels, mask)
                acc = np.sqrt(loss.item())

                total_loss += loss.item()
                total_acc += acc

        return total_loss / num_batches, total_acc / num_batches  # Return averaged values

data = EOS_Dataloader()

x = next(iter(data.train_loader))
x = x[0]

x_t = next(iter(data.test_loader))
x_t = x_t[0]

unet = UNet(data, patch_size=16, mask_ratio=0.25)

x = x.to(unet.device)
x_t = x_t.to(unet.device)

loss_history, val_history, train_acc_history, val_acc_history = unet.train_unet()

x_recon = unet(x).detach().cpu().numpy()[0]
x_t_recon = unet(x_t).detach().cpu().numpy()[0]

x_i = x.detach().cpu().numpy()[0]
x_t_i = x_t.detach().cpu().numpy()[0]

x_i = np.transpose(x_i, (1,2,0))
x_t_i = np.transpose(x_t_i, (1,2,0))
x_recon = np.transpose(x_recon, (1,2,0))
x_t_recon = np.transpose(x_t_recon, (1,2,0))

def normalize_array(arr):
    """Normalize a NumPy array to the [0, 1] range."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if (max_val - min_val) > 0:
        return (arr - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(arr)


x_i = normalize_array(x_i)
x_t_i = normalize_array(x_t_i)
x_recon = normalize_array(x_recon)
x_t_recon = normalize_array(x_t_recon)

fig, ax = plt.subplots(nrows=2, ncols=2)



ax[0][0].imshow(x_i)
ax[0][0].set_title("Original Image - Train")
ax[0][0].axis("off")
ax[0][1].imshow(x_recon)
ax[0][1].set_title("Reconstructed Image")
ax[0][1].axis("off")

ax[1][0].imshow(x_t_i)
ax[1][0].set_title("Original Image - Test")
ax[1][0].axis("off")
ax[1][1].imshow(x_t_recon)
ax[1][1].set_title("Reconstructed Image")
ax[1][1].axis("off")

plt.tight_layout()
plt.show()


x = next(iter(data.train_loader))[0]
x.to(unet.device)

h = unet(x, enc=True)

h = h.detach().cpu().numpy()

fig, ax = plt.subplots(nrows=1,ncols=1)

ax.imshow(h)
ax.set_title('Encodings')
ax.axis('off')

plt.tight_layout()
plt.show()
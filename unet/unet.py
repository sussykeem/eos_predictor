import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision.transforms import CenterCrop
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from eos_dataloader import EOS_Dataloader

class Unet(nn.Module):
    def __init__(self, dataloader, patch_size=16, mask_ratio=0.5):
        super(Unet, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

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
        self.dropout = nn.Dropout(p=0.3)  # Add this dropout layer
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def upscale_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )
    
    def mask_input(self, x):
        """ Masks random patches of the input tensor (x), independently for each sample in the batch """
        batch_size, channels, height, width = x.size()

        patch_h = patch_w = self.patch_size
        patches_h = height // patch_h
        patches_w = width // patch_w
        total_patches = patches_h * patches_w
        num_patches_to_mask = int(total_patches * self.mask_ratio)

        mask = torch.ones_like(x)

        for i in range(batch_size):
            mask_positions = random.sample(range(total_patches), num_patches_to_mask)
            for pos in mask_positions:
                row = pos // patches_w
                col = pos % patches_w
                h0, h1 = row * patch_h, (row + 1) * patch_h
                w0, w1 = col * patch_w, (col + 1) * patch_w
                mask[i, :, h0:h1, w0:w1] = 0

        masked_x = x * mask
        return masked_x, mask


    def forward(self, x, enc=False):
        # Encoder path (down-sampling)
        enc1 = self.encoder1(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        bottleneck = self.dropout(bottleneck)  # Apply dropout
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
        output = torch.tanh(self.final_conv(up0))
        
        if enc:
            return encoded

        return output
    
    def compute_loss(self, output, target, mask):
    def compute_loss(self, output, target, mask):
        """Compute the loss only on the masked patches"""
        # Ensure the mask is in the same device as the output and target
        # mask = mask.to(output.device)
        
        # # Calculate Mean Squared Error loss only on the masked patches
        # loss = F.mse_loss(output, target, reduction='mean')
        ssim_loss = 1 - self.ssim(self.denormalize(output), self.denormalize(target))
        mse_loss  = F.mse_loss(output, target, reduction='mean')
        return 0*ssim_loss + 1*mse_loss
    
    def compute_ssim(self, pred, target):

        pred = self.denormalize(pred)
        target = self.denormalize(target)

        with torch.no_grad():
            return self.ssim(pred, target).cpu()
    
    def denormalize(self, tensor):
        """Denormalize a tensor image."""
        return tensor.to(device=self.device) * self.std + self.mean
    
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures
    
    def train_unet(self, epochs=1000, learning_rate=0.01, patience=10, min_delta=0.001):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        scaler = GradScaler()  # Initialize GradScaler for mixed precision training

        self.to(self.device)  # Move the model to GPU if available

        loss_history = []
        val_history = []
        train_smi_history = []
        train_rmse_history = []
        val_rmse_history = []
        val_smi_history = []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        self.train()  # Set the model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            running_smi = 0.0
            running_train_rmse = 0.0

            for data in self.dataloader.train_loader:
                input, labels = data
                # Mask the input
                masked_input, mask = self.mask_input(input)
                masked_input, labels = masked_input.to(self.device), labels.to(self.device)
                input, labels = data
                # Mask the input
                masked_input, mask = self.mask_input(input)
                masked_input, labels = masked_input.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                # Use autocast for mixed precision during forward pass
                with autocast(device_type='cuda'):  # Forward pass in mixed precision
                    outputs = self(masked_input)
                    loss = self.compute_loss(outputs, labels, mask)

                # Scale the loss and backpropagate
                scaler.scale(loss).backward()

                # Optimizer step
                scaler.step(optimizer)

                # Update the scaler
                scaler.update()

                running_loss += loss.item()
                running_train_rmse += np.sqrt(loss.item())

                running_smi  += self.compute_ssim(outputs, labels)
    

            # Validation
            avg_val_loss, avg_val_rmse, avg_val_smi = self.train_val()
            val_history.append(avg_val_loss)
            val_rmse_history.append(avg_val_rmse)
            val_smi_history.append(avg_val_smi)

            # Compute average losses and accuracy
            avg_train_loss = running_loss / len(self.dataloader.train_loader)
            avg_train_rmse = running_train_rmse / len(self.dataloader.train_loader)
            avg_train_smi  = running_smi / len(self.dataloader.train_loader)

            loss_history.append(avg_train_loss)
            train_rmse_history.append(avg_train_rmse)
            train_smi_history.append(avg_train_smi)

            scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_train_loss:.4f} | RMSE: {avg_train_rmse:.4f} | SSIM: {avg_train_smi:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"\tValidation Loss: {avg_val_loss:.4f} | Validation RMSE: {avg_val_rmse:.4f} | Validation SSIM: {avg_val_smi:.4f}")

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
        plt.title('Unet Training Loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_rmse_history, label='Training RMSE')
        plt.plot(val_rmse_history, label='Validation Accuracy', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE (Lower is Better)')
        plt.title('Unet RMSE')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_smi_history, label='Training SSIM')
        plt.plot(val_smi_history, label='Validation SSIM', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('SSIM (Higher is Better)')
        plt.title('Unet SSIM')
        plt.legend()
        plt.show()


        print("Unet Training Complete!")

        return loss_history, val_history, train_rmse_history, val_rmse_history

    
    def train_val(self):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            total_loss = 0.0
            total_rmse = 0.0
            total_smi  = 0.0
            num_batches = len(self.dataloader.test_loader)

            for imgs, labels in self.dataloader.test_loader:
                masked_imgs, mask = self.mask_input(imgs)
                masked_imgs, labels = masked_imgs.to(self.device), labels.to(self.device)
                masked_imgs, mask = self.mask_input(imgs)
                masked_imgs, labels = masked_imgs.to(self.device), labels.to(self.device)

                # Use autocast for mixed precision during forward pass
                with autocast(device_type='cuda'):  # Forward pass in mixed precision
                    outputs = self(masked_imgs)
                    loss = self.compute_loss(outputs, labels, mask)

                rmse = np.sqrt(loss.item())

                total_loss += loss.item()
                total_rmse += rmse

                total_smi  += self.compute_ssim(outputs, labels)

        return total_loss / num_batches, total_rmse / num_batches, total_smi / num_batches   # Return averaged values
    
    def save_model(self, file_path="unet_model.pth"):
        """ Save the model state dictionary to a file """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

data = EOS_Dataloader()

x = next(iter(data.train_loader))
x = x[0]

x_t = next(iter(data.test_loader))
x_t = x_t[0]

unet = Unet(data, patch_size=16, mask_ratio=0.25)

x = x.to(unet.device)
x_t = x_t.to(unet.device)

loss_history, val_history, train_rmse_history, val_rmse_history = unet.train_unet()

x_recon = unet(x).detach().cpu()[0]
x_t_recon = unet(x_t).detach().cpu()[0]

x_i = x.detach().cpu()[0]
x_t_i = x_t.detach().cpu()[0]

x_i = unet.denormalize(x_i).cpu()
x_t_i = unet.denormalize(x_t_i).cpu()
x_recon = unet.denormalize(x_recon).cpu()
x_t_recon = unet.denormalize(x_t_recon).cpu()

x_i = x_i[0]
x_t_i = x_t_i[0]
x_recon = x_recon[0]
x_t_recon = x_t_recon[0]

x_i = np.transpose(x_i.numpy(), (1,2,0))
x_t_i = np.transpose(x_t_i.numpy(), (1,2,0))
x_recon = np.transpose(x_recon.numpy(), (1,2,0))
x_t_recon = np.transpose(x_t_recon.numpy(), (1,2,0))

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0][0].imshow(x_i)
ax[0][0].set_title("Original Image - Train")
ax[0][0].axis("off")
ax[0][1].imshow(x_recon)
ax[0][1].set_title("Reconstructed Image")
ax[0][1].axis("off")

ax[1][0].imshow(x_t_i)
ax[1][0].imshow(x_t_i)
ax[1][0].set_title("Original Image - Test")
ax[1][0].axis("off")
ax[1][1].imshow(x_t_recon)
ax[1][1].set_title("Reconstructed Image")
ax[1][1].axis("off")

plt.tight_layout()
plt.show()


x = next(iter(data.train_loader))[0]
x = x.to(unet.device)

h = unet(x, enc=True)

h = h.detach().cpu().numpy()

fig, ax = plt.subplots(nrows=1,ncols=1)

ax.imshow(h)
ax.set_title('Encodings')
ax.axis('off')

plt.tight_layout()
plt.show()

unet.save_model("unet_model.pth")


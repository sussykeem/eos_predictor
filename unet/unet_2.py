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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from eos_dataloader import EOS_Dataloader

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SelfAttention(nn.Module):
    def __init__(self, in_dim, device):
        super().__init__()
        # Ensure at least 1 output channel for query and key
        reduced_dim = max(1, in_dim // 8)

        self.query = nn.Conv2d(in_dim, reduced_dim, 1)
        self.key = nn.Conv2d(in_dim, reduced_dim, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self._initialize_weights()
        self.to(device=device)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        proj_key = self.key(x).view(B, -1, H * W)                        # (B, C', HW)
        energy = torch.bmm(proj_query, proj_key)                        # (B, HW, HW)
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, -1, H * W)                   # (B, C, HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))        # (B, C, HW)
        out = out.view(B, C, H, W)

        return self.gamma * out + x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
         # Ensure at least 1 output channel for query and key
        reduced_dim = max(1, in_channels // reduction)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, in_channels, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Unet(nn.Module):
    def __init__(self, dataloader, patch_size=16, mask_ratio=0.75, enc=False):
        super(Unet, self).__init__()

        self.encoder = enc

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.dataloader = dataloader
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Self attention block
        self.attention = SelfAttention(64, self.device)

        # Channel Attention Block
        self.channel1 = ChannelAttention(3)
        self.channel2 = ChannelAttention(32)
        self.channel3 = ChannelAttention(64)
        
        # Encoder block (downsampling) 256x256x3
        self.encoder1 = self.conv_block(3, 32, 3) # 64x64x32
        self.encoder2 = self.conv_block(32, 64, 2) # 16x16x64
        self.dropout1 = nn.Dropout(p=0.5)  # Add this dropout layer
        
        # Bottleneck
        self.bottleneck = self.conv_block(64, 128, 2) # 4x4x128
        self.flatten = nn.Flatten()
        
        # Decoder block (upsampling)
        self.dropoutu = nn.Dropout(p=0.6)  # Add this dropout layer
        self.upscale4 = self.upscale_block(128, 64)
        self.decoder4 = self.conv_block(64, 64, 3, pool=False)
        self.upscale3 = self.upscale_block(64, 32)
        self.decoder3 = self.conv_block(32, 32, 3, pool=False)
        
        # Final output layer
        self.upscale0 = self.upscale_block(32, 3)
        self.final_conv = nn.Conv2d(3, 3, kernel_size=1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def conv_block(self, in_channels, out_channels, conv_layers=0, pool=True):
        conv = nn.Sequential()
        conv.add_module(f'conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        for i in range(conv_layers):
            conv.add_module(f'conv{i}', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        conv.add_module(f'relu', nn.ReLU(inplace=True))
        if pool:
            conv.add_module(f'pool',nn.MaxPool2d(kernel_size=4, stride=4))
        return conv
    
    # def upscale_block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4),
    #     )
    
    def upscale_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    
    def mask_input(self, x):
        """ Masks random patches of the input tensor (x), independently for each sample in the batch """
        batch_size, _, height, width = x.size()
        total_patches = (height * width) / (self.patch_size ** 2)
        num_patches_to_mask = int(total_patches * self.mask_ratio)

        mask = torch.ones_like(x)

        for i in range(batch_size):
            mask_positions = random.sample(range(int(total_patches)), num_patches_to_mask)
            for pos in mask_positions:
                row = pos // self.patch_size
                col = pos % self.patch_size
                h0, h1 = row * self.patch_size, (row + 1) * self.patch_size
                w0, w1 = col * self.patch_size, (col + 1) * self.patch_size
                mask[i, :, h0:h1, w0:w1] = 0

        # for i in range(batch_size):
        #     mask_positions = random.sample(range(int(total_patches)), num_patches_to_mask)
        #     for pos in range(int(total_patches)):
        #         row = pos // self.patch_size
        #         col = pos % self.patch_size
        #         h0, h1 = row * self.patch_size, (row + 1) * self.patch_size
        #         w0, w1 = col * self.patch_size, (col + 1) * self.patch_size
        #         sector = mask[i, :, h0:h1, w0:w1]

        #         sector_avg = sum(sum(sum(sector))) / (len(sector) * len(sector[0]) * len(sector[0][0]))
                
        #         if sector_avg == 1.0:
        #             mask[i, :, h0:h1, w0:w1] = 0.0


        masked_x = x * mask
        return masked_x, mask


    def forward(self, x, enc=False):
        #enc = enc if enc is not None else self.encoder # set mode as forward default behavior

        chn1 = self.channel1(x)
        # Encoder path (down-sampling)
        enc1 = self.encoder1(chn1)
        chn2 = self.channel2(enc1)
        enc2 = self.encoder2(chn2)
        chn3 = self.channel3(enc2)
        # attn = self.attention(chn3)

        
        # # Encoder path (down-sampling)
        # enc1 = self.encoder1(x)
        # enc2 = self.encoder2(enc1)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.dropout1(chn3))
        encoded = self.flatten(bottleneck)
        
        # Decoder path (up-sampling)
        
        up4  = self.upscale4(self.dropoutu(bottleneck))
        dec4 = self.decoder4(up4)
        up3  = self.upscale3(self.dropoutu(dec4))
        dec3 = self.decoder3(up3)
        
        # Final output layer
        up0  = self.upscale0(dec3)
        output = torch.sigmoid(self.final_conv(up0))
        
        if enc:
            return encoded

        return output
    
    def compute_loss(self, output, target, mask, factor=100.0):
        """Compute the loss only on the masked patches"""
        # Ensure the mask is in the same device as the output and target
        if mask is None:
            return factor*F.mse_loss(output, target)
            #return 100*self.foreground_weighted_mse(output, target)
        
        mask = mask.to(output.device)

        output_m = output * mask
        target_m = target * mask
        
        # # Calculate Mean Squared Error loss only on the masked patches
        # loss = F.mse_loss(output, target, reduction='mean')
        #ssim_loss = 1 - self.ssim(self.denormalize(output_m), self.denormalize(target_m))
        #ssim_loss = 1 - self.ssim(output_m, target_m)

        #mse_loss  = self.foreground_weighted_mse(output_m, target_m)
        mse_loss = F.mse_loss(output_m, target_m, reduction='mean')
        return factor*mse_loss
    
    def foreground_weighted_mse(self, pred, target):
        # Weight mask: 1 where molecule is, alpha elsewhere
        summed_target = torch.mean(target, dim=0)[0]
        weight = torch.ones_like(summed_target)
        weight[summed_target == 0.5] = 0  # adjust threshold for "non-black"
        weight = weight.unsqueeze(0).repeat(3, 1, 1)

        loss = weight * F.mse_loss(pred, target, reduction='mean')
        return loss.mean()
    
    def compute_ssim(self, pred, target):

        # pred = self.denormalize(pred)
        # target = self.denormalize(target)

        with torch.no_grad():
            return self.ssim(pred, target).cpu()
    
    def denormalize(self, tensor):
        """Denormalize a tensor image."""
        tensor = tensor.to(device=self.device) * 255
        i_tensor = torch.tensor(tensor, dtype=int)
        return i_tensor
    
    def train_unet(self, epochs=10, learning_rate=1e-3, patience=10, min_delta=0.001):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        scaler = GradScaler()  # Initialize GradScaler for mixed precision training
        writer = SummaryWriter()

        self.to(self.device)  # Move the model to GPU if available

        loss_history = []
        val_history = []
        #train_rmse_history = []
        #val_rmse_history = []

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        self.train()  # Set the model to training mode
        for epoch in tqdm(range(epochs), desc='Epochs', position=0):
            running_loss = 0.0
            #running_train_rmse = 0.0

            for data in tqdm(self.dataloader.train_loader, desc='Training', position=1, leave=False):
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
                #running_train_rmse += np.sqrt(loss.item())

                #running_smi  += self.compute_ssim(outputs, labels

            img_grid = vutils.make_grid(
                torch.cat([masked_input[:4], outputs[:4], labels[:4]]),  # Show 4 samples
                    nrow=4, normalize=True, scale_each=True
            )
            writer.add_image('Reconstructions - Train', img_grid, epoch)

            # Validation
            avg_val_loss = self.train_val(epoch_num=epoch+1, writer=writer)
            val_history.append(avg_val_loss)
            #val_rmse_history.append(avg_val_rmse)

            # Compute average losses and accuracy
            avg_train_loss = running_loss / len(self.dataloader.train_loader)
            #avg_train_rmse = running_train_rmse / len(self.dataloader.train_loader)

            loss_history.append(avg_train_loss)
            #train_rmse_history.append(avg_train_rmse)

            scheduler.step(avg_val_loss)

            # for name, param in self.named_parameters():
            #     writer.add_histogram(f'weights/{name}', param, epoch)
            #     writer.add_histogram(f'grads/{name}', param.grad, epoch)

            tqdm.write(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            tqdm.write(f"\tValidation Loss: {avg_val_loss:.4f}")

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

        print("Unet Training Complete!")

        writer.close()

        return loss_history, val_history

    
    def train_val(self, epoch_num, writer):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            #total_rmse = 0.0
            #total_smi  = 0.0
            #num_batches = len(self.dataloader.test_loader)

            # for imgs, labels in tqdm(self.dataloader.test_loader, desc='Validation', position=1, leave=False):
            #     imgs, labels = imgs.to(self.device), labels.to(self.device)
            #     # Use autocast for mixed precision during forward pass
            #     with autocast(device_type='cuda'):  # Forward pass in mixed precision
            #         outputs = self(imgs)
            #         loss = self.compute_loss(outputs, labels, mask=None)

            #     #rmse = np.sqrt(loss.item())

            #     total_loss += loss.item()
            #     #total_rmse += rmse

            imgs, labels = next(iter(self.dataloader.test_loader))
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with autocast (device_type='cuda'):
                outputs = self(imgs)
                loss = self.compute_loss(outputs, labels, mask=None)

            img_grid = vutils.make_grid(
                torch.cat([imgs[:4], outputs[:4], labels[:4]]),  # Show 4 samples
                    nrow=4, normalize=True, scale_each=True
            )
            writer.add_image('Reconstructions - Validation', img_grid, epoch_num)
        return loss.item() # Return averaged values
    
    def save_model(self, file_path="unet2_model.pth"):
        """ Save the model state dictionary to a file """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")


class Encoder(nn.Module):
    def __init__(self, full_unet: Unet):
        super().__init__()
        self.device = full_unet.device
        self.channel1 = full_unet.channel1
        self.channel2 = full_unet.channel2
        self.channel3 = full_unet.channel3
        self.encoder1 = full_unet.encoder1
        self.encoder2 = full_unet.encoder2
        self.dropout1 = full_unet.dropout1
        self.bottleneck = full_unet.bottleneck
        self.flatten = full_unet.flatten

    def forward(self, x):

        chn1 = self.channel1(x)
        enc1 = self.encoder1(chn1)
        chn2 = self.channel2(enc1)
        enc2 = self.encoder2(chn2)
        chn3 = self.channel3(enc2)

        bottleneck = self.bottleneck(self.dropout1(chn3))
        encoded = self.flatten(bottleneck)

        return encoded


def unet_main():

    seed_everything(42)

    data = EOS_Dataloader(mode='reconstruct', batch_size=32, num=50000)

    x = next(iter(data.train_loader))
    x = x[0]

    x_t = next(iter(data.test_loader))
    x_t = x_t[0]

    unet = Unet(data, patch_size=16, mask_ratio=0.75)

    x = x.to(unet.device)
    x_t = x_t.to(unet.device)

    unet.to(unet.device)

    loss_history, val_history = unet.train_unet(learning_rate=1e-4,epochs=100)
    unet.save_model("unet2_model.pth")

def load_unet():
    seed_everything(42)
    data = EOS_Dataloader(mode='reconstruct', batch_size=32, num=50000)
    full_net = Unet(data)
    full_net.load_state_dict(torch.load('unet2_model.pth', weights_only=True))
    full_net.eval()
    full_net.to(full_net.device)

    x = next(iter(data.train_loader))
    x = x[0]

    x_t = next(iter(data.test_loader))
    x_t = x_t[0]

    x = x.to(full_net.device)
    x_t = x_t.to(full_net.device)
    with torch.no_grad():
        x_enc = full_net(x)
        x_t_enc = full_net(x_t)

    x_enc_i = x_enc.detach().cpu().numpy()
    x_t_enc_i = x_t_enc.detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=2,ncols=2)

    ax[0][0].imshow(x[0].detach().cpu().numpy().transpose(1, 2, 0))
    ax[0][0].set_title('Train Original')
    ax[0][0].axis('off')

    ax[0][1].imshow(x_enc_i[0].transpose(1, 2, 0))
    ax[0][1].set_title('Train Reconstruction')
    ax[0][1].axis('off')

    ax[1][0].imshow(x_t[0].detach().cpu().numpy().transpose(1, 2, 0))
    ax[1][0].set_title('Test Original')
    ax[1][0].axis('off')

    ax[1][1].imshow(x_t_enc_i[0].transpose(1, 2, 0))
    ax[1][1].set_title('Test Reconstruction')
    ax[1][1].axis('off')

    plt.tight_layout()
    plt.show()

def load_encoder():
    seed_everything(42)
    data = EOS_Dataloader(mode='reconstruct', batch_size=32, num=50000)

    full_net = Unet(data)
    full_net.load_state_dict(torch.load('unet2_model.pth', weights_only=True))
    full_net.eval()

    encoder = Encoder(full_unet=full_net)
    encoder.eval()
    encoder.to(encoder.device)

    print(encoder)

    x = next(iter(data.train_loader))
    x = x[0]

    x_t = next(iter(data.test_loader))
    x_t = x_t[0]

    x = x.to(encoder.device)
    x_t = x_t.to(encoder.device)
    with torch.no_grad():
        x_enc = encoder(x).detach().cpu()
        x_t_enc = encoder(x_t).detach().cpu()

    x_enc_i = x_enc.detach().cpu().numpy()
    x_t_enc_i = x_t_enc.detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=2,ncols=1)

    ax[0].imshow(x_enc_i)
    ax[0].set_title('Train Encoding')
    ax[0].axis('off')

    ax[1].imshow(x_t_enc_i)
    ax[1].set_title('Test Encoding')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
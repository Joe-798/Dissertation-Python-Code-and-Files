#!/usr/bin/env python
# coding: utf-8

# In[66]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# In[67]:


# Define a helper class called DoubleConv which is a basic building block of U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
    
# Define the U-Net architecture, a common network used for image segmentation tasks    
class UNet(nn.Module):
    def __init__(self, original_channels=3, num_classes=4):
        super(UNet, self).__init__()
        self.original_channels = original_channels
        self.num_classes = num_classes

        self.encoder1 = DoubleConv(self.original_channels, 64, 64)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(64, 128, 128)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(128, 256, 256)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(256, 512, 512)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = DoubleConv(512, 1024, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(1024, 512, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(512, 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(128, 64, 64)

        self.decoder5 = nn.Conv2d(64, num_classes, kernel_size=1)

    # Define the forward pass for the U-Net
    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder1_pool = self.down1(encoder1)
        encoder2 = self.encoder2(encoder1_pool)
        encoder2_pool = self.down2(encoder2)
        encoder3 = self.encoder3(encoder2_pool)
        encoder3_pool = self.down3(encoder3)
        encoder4 = self.encoder4(encoder3_pool)
        encoder4_pool = self.down4(encoder4)
        encoder5 = self.encoder5(encoder4_pool)

        decoder1_up = self.up1(encoder5)
        decoder1 = self.decoder1(torch.cat((encoder4, decoder1_up), dim=1))

        decoder2_up = self.up2(decoder1)
        decoder2 = self.decoder2(torch.cat((encoder3, decoder2_up), dim=1))

        decoder3_up = self.up3(decoder2)
        decoder3 = self.decoder3(torch.cat((encoder2, decoder3_up), dim=1))

        decoder4_up = self.up4(decoder3)
        decoder4 = self.decoder4(torch.cat((encoder1, decoder4_up), dim=1))

        out = self.decoder5(decoder4)
        return out


# In[68]:


# Define a custom convolutional block class called ContinusParalleConv which is a basic building block of U-Net++
class ContinusParalleConv(nn.Module):
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conditional application of BatchNorm before Convolution or after, depending on the flag
        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.Conv_forward(x)
        return x

# Define the U-Net++ architecture with nested skip connections
class UnetPlusPlus(nn.Module):
    def __init__(self, original_channels=3, num_classes=4):
        super(UnetPlusPlus, self).__init__()
        self.original_channels = original_channels
        self.num_classes = num_classes
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(512*2, 512, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(256*3, 256, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(256*2, 256, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(128*2, 128, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(128*3, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(128*4, 128, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(64*2, 64, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(64*3, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(64*4, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(64*5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(self.original_channels, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        out_put4 = self.final_super_0_4(x_0_4)
        return out_put4


# In[69]:


# Custom dataset class for loading image and mask pairs from a directory
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.samples = []

        for subdir, _, _ in os.walk(root_dir):
            image_path = os.path.join(subdir, 'img.png')
            mask_path = os.path.join(subdir, 'label_mask.png')
            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.samples.append((image_path, mask_path))
        
        if augment:
            self.samples = self.samples * 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Apply transformations (including augmentations) if provided
        if self.augment:
            if idx < len(self.samples) // 2:
                transformed = self.transform[0](image=image, mask=mask)
            else:
                transformed = self.transform[1](image=image, mask=mask)
        else:
            transformed = self.transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']
        
        # Convert image and mask to the correct data types for PyTorch
        image = image.float()
        mask = mask.long() 
        
        return image, mask


# In[70]:


image_size = (512,512)

# Create two sets of transformations: basic and with augmentation
train_transform_1 = A.Compose([
    A.Resize(height=image_size[0], width=image_size[1]),
    ToTensorV2(),
 ])

train_transform_2 = A.Compose([
    A.Resize(height=image_size[0], width=image_size[1]),
    A.Rotate(limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.5),
    ToTensorV2(),
    ])

# Combine the two transformation sets
train_transform = (train_transform_1, train_transform_2)

val_transform = A.Compose([
    A.Resize(height=image_size[0], width=image_size[1]),
    ToTensorV2(),
 ])


# In[71]:


train_dir = "C:\\Users\\18229\\Desktop\\Sunderland_labels\\train"
val_dir = "C:\\Users\\18229\\Desktop\\Sunderland_labels\\val"
# Create training and validation datasets using the CustomDataset class
train_dataset = CustomDataset(root_dir=train_dir, transform=train_transform, augment=True)
val_dataset = CustomDataset(root_dir=val_dir, transform=val_transform, augment=False)

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# In[72]:


# Function to compute Intersection over Union (IoU) metric for segmentation
def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(union))
    return np.mean(ious)


# In[73]:


# Function to decode a segmentation mask into an RGB image for visualization
def decode_segmap(mask, num_classes):
    # Generate a color map for visualizing the segmentation mask
    label_colors = np.array([
        [0, 0, 0],        # 0 - black
        [255, 0, 0],      # 1 - red
        [0, 255, 0],      # 2 - green
        [0, 0, 255],      # 3 - blue
    ])
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in range(0, num_classes):
        rgb[mask == label] = label_colors[label]
    return rgb


# In[74]:


# Function to visualize U-Net predictions during validation
def visualize_unet_predictions(val_loader, epoch, model_path, num_images=3):
    model = UNet(original_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            if i < num_images:
                for j in range(images.size(0)):
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 3, 1)
                    img = images[j].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                    plt.imshow(img)
                    plt.title('Input Image')

                    plt.subplot(1, 3, 2)
                    decoded_mask = decode_segmap(masks[j].cpu().numpy(), num_classes=4)
                    plt.imshow(decoded_mask)
                    plt.title('Ground Truth')

                    plt.subplot(1, 3, 3)
                    decoded_preds = decode_segmap(preds[j].cpu().numpy(), num_classes=4)
                    plt.imshow(decoded_preds)
                    plt.title('Prediction')
                    
                    plt.show()
                    plt.savefig(f'U-net val_image_epoch_{epoch}_batch_{i+1}_image_{j+1}.png')
                    plt.close()


# In[75]:


# Function to visualize U-Net++ predictions during validation
def visualize_unetpp_predictions(val_loader, epoch, model_path, num_images=3):
    model = UnetPlusPlus(original_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            if i < num_images:
                for j in range(images.size(0)):
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 3, 1)
                    img = images[j].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                    plt.imshow(img)
                    plt.title('Input Image')

                    plt.subplot(1, 3, 2)
                    decoded_mask = decode_segmap(masks[j].cpu().numpy(), num_classes=4)
                    plt.imshow(decoded_mask)
                    plt.title('Ground Truth')

                    plt.subplot(1, 3, 3)
                    decoded_preds = decode_segmap(preds[j].cpu().numpy(), num_classes=4)
                    plt.imshow(decoded_preds)
                    plt.title('Prediction')
                    
                    plt.show()
                    plt.savefig(f'U-net++ val_image_epoch_{epoch}_batch_{i+1}_image_{j+1}.png')
                    plt.close()


# In[76]:


# U-Net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_unet = UNet(original_channels=3, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_unet = optim.Adam(model_unet.parameters(), lr=0.001)

num_epochs = 100
unet_train_losses = []
unet_train_ious = []
unet_val_losses = []
unet_val_ious = []

unet_best_val_iou = 0
unet_best_model_path = "C:\\Users\\18229\\Desktop\\best_unet_model.pth"
unet_best_epoch = 0

# Training loop for U-Net
for epoch in range(num_epochs):
    model_unet.train()
    unet_train_loss = 0
    unet_train_iou = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model_unet(images)
            loss = criterion(outputs, masks)

            optimizer_unet.zero_grad()
            loss.backward()
            optimizer_unet.step()

            unet_train_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            batch_iou = 0.0
            for i in range(preds.shape[0]):
                batch_iou += compute_iou(preds[i], masks[i], num_classes=4)
            batch_iou /= preds.shape[0]
            unet_train_iou += batch_iou * images.size(0)


            pbar.update(1)

    unet_train_loss = unet_train_loss / len(train_loader.dataset)
    unet_train_iou = unet_train_iou / len(train_loader.dataset)

    unet_train_losses.append(unet_train_loss)
    unet_train_ious.append(unet_train_iou)

    # Validation loop
    model_unet.eval()
    unet_val_loss = 0
    unet_val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model_unet(images)
            loss = criterion(outputs, masks)

            unet_val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            batch_iou = 0.0
            for i in range(preds.shape[0]):
                batch_iou += compute_iou(preds[i], masks[i], num_classes=4)
            batch_iou /= preds.shape[0]
            unet_val_iou += batch_iou * images.size(0)

    unet_val_loss = unet_val_loss / len(val_loader.dataset)
    unet_val_iou = unet_val_iou / len(val_loader.dataset)

    unet_val_losses.append(unet_val_loss)
    unet_val_ious.append(unet_val_iou)

    # Save the model if it achieves a better validation IoU
    if unet_val_iou > unet_best_val_iou:
        unet_best_val_iou = unet_val_iou
        unet_best_epoch = epoch + 1
        torch.save(model_unet.state_dict(), unet_best_model_path)

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'U-net Train Loss: {unet_train_loss:.4f}, U-net Train IoU: {unet_train_iou:.4f}, '
          f'U-net Val Loss: {unet_val_loss:.4f}, U-net Val IoU: {unet_val_iou:.4f}')

print('Training complete')


# In[77]:


#U-net plots
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 6))

# Plot U-net Train and Val Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, unet_train_losses, label='U-net Train Loss')
plt.plot(epochs, unet_val_losses, label='U-net Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('U-net Train and Val Loss')
plt.legend()

# Plot U-net Train and Val IoU
plt.subplot(1, 2, 2)
plt.plot(epochs, unet_train_ious, label='U-net Train IoU')
plt.plot(epochs, unet_val_ious, label='U-net Val IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.title('U-net Train and Val IoU')
plt.legend()

plt.tight_layout()
plt.show()


# In[78]:


visualize_unet_predictions(val_loader, unet_best_epoch, unet_best_model_path)


# In[79]:


# U-net++
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_unetpp = UnetPlusPlus(original_channels=3, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_unetpp = optim.Adam(model_unetpp.parameters(), lr=0.001)

num_epochs = 100
unetpp_train_losses = []
unetpp_train_ious = []
unetpp_val_losses = []
unetpp_val_ious = []

unetpp_best_val_iou = 0
unetpp_best_model_path = "C:\\Users\\18229\\Desktop\\best_unet++_model.pth"
unetpp_best_epoch = 0

# Training loop for U-Net++
for epoch in range(num_epochs):
    model_unetpp.train()
    unetpp_train_loss = 0
    unetpp_train_iou = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model_unetpp(images)
            loss = criterion(outputs, masks)

            optimizer_unetpp.zero_grad()
            loss.backward()
            optimizer_unetpp.step()

            unetpp_train_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            batch_iou = 0.0
            for i in range(preds.shape[0]):
                batch_iou += compute_iou(preds[i], masks[i], num_classes=4)
            batch_iou /= preds.shape[0]
            unetpp_train_iou += batch_iou * images.size(0)

            pbar.update(1)

    unetpp_train_loss = unetpp_train_loss / len(train_loader.dataset)
    unetpp_train_iou = unetpp_train_iou / len(train_loader.dataset)

    unetpp_train_losses.append(unetpp_train_loss)
    unetpp_train_ious.append(unetpp_train_iou)

    # Validation loop
    model_unetpp.eval()
    unetpp_val_loss = 0
    unetpp_val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model_unetpp(images)
            loss = criterion(outputs, masks)

            unetpp_val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            batch_iou = 0.0
            for i in range(preds.shape[0]):
                batch_iou += compute_iou(preds[i], masks[i], num_classes=4)
            batch_iou /= preds.shape[0]
            unetpp_val_iou += batch_iou * images.size(0)

    unetpp_val_loss = unetpp_val_loss / len(val_loader.dataset)
    unetpp_val_iou = unetpp_val_iou / len(val_loader.dataset)

    unetpp_val_losses.append(unetpp_val_loss)
    unetpp_val_ious.append(unetpp_val_iou)

    # Save the model if it achieves a better validation IoU
    if unetpp_val_iou > unetpp_best_val_iou:
        unetpp_best_val_iou = unetpp_val_iou
        unetpp_best_epoch = epoch + 1
        torch.save(model_unetpp.state_dict(), unetpp_best_model_path)

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'U-Net++ Train Loss: {unetpp_train_loss:.4f}, U-Net++ Train IoU: {unetpp_train_iou:.4f}, '
          f'U-Net++ Val Loss: {unetpp_val_loss:.4f}, U-Net++ Val IoU: {unetpp_val_iou:.4f}')

print('Training complete')


# In[80]:


#U-net++ plots
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 6))

# Plot U-net++ Train and Val Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, unetpp_train_losses, label='U-net++ Train Loss')
plt.plot(epochs, unetpp_val_losses, label='U-net++ Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('U-net++ Train and Val Loss')
plt.legend()

# Plot U-net++ Train and Val IoU
plt.subplot(1, 2, 2)
plt.plot(epochs, unetpp_train_ious, label='U-net++ Train IoU')
plt.plot(epochs, unetpp_val_ious, label='U-net++ Val IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.title('U-net++ Train and Val IoU')
plt.legend()

plt.tight_layout()
plt.show()


# In[81]:


visualize_unetpp_predictions(val_loader, unetpp_best_epoch, unetpp_best_model_path)


# In[ ]:





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import os
import numpy as np

# --- 1. Dataset Class ---
class HC_SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".png", "_Annotation.png")
        
        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            return torch.zeros((1, 128, 128)), torch.zeros((1, 128, 128))

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(img)
            mask = cv2.resize(mask, (128, 128))
            mask = torch.tensor(mask).unsqueeze(0)
            
        return img, mask

# --- 2. Metrics & Hybrid Loss ---
def calculate_dice(preds, targets):
    smooth = 1e-6
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice_loss

# --- 3. Simple U-Net Model ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_c),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(1, 32)
        self.enc2 = conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# --- 4. Training Function ---
def train():
    device = torch.device("cpu")
    IMG_DIR = r"C:\Priyashree_Panda_Research\Dataset\Part_B\images"
    MASK_DIR = r"C:\Priyashree_Panda_Research\Dataset\Part_B\masks"
    SAVE_PATH = r"C:\Priyashree_Panda_Research\Task2_segmentation\ModelWeights\unet_fetal_head.pth"

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = HC_SegmentationDataset(IMG_DIR, MASK_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleUNet().to(device)

    # --- LOAD PREVIOUS WEIGHTS FOR ROUND 2 ---
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH))
        print("--- Previous Weights Loaded! Starting Fine-Tuning ---")

    criterion = DiceBCELoss()
    # LOWER LEARNING RATE FOR REFINEMENT
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    print("Starting Fine-Tuning (Round 2)...")
    
    for epoch in range(15): # 15 epochs for deeper learning
        epoch_loss = 0
        epoch_dice = 0
        model.train()
        
        for imgs, masks in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_dice += calculate_dice(torch.sigmoid(outputs), masks).item()
        
        print(f"Epoch {epoch+1}/15 | Loss: {epoch_loss/len(loader):.4f} | Dice: {epoch_dice/len(loader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nSaved Fine-Tuned Model to: {SAVE_PATH}")

if __name__ == "__main__":
    train()
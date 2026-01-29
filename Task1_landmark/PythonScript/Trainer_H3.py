import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import pandas as pd
import os

# --- 1. Data Loader ---
class PartA_FinalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Return blank data if image is missing to prevent crashing
        if image is None: 
            return torch.zeros((1, 128, 128)), torch.zeros(8)
        
        h, w = image.shape
        landmarks = self.df.iloc[idx, 1:9].values.astype('float32')
        
        # Normalization (Crucial for ResNet accuracy)
        landmarks[0::2] /= w
        landmarks[1::2] /= h

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(landmarks)

# --- 2. Deeper Model (ResNet34) ---
class FinalLandmarkDetector(nn.Module):
    def __init__(self):
        super(FinalLandmarkDetector, self).__init__()
        # Hypothesis 3 uses a 34-layer residual network
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# --- 3. Main Training Function ---
def train_final():
    CSV_PATH = r"C:\Priyashree_Panda_Research\Dataset\Part_A\role_challenge_dataset_ground_truth.csv"
    IMG_DIR = r"C:\Priyashree_Panda_Research\Dataset\Part_A\images-20251230T170935Z-1-001\images"
    SAVE_PATH = r"C:\Priyashree_Panda_Research\Task1_landmark\ModelWeights\final_model_h3.pth"

    # Optimized for CPU speed
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = PartA_FinalDataset(CSV_PATH, IMG_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cpu") # Force CPU for stability on laptop
    model = FinalLandmarkDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Starting FAST Final Training (ResNet34) on {device}...")
    model.train()
    
    # 8 Epochs is enough since we are using Transfer Learning
    for epoch in range(8): 
        loss_total = 0.0
        for imgs, targs in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targs)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        
        print(f"Epoch {epoch+1}/8 | Avg Loss: {loss_total/len(loader):.6f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"SUCCESS: Final Model H3 saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_final()
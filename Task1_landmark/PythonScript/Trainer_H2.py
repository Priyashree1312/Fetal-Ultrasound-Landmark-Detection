import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import pandas as pd
import os

# --- 1. Data Loader Class (Updated with Normalization) ---
class PartA_LandmarkDataset(Dataset):
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
        
        if image is None:
            return torch.zeros((1, 128, 128)), torch.zeros(8)
        
        # Get original height and width
        h, w = image.shape
        landmarks = self.df.iloc[idx, 1:9].values.astype('float32')

        # --- NORMALIZATION STEP ---
        # Scale X coordinates by width, Y coordinates by height
        landmarks[0::2] /= w  # x1, x2, x3, x4
        landmarks[1::2] /= h  # y1, y2, y3, y4

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(landmarks)

# --- 2. Model Architecture ---
class LandmarkDetector(nn.Module):
    def __init__(self):
        super(LandmarkDetector, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        # We add a Sigmoid layer to force the model to output values between 0 and 1
        return torch.sigmoid(self.model(x))

# --- 3. Main Training Function ---
def train():
    CSV_PATH = r"C:\Priyashree_Panda_Research\Dataset\Part_A\role_challenge_dataset_ground_truth.csv"
    IMG_DIR = r"C:\Priyashree_Panda_Research\Dataset\Part_A\images-20251230T170935Z-1-001\images"
    WEIGHTS_SAVE_PATH = r"C:\Priyashree_Panda_Research\Task1_landmark\ModelWeights\hypothesis_2_model.pth"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = PartA_LandmarkDataset(CSV_PATH, IMG_DIR, transform=transform)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 10 # Increased to 10 for better accuracy with normalization
    print(f"Starting Normalized Training for H2 on {device}...")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    os.makedirs(os.path.dirname(WEIGHTS_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
    print(f"H2 Normalized Training Complete! Saved to: {WEIGHTS_SAVE_PATH}")

if __name__ == "__main__":
    train()
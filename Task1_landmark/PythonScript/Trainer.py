import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import pandas as pd
import os

# --- 1. Data Loader Class ---
class PartA_LandmarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image name from the first column
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image in Grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}. Please check the folder path.")
        
        # Get 8 coordinate values (x1, y1, x2, y2, x3, y3, x4, y4)
        landmarks = self.df.iloc[idx, 1:9].values.astype('float32')
        
        # Logic: If you want to normalize landmarks relative to image size (optional but recommended)
        # h, w = image.shape
        # landmarks[0::2] /= w  # x coordinates
        # landmarks[1::2] /= h  # y coordinates

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(landmarks)

# --- 2. Model Architecture (Hypothesis 1: ResNet18) ---
class LandmarkDetector(nn.Module):
    def __init__(self):
        super(LandmarkDetector, self).__init__()
        # Load a pre-trained ResNet18 as the backbone
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Change the first convolution layer to accept 1 channel (Grayscale)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Change the final fully connected layer to output 8 values
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return self.model(x)

# --- 3. Main Training Function ---
def train():
    # --- UPDATED PATHS WITH THE CORRECT NESTED FOLDER ---
    CSV_PATH = r"C:\Priyashree_Panda_Research\Dataset\Part_A\role_challenge_dataset_ground_truth.csv"
    # Added \images at the end here
    IMG_DIR = r"C:\Priyashree_Panda_Research\Dataset\Part_A\images-20251230T170935Z-1-001\images"
    WEIGHTS_SAVE_PATH = r"C:\Priyashree_Panda_Research\Task_1_Landmark\ModelWeights\hypothesis_1_model.pth"

    # Pre-processing Pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # Standard input size for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Initialize Dataset and DataLoader
    try:
        dataset = PartA_LandmarkDataset(CSV_PATH, IMG_DIR, transform=transform)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"Success: Dataset loaded with {len(dataset)} images.")
    except Exception as e:
        print(f"Configuration Error: {e}")
        return

    # Device configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkDetector().to(device)
    
    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Best for coordinate regression

    # Training Loop
    epochs = 10
    print(f"Starting training on device: {device}")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(WEIGHTS_SAVE_PATH), exist_ok=True)
    
    # Save the trained model weights
    torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
    print(f"Training Complete. Weights saved to: {WEIGHTS_SAVE_PATH}")

if __name__ == "__main__":
    train()
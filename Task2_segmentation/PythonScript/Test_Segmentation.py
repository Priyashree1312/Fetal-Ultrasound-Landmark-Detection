import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. Re-define the EXACT Model Structure ---
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

# --- 2. Inference & Visualization Function ---
def test_and_visualize():
    # Paths - Double check these match your computer!
    MODEL_PATH = r"C:\Priyashree_Panda_Research\Task2_segmentation\ModelWeights\unet_fetal_head.pth"
    TEST_IMAGE = r"C:\Priyashree_Panda_Research\Dataset\Part_B\images\005_HC.png"
    
    device = torch.device("cpu")
    
    # Load the trained weights
    model = SimpleUNet().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights file not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model Loaded Successfully!")

    # Pre-process the Image exactly like Training
    raw_img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print("Error: Image not found!")
        return
        
    h_orig, w_orig = raw_img.shape
    img_resized = cv2.resize(raw_img, (128, 128))
    
    # Normalize: Convert to tensor and apply (img - 0.5) / 0.5
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        # Apply sigmoid and threshold
        pred_mask = torch.sigmoid(output) > 0.5 
        pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

    # Resize mask back to original ultrasound size
    final_mask = cv2.resize(pred_mask, (w_orig, h_orig))

    # --- 3. Calculate Clinical Measurement (HC) ---
    # Standard pixel-to-mm ratio for this dataset is roughly 0.18
    pixel_size = 0.18 
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hc_mm = 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        hc_mm = perimeter * pixel_size

    # Create Visualization Overlay
    color_result = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    color_result[final_mask == 1] = [0, 255, 0] # Green for AI Prediction

    # Display Results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Ultrasound")
    plt.imshow(raw_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"AI Segmentation Overlay\nPredicted HC: {hc_mm:.2f} mm")
    plt.imshow(cv2.cvtColor(color_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    print(f"Final Prediction Complete! Calculated HC: {hc_mm:.2f} mm")
    plt.show()

if __name__ == "__main__":
    test_and_visualize()
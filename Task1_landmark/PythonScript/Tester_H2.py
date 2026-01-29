import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

# 1. Architecture must match Trainer_H2 (ResNet18 + Sigmoid)
class LandmarkDetector(nn.Module):
    def __init__(self):
        super(LandmarkDetector, self).__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return torch.sigmoid(self.model(x)) # Matches the new Trainer

# 2. Paths
WEIGHTS_PATH = r"C:\Priyashree_Panda_Research\Task1_landmark\ModelWeights\hypothesis_2_model.pth"
TEST_IMAGE = r"C:\Priyashree_Panda_Research\Dataset\Part_A\images-20251230T170935Z-1-001\images\489_HC.png"

# 3. Load Model
device = torch.device("cpu")
model = LandmarkDetector()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# 4. Prepare Image
image_raw = cv2.imread(TEST_IMAGE)
gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
orig_h, orig_w = gray.shape

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
img_tensor = transform(gray).unsqueeze(0)

# 5. Predict and Scale back to Pixels
with torch.no_grad():
    preds = model(img_tensor).numpy()[0]

# Draw the 4 landmarks
for i in range(0, 8, 2):
    # Scale from 0-1 range back to original image size
    x = int(preds[i] * orig_w)
    y = int(preds[i+1] * orig_h)
    cv2.circle(image_raw, (x, y), 10, (0, 255, 0), -1)

# 6. Show Result
cv2.imshow("Hypothesis 2 - SUCCESSFUL NORMALIZATION", image_raw)
cv2.waitKey(0)
cv2.destroyAllWindows()
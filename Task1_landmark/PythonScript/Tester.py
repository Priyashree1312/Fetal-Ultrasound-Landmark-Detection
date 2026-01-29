import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

# 1. Re-create the EXACT same model structure as Trainer.py
class LandmarkDetector(nn.Module):
    def __init__(self):
        super(LandmarkDetector, self).__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)
    def forward(self, x): return self.model(x)

# 2. Set your paths
WEIGHTS_PATH = r"C:\Priyashree_Panda_Research\Task_1_Landmark\ModelWeights\hypothesis_1_model.pth"
TEST_IMAGE = r"C:\Priyashree_Panda_Research\Dataset\Part_A\images-20251230T170935Z-1-001\images\489_HC.png"

# 3. Load the Weights
device = torch.device("cpu")
model = LandmarkDetector()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval() # Put model in evaluation mode

# 4. Prepare the image
image_raw = cv2.imread(TEST_IMAGE)
gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
orig_h, orig_w = gray.shape

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
img_tensor = transform(gray).unsqueeze(0)

# 5. Predict and Draw
with torch.no_grad():
    preds = model(img_tensor).numpy()[0]

# Draw the 4 landmarks (Green dots)
for i in range(0, 8, 2):
    # If you didn't normalize landmarks in training, use preds directly:
    x, y = int(preds[i]), int(preds[i+1])
    cv2.circle(image_raw, (x, y), 8, (0, 255, 0), -1)

# 6. Show the result
cv2.imshow("Fetal Landmark Prediction", image_raw)
cv2.waitKey(0)
cv2.destroyAllWindows()
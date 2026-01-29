import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

class FinalLandmarkDetector(nn.Module):
    def __init__(self):
        super(FinalLandmarkDetector, self).__init__()
        self.model = models.resnet34()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)
    def forward(self, x):
        return torch.sigmoid(self.model(x))

WEIGHTS_PATH = r"C:\Priyashree_Panda_Research\Task1_landmark\ModelWeights\final_model_h3.pth"
TEST_IMAGE = r"C:\Priyashree_Panda_Research\Dataset\Part_A\images-20251230T170935Z-1-001\images\489_HC.png"

model = FinalLandmarkDetector()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
model.eval()

image_raw = cv2.imread(TEST_IMAGE)
gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# MUST match 128 size from the Trainer
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
img_tensor = transform(gray).unsqueeze(0)

with torch.no_grad():
    preds = model(img_tensor).numpy()[0]

# Normalization means we multiply by original image width/height
for i in range(0, 8, 2):
    x, y = int(preds[i] * w), int(preds[i+1] * h)
    cv2.circle(image_raw, (x, y), 8, (255, 255, 0), -1) # Cyan Dots

cv2.imshow("Final Result - Hypothesis 3", image_raw)
cv2.waitKey(0)
cv2.destroyAllWindows()
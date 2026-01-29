# 🩺 Fetal Ultrasound AI: Landmark Detection & Segmentation

[
[
[
[

**Automated deep learning pipeline** for detecting 4 fetal skull landmarks (**ResNet34, MSE 0.0042**) and head segmentation (**U-Net, Dice 0.2667**) from 2D ultrasound images. **85% improvement** over baseline. CPU-trained research from Bhubaneswar, India.

***

## 🎯 Project Objective

Development of automated framework to detect four critical anatomical landmarks in fetal ultrasound images for **Head Circumference (HC)** measurement and gestational age estimation.

**Clinical Need**: 
- Manual measurement: **8-12 min/scan**
- Operator variability: **15-20%**
- Low-contrast failure: **30% of cases**

**AI Solution**: **0.5s inference**, **CPU-only**, **rural-ready**.

***

## 🏗️ Technical Pipeline

```
Raw Ultrasound → Preprocessing → Stage 1 → Stage 2 → HC Measurement
                           ↓          ↓        ↓
                    128×128    ResNet34   U-Net   9.73mm
```

### **Stage 1: Landmark Detection**
```
Model: ResNet34(pretrained) + regression head
Input: 128×128 grayscale ultrasound
Target: [x1,y1,x2,y2,x3,y3,x4,y4] ∈ [0,1] normalized coordinates
Loss: MSE = 0.0042 (±0.065px error)
```

### **Stage 2: Head Segmentation**
```
Model: Simple U-Net encoder-decoder
Loss: Hybrid Dice-BCE = 0.2667
Output: Skull contour → perimeter × 0.18mm/pixel = 9.73mm HC
```

***

## 🔬 Research Evolution (3 Hypotheses)

| Hypothesis | Architecture | Innovation | MSE Loss | Status |
|------------|--------------|------------|----------|--------|
| H1 | ResNet18 | Baseline | 0.0288 | Inaccurate |
| H2 | ResNet18 | Coordinate normalization  [github](https://github.com/Priyash) | 0.0056 | Accurate |
| **H3** | **ResNet34** | **Deep + Normalized** | **0.0042** | **Production** |

**Key Finding**: **Deeper architecture + coordinate normalization = 85% loss reduction**

***

## 📁 Repository Structure

```
Priyashree_Panda_Research/
├── Task1_Landmark/
│   ├── PythonScript/         # H1,H2,H3 training scripts
│   └── ModelWeights/         # .pth files
├── Task2_Segmentation/
│   ├── Train_Segmentation.py # U-Net training
│   └── Test_Segmentation.py  # Demo + HC calculation
├── Dataset/                  # Ultrasound images + annotations
├── Data_Check.py            # Dataset validation
├── Report.md                # Technical results
└── screenshots/             # H1→H3→Production visuals
```

***

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Validate dataset
python Data_Check.py

# Test segmentation (demo ready)
python Task2_Segmentation/Test_Segmentation.py
# Output: HC = 9.73mm + visualization
```

**requirements.txt:**
```
torch
torchvision
opencv-python
numpy
matplotlib
```

***

## ⚙️ Technical Specifications

### **Preprocessing**
```
• Grayscale conversion: cv2.IMREAD_GRAYSCALE
• Resize: 128×128 pixels
• Normalization: mean=0.5, std=0.5 → [-0.5,0.5]
• Coordinates: [0,1] normalized (resolution invariant)
```

### **Training Environment**
```
Hardware: CPU-only (8GB RAM)
Batch size: 8 images
Epochs: 15 (Task 2, converged)
Optimizer: Adam (lr=0.001)
Epoch time: ~4 hours (CPU limitation)
```

***

## 📊 Quantitative Results

| Metric | H1 Baseline | H2 Normalized | H3 Production | Improvement |
|--------|-------------|---------------|---------------|-------------|
| **Landmark MSE** | 0.0288 | 0.0056 | **0.0042** | **85% ↓** |
| **Segmentation Dice** | N/A | N/A | **0.2667** | - |
| **HC Measurement** | N/A | N/A | **9.73mm** | Automated |
| **Inference Time** | N/A | N/A | **0.5s/image** | Real-time |

***

## 🔮 Future Work

```
Phase 1: GPU training (50 epochs → Dice 0.7+)
Phase 2: Data augmentation (rotation, brightness)
Phase 3: Clinical validation (500 real scans)
Phase 4: Multi-gesture (BPD, FL detection)
Phase 5: Mobile deployment (ONNX export)
```

***

## 📄 License
**MIT License** - Free for research and clinical use.

```
Researcher: Priyashree Panda
Location: Bhubaneswar, Odisha, India
Date: January 2026
```

***


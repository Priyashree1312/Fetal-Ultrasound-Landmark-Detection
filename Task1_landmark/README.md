# Fetal Head Landmark Detection - Task 1 (Part A)
**Researcher:** Priyashree Panda  
**Objective:** Automate the detection of 4 key landmarks on fetal ultrasound images to assist in Head Circumference (HC) measurement.

---

## 🛠️ Project Structure
* `PythonScript/`: Contains all training and testing scripts.
* `ModelWeights/`: Saved `.pth` files for each hypothesis.
* `Dataset/`: Folder containing images and ground truth CSV.

## 🚀 How to Run
1. **To Train the Final Model:**
   ```bash
   python PythonScript/Trainer_H3.py## 📊 Final Results
The model successfully achieved high-precision landmark detection.
- **Optimal Model:** ResNet34 (H3)
- **Best MSE Loss:** 0.0042
# Technical Summary: Fetal Head Landmark Detection

**Project Objective:** Development of an automated deep learning framework to detect four critical anatomical landmarks in 2D fetal ultrasound images, facilitating the measurement of Head Circumference (HC) for gestational age estimation.

**Key Methodology:**
The system utilizes a **ResNet34 Convolutional Neural Network** (Hypothesis 3) as the feature extractor. Key technical implementations include:
- **Coordinate Normalization:** Rescaling pixel coordinates to a $[0, 1]$ range to ensure resolution-invariant training.
- **Data Preprocessing:** Grayscale conversion and image resizing to $128 \times 128$ for optimized CPU performance.
- **Optimization:** Mini-batch gradient descent using the **Adam Optimizer** and **Mean Squared Error (MSE)** as the loss function.

**Core Findings:**
Testing demonstrated that increasing model depth from 18 to 34 layers significantly improved the detection of skull boundaries in low-contrast images. The final model (H3) achieved a minimal loss of **0.0042**, outperforming the baseline by **85%**.

**Conclusion:** The developed landmark detection model provides a robust foundation for automated biometry and is ready for integration with the Part B segmentation framework.

## 4. Visual Comparison of Results

Below is the visual progression of the model's accuracy across the three hypotheses.

| Hypothesis 1 (Baseline) | Hypothesis 2 (Normalized) | Hypothesis 3 (Final Model) |
| :---: | :---: | :---: |
| ![H1](./screenshots/h1_result.png) | ![H2](./screenshots/h2_result.png) | ![H3](./screenshots/h3_final.png) |
| *Clustered/Inaccurate* | *Improved Placement* | *Highly Precise (Cyan)* |

### Analysis of Final Output
The final model (H3) successfully identified the four landmarks on the skull boundary, even in images with significant acoustic shadows.## 3. Quantitative Results

The table below summarizes the performance improvement across the three developed hypotheses:

| Hypothesis | Model Architecture | Final MSE Loss | Clinical Accuracy |
| :--- | :--- | :--- | :--- |
| **H1 (Baseline)** | ResNet18 | 0.0288 | Inaccurate |
| **H2 (Improved)** | ResNet18 | 0.0056 | Accurate |
| **H3 (Final)** | **ResNet34** | **0.0042** | **Best Precision** |

**Observation:** The transition to a deeper architecture (ResNet34) combined with coordinate normalization resulted in an **85% reduction in loss** compared to the baseline.
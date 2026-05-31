# Ensemble Multi-Class Classification on Optical Handwritten Digits

An end-to-end computer vision and machine learning pipeline developed during my **Data Science Internship** at **DecodeLabs**. This project implements a high-capacity Random Forest Classifier to identify and classify handwritten digits (0–9) using pixel-intensity matrix data.

## 📋 Task Overview
The goal of this project is to transition from linear frameworks to non-linear ensemble methods while working with structural image data. The primary milestones include:
* **Image Data Vectorization:** Processing a classic dataset of low-resolution pixel grids flattened into 1-dimensional feature vectors.
* **Ensemble Optimization:** Instantiating a 1,000-tree Random Forest architecture to handle multi-class boundaries.
* **Diagnostic Visualization:** Plotting multi-class prediction grids and structural confusion heatmaps to map model failure points.

---

## 🗂️ Dataset Glossary & Features
The project utilizes the **Scikit-Learn Optical Recognition of Handwritten Digits Dataset** (inspired by the classic NIST database).
* **Dimensionality:** 1,797 total samples representing handwritten numbers.
* **Features:** Each sample is an 8×8 image patch scaled down to a 64-element matrix of gray-scale pixel integers ranging from 0 (white) to 16 (black).
* **Target Classes:** Distinct structural integers ranging from `0` to `9`.

---

## ⚙️ Project Pipeline & Preprocessing

### 1. Data Splitting & Vectorization
Because Random Forests do not implicitly require localized spatial convolutional patterns like deep neural architectures, the 8×8 pixel matrices were processed using their pre-flattened 64-dimensional feature shapes (`digits.data`). The data was split into a standard training (75%) and testing (25%) partition.

### 2. Ensemble Modeling Strategy
To eliminate high variance and individual tree overfitting on specific pixel arrangements, a `RandomForestClassifier` was built with **1,000 estimators**. This creates an expansive consensus voting group across randomly sub-sampled features, ensuring exceptional generalizability without requiring explicit min-max feature scaling.

---

## 📈 Visual Matrix Analysis & Grid Validation

The initial cells verify input data integrity by rendering a 64-sample layout of the raw 8×8 pixel arrays alongside their verified ground-truth target markers:

```python
# Grid layout generation for baseline verification
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
```

*(Note: The visual grid and the corresponding inverted 10×10 Seaborn prediction heatmap mapping true vs. predicted classifications are located within the `visualizations/` repository directory).*

---

## 📊 Model Evaluation Results

The baseline ensemble framework achieved an exceptional overall classification capability across 450 unseen test images:

### 1. Model Accuracy
* **Total Multi-Class Prediction Accuracy Score:** `98.00%`

### 2. Structural Classification Report
Below is the definitive performance breakdown mapping precision and recall for each distinct digit target:


| Digit Target | Precision | Recall | F1-Score | Support (Samples) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.97 | 1.00 | 0.99 | 37 |
| **1** | 0.96 | 1.00 | 0.98 | 43 |
| **2** | 1.00 | 0.95 | 0.98 | 44 |
| **3** | 0.98 | 0.98 | 0.98 | 45 |
| **4** | 1.00 | 0.97 | 0.99 | 38 |
| **5** | 0.98 | 0.98 | 0.98 | 48 |
| **6** | 1.00 | 1.00 | 1.00 | 52 |
| **7** | 0.96 | 1.00 | 0.98 | 48 |
| **8** | 0.98 | 0.94 | 0.96 | 48 |
| **9** | 0.98 | 0.98 | 0.98 | 47 |
| **Overall Accuracy** | | | **0.98** | **450** |
| *Macro Average* | 0.98 | 0.98 | 0.98 | 450 |
| *Weighted Average* | 0.98 | 0.98 | 0.98 | 450 |

### 3. Confusion Matrix Derived Insights
* **Perfect Classification:** The digit **6** achieved an absolute 1.00 across all evaluation metrics, showcasing entirely distinct spatial signatures.
* **Minor Confusions:** Minor variance drops occurred within digits **2** and **8** (Recall dropping to 0.95 and 0.94, respectively). This mathematically reveals that the model occasionally struggles with the looping intersections common between curved handwritten characters.
<p align="center">
  <img width="428" height="428" alt="image" src="https://github.com/user-attachments/assets/5dd130d3-809c-4db3-a709-cb345a141fae" />
</p>

---

## 🚀 Future Enhancements
To scale this baseline pipeline into production-grade systems, upcoming repository revisions will focus on:
1. **Hyperparameter Cost Reduction:** Using `RandomizedSearchCV` to evaluate if the estimator count can be pruned below 1,000 without sacrificing the 98% accuracy baseline.
2. **Dimension Pruning:** Executing Principal Component Analysis (PCA) to map the 64 pixels into lower orthogonal dimensions to boost fitting speeds.
3. **Advanced Neural Architectures:** Introducing a basic Multi-Layer Perceptron (MLP) or Convolutional Neural Network (CNN) to capture spatial pixel structures natively.

---

## 🤝 Acknowledgments
* **DecodeLabs** for providing the structured professional data science internship scope.
* **O'Reilly Media** for providing foundational educational patterns via the *Python Data Science Handbook*.

# 🚀 Neural Architecture Search for CIFAR-10 Classification Using Optuna


<!-- Badges -->
![Validation Accuracy](https://img.shields.io/badge/Validation_Accuracy-72.62%25-brightgreen)
![Best Trial](https://img.shields.io/badge/Best_Trial-3_Layers-blue)
![Epochs](https://img.shields.io/badge/Epochs-30-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.1-purple)


---

## 📌 Executive Summary
This project implements **Neural Architecture Search (NAS)** to automatically design **Convolutional Neural Networks (CNNs)** for the **CIFAR-10 dataset**, a benchmark dataset in image classification.  

**Key Highlights:**  
- Implementation in **PyTorch** with **Optuna** for hyperparameter optimization.  
- **Hyperband pruning** reduces computational cost by early termination of underperforming trials.  
- Optimal CNN architecture discovered:
  - **3 convolutional layers**, each with **64 channels**  
  - Kernel sizes: **5, 3, 5**  
  - Learning rate: **0.00113**  
- Achieved **72.62% validation accuracy** during NAS search.  
- Fully trained optimal model for **30 epochs** with consistent convergence.  

This demonstrates a **systematic approach** to designing efficient CNN architectures with minimal manual effort.

---

## 📑 Table of Contents
1. [Introduction](#introduction)  
2. [Methodology](#methodology)  
   - [Data Preparation](#data-preparation)  
   - [Neural Architecture Search Setup](#neural-architecture-search-setup)  
   - [Model Training and Evaluation](#model-training-and-evaluation)  
3. [Results](#results)  
   - [Dataset Analysis](#dataset-analysis)  
   - [NAS Optimization Results](#nas-optimization-results)  
   - [Best Model Training](#best-model-training)  
   - [Final Evaluation](#final-evaluation)  
4. [Discussion](#discussion)  
5. [Conclusion](#conclusion)  
6. [Recommendations](#recommendations)  

---

## 1️⃣ Introduction
Image classification is a **fundamental task in computer vision**, with applications in autonomous driving, medical imaging, and security systems.  

The **CIFAR-10 dataset** consists of **60,000 color images (32×32)** across **10 classes**. Traditional CNN design requires **manual tuning of hyperparameters**, which is time-consuming and suboptimal.  

This project uses **Neural Architecture Search (NAS)** to **automate CNN architecture discovery**, selecting the optimal combination of layers, channels, kernels, and learning rates to maximize validation performance.  

**Objectives:**  
- ✅ Load and preprocess CIFAR-10 dataset.  
- ✅ Define a **search space for CNN architectures** and optimize it using Optuna.  
- ✅ Fully train the best-identified CNN model.  
- ✅ Evaluate performance using **accuracy, precision, recall, F1-score**, and **confusion matrix**.  
- ✅ Visualize NAS results and training progress.

---

## 2️⃣ Methodology

### 🗂 Data Preparation
- Downloaded CIFAR-10 via `torchvision.datasets` and normalized:

\[
X_\text{norm} = \frac{X - \mu}{\sigma}, \quad \mu = 0.5, \sigma = 0.5
\]

- Dataset splits:  
  - **Training:** 45,000 images  
  - **Validation:** 5,000 images  
  - **Test:** 10,000 images  
- Batch size = 128  
- Class distributions visualized to ensure balance.

![Class Distribution](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/class%20distridution%20in%20cifar-10.png)

*Figure 1: Class distribution in dataset.*

![Train Class Distribution](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/training%20set.png)

*Figure 2: Training set distribution.*

![Validation Class Distribution](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/validation%20set.png)

*Figure 3: Validation set distribution.*

---

### 🔍 Neural Architecture Search Setup
**Search Space Definition:**  
- **Conv layers:** 2–4  
- **Channels per layer:** {16, 32, 64}  
- **Kernel sizes per layer:** {3, 5}  
- Each layer: **Conv → ReLU → MaxPool(2×2)**  
- **Classifier:** Fully connected layer with 10 outputs  

**Objective Function:**  
- Optimizer: Adam  
- Loss function: **CrossEntropyLoss**  

\[
L = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
\]

Where:  
- \(N\) = batch size  
- \(C\) = number of classes (10)  
- \(y_{i,c}\) = true label  
- \(\hat{y}_{i,c}\) = predicted probability  

- **NAS Strategy:** Optuna + Hyperband pruning  
- **Trials:** 10, each trained for 5 epochs  

---

### 🏋️ Model Training and Evaluation
- Best architecture retrained on **full training+validation set** for **30 epochs**  
- Metrics monitored:  
  - Training & validation **loss**  
  - Training & validation **accuracy**  
  - Multi-class **AUC**  

- Evaluation metrics:  
  - **Accuracy:** \(\text{Acc} = \frac{TP + TN}{TP+TN+FP+FN}\)  
  - **Precision:** \(P = \frac{TP}{TP+FP}\)  
  - **Recall:** \(R = \frac{TP}{TP+FN}\)  
  - **F1-score:** \(F1 = \frac{2PR}{P+R}\)  

---

## 3️⃣ Results

### 📊 Dataset Analysis
- Dataset balanced across all classes.  
- Sample 5×5 grid demonstrates **class diversity**.

![Sample Images](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/cifar10%20sample%20data.png)

*Figure 4: Sample images.*

---

### 🧠 NAS Optimization Results
**Best Trial:**  
- Layers = 3  
- Channels = [64, 64, 64]  
- Kernels = [5, 3, 5]  
- Learning rate = 0.00113  
- Validation Accuracy = **72.62%**  

**Hyperparameter Importance:**  
- Learning rate most influential (~37%)  
- Conv layers & channels moderately important  
- Kernel size least significant  

![Optimization History](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/optimization%20history%20plot.png)

*Figure 5: Optuna optimization history.*

![Intermediate Values](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/intermediate%20values%20plot.png)

*Figure 6: Intermediate values.*

![Hyperparameter Importance](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/Hyperperameter%20importance.png)

*Figure 7: Hyperparameter importance.*

![Parallel Coordinate](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/parallel%20coordinate%20plote.png)

*Figure 8: Parallel coordinate plot.*

![Slice Plot](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/slice%20plot.png)

*Figure 9: Slice plot.*

---

### 🏆 Best Model Training
- Trained for 30 epochs; **loss curves** indicate smooth convergence.  
- Training accuracy: ~78%, Validation accuracy: ~73%

![Training Curve](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/training%20curve.png)

*Figure 10: Training curve.*

---

### ✅ Final Evaluation
- **Classification Report:** Per-class metrics (precision, recall, F1-score)  
- **Confusion Matrix:** Class-wise performance

![Classification Report](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/classification%20report.png)

*Figure 11: Classification report.*

![Confusion Matrix](https://github.com/ashokpal122/NAS-CIFAR10-Optimizer/blob/main/figure/images/confusion%20matrix.png)

*Figure 12: Confusion matrix.*

---

## 4️⃣ Discussion
### Strengths
- NAS **automates architecture discovery**, saving manual effort  
- Hyperband pruning reduces **compute cost by ~50%**  
- Balanced dataset ensures fair evaluation  

### Limitations
- Search space **excludes dropout, batch normalization, or skip connections**  
- Limited trials (10) & short epochs (5 per trial) → shallow search  
- **No data augmentation** applied  

### Insights
- Learning rate = **most critical hyperparameter**  
- Wider networks perform better  
- Kernel size has minor impact

---

## 5️⃣ Conclusion
This project demonstrates the **effective application of NAS using Optuna** for CIFAR-10 classification.  
The discovered CNN architecture achieves high validation accuracy with **efficient training**, proving NAS to be a **valuable tool in resource-constrained settings**.

---

## 6️⃣ Recommendations
1. **Expand search space:** Include dropout, batch normalization, skip connections  
2. Increase **trials (≥50)** and **epochs (≥20)**  
3. Apply **data augmentation**: random crops, flips, color jitter  
4. Explore **advanced NAS**: DARTS, ENAS  
5. Use **transfer learning** for higher accuracy

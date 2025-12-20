# Brain Tumor Classification using Metaheuristic Optimization & Explainable AI (XAI)

This project explores the automation of Deep Learning model tuning for brain tumor classification. We applied **8 different Metaheuristic Algorithms** to optimize the hyperparameters of a **MobileNetV2** architecture and subsequently optimized the parameters of **Explainable AI (XAI)** techniques to ensure trustworthy model predictions.

---

## 📌 Project Overview
* **Goal:** Automate hyperparameter tuning for CNNs to classify MRI images into 4 categories.
* **Model:** MobileNetV2 (Transfer Learning).
* **Optimization Strategy:** 8 Metaheuristic Algorithms (PSO, SA, GA, WOA, GWO, Firefly, Flower Pollination, Tabu Search).
* **XAI Integration:** Optimized parameters for Grad-CAM, LIME, and DeepLIFT to improve interpretability.

---

## 📂 Dataset
* **Source:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/)
* **Total Images:** 7,025 MRI scans.
* **Classes:**
    1.  `Glioma`
    2.  `Meningioma`
    3.  `Pituitary`
    4.  `No Tumor`
* **Preprocessing:**
    * Resize: `224x224`
    * Normalization: `[0, 1]`
    * Augmentation: Rotation (±15°), Zoom (±10%), Shifts (±5%), Horizontal Flip.

---

## ⚙️ Methodology

### 1. Model Architecture Search
We defined a search space for the metaheuristic algorithms to optimize:
* **Learning Rate:** `[1e-3, 1e-4, 5e-5]`
* **Dense Units:** `[128, 256, 512]`
* **Dropout Rate:** `[0.3, 0.4, 0.5]`

### 2. Algorithms Implemented
We compared the performance of the following algorithms:
1.  **Particle Swarm Optimization (PSO)** 🏆 *(Top Performer)*
2.  **Whale Optimization Algorithm (WOA)** 🏆 *(Top Performer)*
3.  **Grey Wolf Optimization (GWO)**
4.  **Firefly Algorithm**
5.  **Simulated Annealing (SA)**
6.  **Genetic Algorithm (GA)**
7.  **Flower Pollination Algorithm (FPA)**
8.  **Tabu Search (Hybridized with SA/PSO)**

---

## 📊 Experimental Results

### Model Optimization Performance
The table below summarizes the best configuration found by each algorithm.

| Algorithm | Validation Accuracy | Best Configuration | Time (s) |
| :--- | :--- | :--- | :--- |
| **Particle Swarm (PSO)** | **54.78%** | `lr=0.0003`, `drop=0.19` | 4842s |
| **Whale Optimization (WOA)** | **54.69%** | `lr=0.0001`, `units=512`, `drop=0.3` | 9397s |
| **Grey Wolf (GWO)** | 54.60% | `lr=0.001`, `units=256`, `drop=0.4` | 9124s |
| **Simulated Annealing** | 53.46% | `lr=5e-05`, `units=512`, `drop=0.3` | 1884s |
| **Firefly Algorithm** | 52.85% | `lr=0.001`, `units=512`, `drop=0.5` | 8748s |
| **Genetic Algorithm** | 51.88% | `lr=0.001`, `units=128`, `drop=0.4` | 6446s |
| **Tabu Search + PSO** | 48.47% | `c1=1.0`, `c2=1.0`, `w=0.4` | - |
| **Tabu Search + SA** | 46.89% | `c1=1.0`, `c2=1.0`, `w=0.4` | - |

### XAI Parameter Optimization
We applied metaheuristics to find the best parameters for visualization techniques (Grad-CAM, LIME, DeepLIFT).

* **Best XAI Optimizer:** **Firefly Algorithm** 🟢
* **Metric:** Optimized for highest **Faithfulness** and **Sparsity**.
* **Comparison:**
    * **Firefly (Grad-CAM Score):** `0.2381`
    * **Flower Pollination (Grad-CAM Score):** `0.1600`
    * **WOA (Grad-CAM Score):** `0.0001`

---

## 📉 Analysis & Key Findings

### 1. Success of Swarm Intelligence
**PSO** and **WOA** were the only algorithms capable of finding a stable configuration that learned features from multiple classes. They achieved ~55% accuracy and demonstrated active exploration of the search space.

### 2. Failure of Evolutionary/Trajectory Methods
**Simulated Annealing** and **Genetic Algorithms** suffered from **Mode Collapse**.
* *SA Result:* Optimized loss by guessing "Meningioma" for almost every image.
* *GA Result:* Optimized loss by guessing "Glioma" for almost every image.
* This indicates these algorithms got stuck in local optima very early in the search process.

---

## 🚀 Conclusion

* **Best Model Optimizer:** **Whale Optimization Algorithm (WOA)** was selected as the most robust architecture searcher for this dataset, balancing accuracy (54.69%) and parameter stability.
* **Best XAI Optimizer:** **Firefly Algorithm** provided the most faithful visual explanations, ensuring the model focuses on the actual tumor regions rather than background noise.

---

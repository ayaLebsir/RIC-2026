# 🧠 MLP Regression from Scratch (NumPy)  LEBSIR AYA

## 📌 Overview

This project implements a **Multi-Layer Perceptron (MLP)** from scratch using only **NumPy**, without any deep learning frameworks.
The goal is to approximate a **non-linear 2D function** and demonstrate how neural networks learn through **forward propagation, backpropagation, and optimization**.

---

## 🎯 Objective

We aim to learn the following function:

[
f(x, y) = \sin(\sqrt{x^2 + y^2}) + 0.5 \cdot \cos(2x + 2y)
]

This is a **non-linear regression problem**, making it a good benchmark for neural networks.

---

## 🧩 Features

* ✅ MLP implemented from scratch (no TensorFlow / PyTorch)
* ✅ Forward propagation
* ✅ Backpropagation (gradient computation)
* ✅ Mini-batch Gradient Descent
* ✅ Momentum optimization
* ✅ He weight initialization
* ✅ Z-score normalization
* ✅ Train / Validation split
* ✅ Visualization of:

  * Dataset (3D)
  * Loss curve
  * Prediction vs Ground Truth

---

## 🏗️ Model Architecture

```
Input Layer : 2 neurons (x, y)
Hidden Layer 1 : 64 neurons (ReLU)
Hidden Layer 2 : 64 neurons (ReLU)
Output Layer : 1 neuron (Linear)
```

---

## ⚙️ Training Details

* Loss Function: **Mean Squared Error (MSE)**
* Optimizer: **Gradient Descent + Momentum**
* Epochs: **800**
* Batch size: **128**
* Learning rate: **0.005**

---

## 📊 Results

* ✔ Converging training and validation loss
* ✔ No significant overfitting
* ✔ Accurate approximation of the target function

---

## 📈 Visualizations

### 🔹 Ground Truth (Dataset)

3D scatter plot of the original function.

### 🔹 Learning Curve

Shows training vs validation loss over epochs.

### 🔹 Model Comparison

* Ground truth
* MLP prediction
* Absolute error

---

## 🧪 How to Run

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python Dataset.py
```

---

## 🧠 Key Concepts

* Neural networks from scratch
* Non-linear regression
* Activation functions (ReLU vs Linear)
* Gradient-based optimization
* Data normalization

---

## 🚀 Future Improvements

* 🔹 Add other activations (Tanh, Leaky ReLU)
* 🔹 Try different optimizers (Adam)
* 🔹 Hyperparameter tuning
* 🔹 Extend to higher-dimensional data

---

## 👩‍💻 Author

**Aya Lebsir**
AI Engineering Student
[LinkedIn](https://www.linkedin.com/in/aya-lebsir-57782636a)

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!

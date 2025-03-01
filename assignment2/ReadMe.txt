# README

## Overview
This project implements and evaluates multiple machine learning models using Support Vector Machines (SVM) and Neural Networks for image classification. It leverages k-fold cross-validation to tune hyperparameters and optimize performance on noisy image datasets. The project consists of three main parts:

1. **Linear SVM (Part 1)**: Trains a linear SVM with cross-validation to find the optimal regularization parameter `C`.
2. **Gaussian SVM (Part 2)**: Uses an RBF kernel and tunes both `gamma` and `C` to improve classification accuracy.
3. **Neural Network Classifier (Part 3)**: Implements a multi-layer perceptron (MLP) to classify images and optimizes its hidden layer size.

Each of these parts includes **Part 4**, which is embedded in all files and focuses on evaluating performance across different hyperparameters.

---

## Installation and Dependencies
This project requires Python 3 and the following libraries:

- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using:
```bash
pip install numpy matplotlib scikit-learn
```

---

## Dataset
The project works with preprocessed image datasets stored as `.npy` files:

- `train_images_flat.npy`: Training images.
- `train_labels_noisy.npy`: Noisy training labels.
- `train_images_flat_subset.npy`: A subset of training images for cross-validation.
- `train_labels_noisy_subset.npy`: A subset of noisy labels for cross-validation.
- `test_images_flat.npy`: Test images.
- `test_labels_binary.npy`: Binary test labels.

Ensure these files are available in the working directory before running the scripts.

---

## Implementation Details
### **Part 1: Linear SVM**
- Uses `SVC(kernel='linear')` with k-fold cross-validation (`k=5`).
- Tunes the regularization parameter `C` using logarithmically spaced values.
- Trains on a subset of data, selects the best `C`, then retrains on the full dataset.
- Evaluates final model performance on the test set.
- Plots training and test error versus `C`.

### **Part 2: Gaussian SVM (RBF Kernel)**
- Uses `SVC(kernel='rbf')` to handle complex decision boundaries.
- Tunes `gamma` and `C` using nested cross-validation.
- Finds optimal `gamma`, then selects the best `C` for that `gamma`.
- Retrains the model on the full dataset and evaluates it on the test set.
- Compares test error with the linear SVM model.
- Plots training and test error versus `gamma`.

### **Part 3: Neural Network (MLP)**
- Implements a `MLPClassifier` with a single hidden layer.
- Tunes the hidden layer size using k-fold cross-validation.
- Selects the best hidden layer size, then retrains on the full dataset.
- Evaluates final model performance on the test set.
- Plots training and test error versus hidden layer size.

### **Part 4 (Embedded in All Parts)**
- Handles dataset loading and preprocessing.
- Implements k-fold cross-validation for error estimation.
- Stores and tracks the best model hyperparameters.
- Standardizes error reporting and visualization across all models.

---

## Results and Performance Evaluation
- The optimal hyperparameters for each model are displayed in the console.
- Training and test errors are visualized using plots.
- The best performing model is identified based on the lowest test error.
- Performance comparison across different models helps determine the most suitable approach for the dataset.
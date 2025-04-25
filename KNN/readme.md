# K-Nearest Neighbours for Binary Classification (50 points)

## Overview
This project implements the **K-Nearest Neighbours (KNN)** algorithm from scratch for binary classification, incorporating custom distance functions, data normalization techniques, and hyperparameter tuning.

---

## Files to Modify
Only edit the following files in the `work/` directory:
- `knn.py`
- `utils.py`

---

## Components

### 1. F1 Score and Distance Functions
Implemented in `utils.py`:
- `f1_score(y_true, y_pred)`
- `Distance` class:
  - `euclidean_distance(x1, x2)`
  - `minkowski_distance(x1, x2)`
  - `cosine_similarity_distance(x1, x2)`

### 2. KNN Algorithm
Implemented in `knn.py`:
- `KNN` class:
  - `train(train_features, train_labels)`
  - `get_k_neighbors(test_point)`
  - `predict(test_features)`

### 3. Data Transformation
Implemented in `utils.py`:
- `NormalizationScaler` class:
  - `__call__(features)`
- `MinMaxScaler` class:
  - `__call__(features)`

### 4. Hyperparameter Tuning
Implemented in `utils.py`:
- `HyperparameterTuner` class:
  - `tuning_without_scaling(distance_funcs, hyperparams, X_train, y_train, X_val, y_val)`
  - `tuning_with_scaling(distance_funcs, scaling_classes, hyperparams, X_train, y_train, X_val, y_val)`

### 5. Testing
Use `test.py` to validate your implementation. You should see a green output as shown in `test.png` if all is implemented correctly.

---

## HyperparameterTuner Class

### Methods

#### `tuning_without_scaling`
Tunes `k` and distance function without applying any scaling.
- Inputs: model, X, y, param_grid, cv
- Output: best model with highest F1 score

#### `tuning_with_scaling`
Tunes `k`, distance function, and scaler class (Normalization or MinMax).
- Inputs: model, X, y, param_grid, scaler, cv
- Output: best model with highest F1 score


## Notes
- Do **not** import any new libraries.
- Follow type annotations carefully.
- Only changes to `knn.py` and `utils.py` will be graded.
- Submit via Vocareum after you're done.

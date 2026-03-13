# predictive-modeling-learning
A repository containing my learning experience with programming various predictive modeling techniques  with data coming from a locally hosted sql database using docker.

Models I want to implement include:
- Regression Models (Continuous Value Prediction)
    - [] Linear Regression: Predicts a target numerical value based on input features.
    - [] Polynomial Regression: Models relationships as \(n\)-th degree polynomials.
    - [] Support Vector Regression (SVR): Finds a hyperplane to fit data points within a threshold.
- Classification Models (Categorical Prediction)
    - [] Logistic Regression: Predicts binary outcomes (yes/no).
    - [] Decision Trees: Uses a tree-like, hierarchical structure for decisions.
    - [] Random Forests: An ensemble method combining multiple decision trees for higher accuracy.
    - [] Support Vector Machines (SVM): Classifies data by finding the optimal hyperplane.
    - [] Naive Bayes: Based on Bayes' theorem for probability classification.
    - [] K-Nearest Neighbors (KNN): Classifies based on the closest training examples.
    - [] Gradient Boosting (XGBoost, LightGBM): Boosts performance by training models in sequence to correct errors.
- Machine Learning & Neural Networks
    - [] Neural Networks (MLP): Complex, layered models designed for high-dimensional or non-linear data.
    - [] CNN (Convolutional Neural Networks): Used for image and structured data.
    - [] RNN (Recurrent Neural Networks): Used for sequential data.
    - [] Deep Learning: Advanced, multi-layered neural networks.
- Clustering & Pattern Recognition
    - [] K-Means Clustering: Segments data into \(k\) clusters.
    - [] Hierarchical Clustering: Builds a tree of clusters.
    - [] Density-Based Clustering (DBSCAN): Groups points based on density.
- Time Series & Forecasting
    - [] ARIMA (Autoregressive Integrated Moving Average): Analyzes and forecasts time-dependent data.
    - [] Exponential Smoothing: Forecasts by weighting past data with exponential decay.
- Other Methods
    - [] Ensemble Methods: Techniques like boosting and bagging that combine multiple models.
    - [] Dimensionality Reduction (PCA): Reduces variable count while retaining information.
    - [] Anomaly Detection: Identifies outliers in data.
- Key Steps in Selecting and Implementing Models
    - Define the Business Problem: Determine if the goal is classification, regression, or forecasting.
    - Data Preprocessing: Clean, transform, and prepare data (crucial for model success).
    - Model Selection & Training: Choose algorithms based on data type and split data into training/validation sets.
    - Evaluation & Tuning: Use metrics (e.g., accuracy, ROC-AUC) to test performance and optimize hyperparameters.
    - Deployment: Implement the final model for real-time or batch prediction.

## File Structure
```
src/predictive_modeling_learning/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── main.py              # Root click group, registers all subgroups
│   ├── regression.py        # click group: linear, polynomial
│   ├── classification.py    # click group: 7 models
│   ├── clustering.py        # click group: kmeans, hierarchical, dbscan
│   ├── time_series.py       # click group: arima, exp_smoothing
│   ├── neural.py            # click group (renamed from advanced.py)
│   └── common.py            # Shared click options/decorators (keep this)
├── models/
│   ├── __init__.py
│   ├── base.py              # AbstractBaseModel (NEW - most critical)
│   ├── registry.py          # MODEL_REGISTRY dict (NEW)
│   ├── regression/
│   │   ├── linear.py
│   │   └── polynomial.py
│   ├── classification/
│   │   ├── decision_tree.py
│   │   ├── gradient_boosting.py
│   │   ├── k_nearest_neighbors.py
│   │   ├── logistic_regression.py
│   │   ├── naive_bayes.py
│   │   ├── random_forests.py
│   │   └── support_vector_machines.py
│   ├── clustering/
│   │   ├── k_means.py
│   │   ├── hierarchical.py
│   │   └── density_based.py
│   ├── time_series/
│   │   ├── arima.py
│   │   └── exponential_smoothing.py
│   └── neural_networks/     # renamed from machine_learning/
│       ├── neural_networks.py
│       ├── deep_learning.py
│       ├── convolutional.py
│       └── recurrent.py
├── io/
│   ├── __init__.py
│   ├── csv_loader.py
│   ├── db_loader.py
│   ├── loaders.py
│   ├── preprocessor.py
│   └── splitter.py
└── utils/
    ├── __init__.py
    ├── metrics.py
    └── serialization.py
```

---

## To Do

### Phase 1 — Data Layer
- [X] `io/csv_loader.py` — load CSV files into a DataFrame
- [X] `io/db_loader.py` — connect, query, list_tables, inspect_table
- [X] `io/loaders.py` — unified load() entry point for csv and db sources
- [X] `io/preprocessor.py` — scaling, encoding, imputation
- [ ] `io/splitter.py` — train/test split, returns DataBundle

### Phase 2 — Model Foundation
- [ ] `models/base.py` — AbstractBaseModel with build, train, predict, evaluate, save, load
- [ ] `models/registry.py` — MODEL_REGISTRY dict and @register decorator

### Phase 3 — Models
#### Regression
- [ ] `models/regression/linear.py` — Linear Regression
- [ ] `models/regression/polynomial.py` — Polynomial Regression

#### Classification
- [ ] `models/classification/logistic_regression.py` — Logistic Regression
- [ ] `models/classification/decision_tree.py` — Decision Tree
- [ ] `models/classification/random_forests.py` — Random Forest
- [ ] `models/classification/support_vector_machines.py` — SVM
- [ ] `models/classification/naive_bayes.py` — Naive Bayes
- [ ] `models/classification/k_nearest_neighbors.py` — KNN
- [ ] `models/classification/gradient_boosting.py` — Gradient Boosting

#### Clustering
- [ ] `models/clustering/k_means.py` — K-Means
- [ ] `models/clustering/hierarchical.py` — Hierarchical Clustering
- [ ] `models/clustering/density_based.py` — DBSCAN

#### Time Series
- [ ] `models/time_series/arima.py` — ARIMA
- [ ] `models/time_series/exponential_smoothing.py` — Exponential Smoothing

#### Neural Networks
- [ ] `models/neural_networks/neural_networks.py` — MLP
- [ ] `models/neural_networks/convolutional.py` — CNN
- [ ] `models/neural_networks/recurrent.py` — RNN
- [ ] `models/neural_networks/deep_learning.py` — Deep Learning

### Phase 4 — Utilities
- [ ] `utils/metrics.py` — shared metric helpers per model category
- [ ] `utils/serialization.py` — save/load model artifacts via joblib

### Phase 5 — CLI Layer
- [ ] `cli/main.py` — root click group, registers all subgroups
- [ ] `cli/common.py` — shared click options/decorators
- [ ] `cli/regression.py` — commands: linear, polynomial
- [ ] `cli/classification.py` — commands: logistic, decision-tree, random-forest, svm, naive-bayes, knn, gradient-boosting
- [ ] `cli/clustering.py` — commands: kmeans, hierarchical, dbscan
- [ ] `cli/time_series.py` — commands: arima, exp-smoothing
- [ ] `cli/neural.py` — commands: mlp, cnn, rnn, deep

### Phase 6 — Testing
- [ ] Tests for `io/` layer
- [ ] Tests for `models/base.py` and registry
- [ ] Tests for each model category
- [ ] End-to-end CLI tests
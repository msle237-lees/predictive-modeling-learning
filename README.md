# predictive-modeling-learning
A repository containing my learning experience with programming various predictive modeling techniques  with data coming from a locally hosted sql database using docker.

Models I want to implement include:
- Regression Models (Continuous Value Prediction)
    - [X] Linear Regression: Predicts a target numerical value based on input features.
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

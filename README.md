# Credit Card Fraud Detection

This repository contains a machine learning project focused on detecting fraudulent credit card transactions using a highly imbalanced dataset. The primary goal is to build and evaluate several classification models to accurately identify fraudulent transactions while minimizing false positives.

## Project Overview

The notebook walks through the complete machine learning pipeline:

1.  **Exploratory Data Analysis (EDA)**: The dataset is analyzed to understand its structure and the severe class imbalance (only 0.17% of transactions are fraudulent). Visualizations like heatmaps and boxplots are used to explore feature correlations and distributions.

2.  **Data Preprocessing**:
    * The `Time` and `Amount` features are scaled using `StandardScaler` to normalize their distributions.
    * The dataset is split into training and testing sets, using stratification to maintain the same proportion of fraud cases in both sets.

3.  **Feature Importance and Selection**:
    * A `RandomForestClassifier` is trained to determine the most influential features for predicting fraud.
    * Only the top features (with an importance score > 0.05) are selected for model training to reduce noise and prevent overfitting.

4.  **Model Training and Hyperparameter Tuning**:
    * Four different classification models are trained and compared:
        * Logistic Regression
        * K-Nearest Neighbors (KNN)
        * Gaussian Naive Bayes
        * XGBoost
    * `GridSearchCV` is used to find the optimal hyperparameters for each model, with a focus on maximizing the **F1-Score** for the minority class (fraud).

## Results and Conclusion

The performance of each tuned model was evaluated based on its precision, recall, and F1-score for the fraud class (Class 1).

* **XGBoost** emerged as the best-performing model, achieving a strong balance between precision and recall, with an **F1-Score of 0.89**.
* **K-Nearest Neighbors (KNN)** showed the highest precision but had a slightly lower recall.
* **Naive Bayes** had very high recall but extremely low precision, leading to a high number of false positives.

The final analysis concludes that **XGBoost** is the most robust and reliable model for this real-world fraud detection scenario, as it provides the best trade-off between detecting fraud and minimizing false alarms.

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection** dataset from Kaggle. It contains anonymized transactions made by European cardholders.

* **Source**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Note**: Due to its size, the `creditcard.csv` file should be handled with Git LFS or downloaded directly from the source.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib & Seaborn
* Jupyter Notebook

## How to Run

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/your-username/credit-card-fraud-detection.git](https://github.com/your-username/credit-card-fraud-detection.git)
    ```

2.  Navigate to the project directory:
    ```bash
    cd credit-card-fraud-detection
    ```

3.  Install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
    ```

4.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project directory.

5.  Launch the Jupyter Notebook:
    ```bash
    jupyter notebook MLProject.ipynb
    ```

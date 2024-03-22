[![Documentation](https://img.shields.io/badge/documentation-read-green)](https://github.com/vineetkukreti/Titanic_technocart)
[![GitHub stars](https://img.shields.io/github/stars/deepfence/FlowMeter)](https://github.com/vineetkukreti/Titanic_technocart)
[![GitHub issues](https://img.shields.io/github/issues/deepfence/FlowMeter)](https://github.com/vineetkukreti/Titanic_technocart)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1op08QCOikM5mRLWFyjdkuDFq3B0L_-ft?usp=sharing)


# Predicting Titanic Survivors

## Problem Statement

The objective is to develop a machine learning model that accurately predicts the survival status of passengers aboard the Titanic. This involves analyzing the dataset containing information about passengers and their survival outcome, and then training a model to make predictions based on relevant features.

<div style="text-align:center">
    <img src="images\titanic_image.jpg" alt="Alt Text" width="500" height="300">
</div>



## 1. Dataset

The data has been split into two groups:

- training set [train.csv](Dataset\train.csv)
- test set [test.csv](Dataset\results.csv)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.


| Variable | Definition                 | Key                                      |
|----------|----------------------------|------------------------------------------|
| survival | Survival                   | 0 = No, 1 = Yes                          |
| pclass   | Ticket class               | 1 = 1st, 2 = 2nd, 3 = 3rd                |
| sex      | Sex                        |                                          |
| Age      | Age in years               |                                          |
| sibsp    | has of siblings / spouses   |                                          |
|          | aboard the Titanic         |                                          |
| parch    | has of parents / children   |                                          |
|          | aboard the Titanic         |                                          |
| ticket   | Ticket number              |                                          |
| fare     | Passenger fare             |                                          |
| cabin    | Cabin number               |                                          |
| embarked | Port of Embarkation        | C = Cherbourg, Q = Queenstown, S = Southampton |
| name     | Name's of Passengers       |                                          |



## 2. Exploratory Data Analysis (EDA)

The EDA phase involves:
- Data visualization: plotting relationships between target class and predictor variables, exploring distributions of features, etc.
- Data profiling: generating a profiling report to understand the dataset's structure, summary statistics, missing values, etc. visit   [output_file](output_file.html)


<div style="text-align:center">
    <img src="images\image.png" alt="Alt Text" width="500" height="300">
</div>




## 3. Data Preprocessing

This phase includes:
- Handling missing values: imputing null values for certain features like Age, Fare, Embarked, etc.
- Feature engineering: creating new features like Deck, Title, Age groups, Fare groups, etc.
- Label encoding: converting categorical features into numerical representations using label encoding.
- Creating additional features: like 'isAlone' to indicate if a passenger is traveling alone, 'classAge' by multiplying Pclass and Age, etc.

## 4. Feature Selection

Utilizing ensemble methods to identify the most important features for survival prediction. This involves using algorithms such as Random Forest, Extra Trees, AdaBoost, and Gradient Boosting to evaluate feature importances.

## 5. Model Training and Prediction

We employ various machine learning algorithms for prediction, including Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Support Vector Classifier, XGBoost, and Decision Trees. Hyperparameter tuning is performed to optimize the models for accuracy.

### Model Parameters
- **Random Forest Parameters**: 
  - Number of Estimators: 1000
  - Criterion: Entropy
  - Max Features: 4
  - Max Depth: 50
  - Min Samples Leaf: 5
- **AdaBoost Parameters**:
  - Number of Estimators: 500
  - Learning Rate: 0.75
- **Gradient Boosting Parameters**:
  - Number of Estimators: 500
  - Max Depth: 5
  - Min Samples Leaf: 2
- **Support Vector Classifier Parameters**:
  - Kernel: Linear
  - C: 0.025
- **XGBoost Parameters**:
  - Number of Estimators: 600
  - Learning Rate: 0.15
  - Max Depth: 5
  - Min Child Weight: 7


## 6. Model Evaluation

We evaluate the performance of each model using cross-validation techniques and calculate accuracy scores. Feature importances are analyzed to understand the significance of each feature in predicting survival.

## 7. Result and Inference

- **Feature Importance**: Features such as gender, title, age, fare, and deck are found to be significant predictors of survival across multiple classifiers.
- **Model Performance**: XGBoost, Random Forest, and Extra Trees classifiers exhibit high accuracy in predicting survival, with XGBoost achieving the highest performance.

These are the Train Accuracy we have got so far :
- Training Data Accuracy Matrix:
- XGBoost Algorithm: 86.20%
- RandomForest Algorithm: 86.20%
- Decision Trees Classifier: 86.31%

## 8. Dependencies

- Python libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost.
- Other packages: ydata-profiling (for data profiling).

## 9. Files

- Titanic_Survivors.ipynb: Jupyter Notebook containing the entire code implementation.
- train.csv: Training dataset containing passenger information.
- test.csv: Test dataset for making predictions.
- preprocessed_train.csv: Preprocessed training dataset.
- preprocessed_test.csv: Preprocessed test dataset.
- results.csv: CSV file containing the predictions generated by the model.
- output_file.html: webview of the dataset
- requirements.txt: Packages used in the project are present in the this file . While clone the repo ,it's suggested that one should , firstly run this file. ```pip install -r requirements.txt ```

## 10. Usage

- Clone the repository and ensure all dependencies are installed.
- Run Titanic_Survivors.ipynb in a Jupyter Notebook environment or any compatible platform.
- Follow the instructions and execute the code cells to perform data preprocessing, model training, and prediction.
- Access the results.csv file to view the predictions made by the model.

## 11. Author

**Author:** Vineet Kukreti

---

## 12. License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).


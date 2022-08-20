**SUMMARY**

**Introduction**

The Main purpose of the project is use historical customer data to predict behaviour to retain customers by identifying the potential customers who have a higher probability to churn.
This help the company to understand the pinpoints and patterns of customer churn and will increase the focus on strategising customer retention. 

**Process (Import, Cleaning, wrangling, correction, Analysis, visualization, pre processing)**

The Data is imported from an SQL database into Python Jupyter notebooks and the respective data wrangling, cleaning, correction are performed. Then Statistical analysis and visualization with univariate, and multivariate plots.
Before Data Modelling, the data is pre processed, under which, the Data Segregation into Train and Test, Scaling/Normalization of data using Min-Max Scaler. 

Then it is verified for Target class imbalance, From this it is understandable that the target variable is unevenly distributed where Not Churn - 0 has 73.4% whileChurn - 1 has 26.6%, so majority of the data would assume Not Churn. That is around 5163 for Not Churn and 1869 for Churn.

![image](https://user-images.githubusercontent.com/88423149/185735832-c50fe55b-9bcf-44af-83fb-4df8590915e9.png)

If minority is within this percentage, then it is considered as imbalanced, but the levels are as follows : (Citation: Arora, Nisha. (2021). Re: How to know that our dataset is imbalance?).

mild ---> 20-40% of the data set

Moderate ---> 1-20% of the data set

Extreme ---> <1% of the data set

In this case, the Churn target attribute is Mildly imbalanced at 26.6% (between 20-40%) and is not required to get treated for modelling.

**Process continuation (Train-test split, Model Training, testing, evaluation, hyper parameter training, selection, Pickling for future use)**

• Model is fitted, trained and tuned with

1) Decision Tree Classifier 2) RandomForestClassifier 3) Adaboost Classifier 4) Bagging Classifier 5) Graident Boost Classifier 6) LightGBM Classifier 7) XGBoost Classifier 8) CatBoost Classifier

• The classification accuracies before hyperparameter tuning are as follows:

![image](https://user-images.githubusercontent.com/88423149/185736782-5cae98d6-47e8-4d4f-aaf1-fc84a7c870f6.png)

![image](https://user-images.githubusercontent.com/88423149/185736788-33d43c17-6c36-43d1-8c84-66505a54b162.png)

Before and After hyperparameter tuning for LightGBM Classification model :

**A) Before Hyperparameter Tuning:**

1) Accuracy = 0.798 (79.8%)

2) Precision = 0.645 (64.5%)

3) Recall = 0.527 (52.7%)

4) F1-Score = 0.580 (58.0%)

5) Area under the curve value (from ROC) = 0.711 (71.1%)

6) Confusion matrix = 1684 predicted correctly and 426 predicted incorrectly out of 2110

**B) After Hyperparameter Tuning:**

1) Accuracy = 0.802 (80.2%)

2) Precision = 0.662 (66.2%)

3) Recall = 0.533 (53.3%)

4) F1-Score = 0.590 59.0%)

5) Area under the curve value (from ROC) = 0.717 (71.7%)

6) Confusion matrix = 1697 predicted correctly and 413 predicted incorrectly out of 2110

The reason why LightGBM is the best Hypertuned Classification model for this dataset is because of the following:

1) 0.4% increase in accuracy

2) 1.7% increase in Precision

3) 0.6% increase in Recall

4) 1% increase in F1-Score

6) 0.6 % increse in AUC (Area under the curve)


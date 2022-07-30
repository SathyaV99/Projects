**SUMMARY** 

The main objective of the project is to predict the class as "Normal", "Type H", "Type S" from the provided biomechanical features. In this project, data was cleaned, explored, analysed, visualized, pre processed, trained, tested and tuned. 

*Refer to "Project 3.ipynb" for all the details, the inference and the conclusions are also detailed within the notebook.*

KNN Classifier was used for this project, it was found that there was slight imbalance in the dataset. 

**CLASS IMBALANCE**

![image](https://user-images.githubusercontent.com/88423149/181877501-a0a1687a-8c14-4ccb-bb76-da6eaf550ef0.png)

From the following research article, this was concluded as Mild as the above image had the following imbalance : Abnormal has 67% and Normal has 33%

If minority is within this percentage, then it is considered as : (Citation: Arora, Nisha. (2021). Re: How to know that our dataset is imbalance?).

mild ---> 20-40% of the data set

Moderate ---> 1-20% of the data set

Extreme ---> <1% of the data set

In this case, the class attribute is mildly imbalanced at 33% (between 20-40%) but it is not required to be balanced as it is viable for the dataset.

**The model was trained, tested and tuned using "GridSearchCV". The final results are as follows:**

• The model selection is critically chosen based on Precision,Recall, f1-score, AUC values, because when Hypertuning KNN which is a classification model, we dont rely on accuracy.

• In this case, the dataset was scaled and target 'class' attribute was hot encoded after unifying the Abnormal categories(Type_S,Type_H).

• The reason why this model after hyperparameter tuning is the best, is because of the following:

**A) Before Hyperparameter Tuning:**

1) Accuracy = 0.817

2) Precision = 0.777

3) Recall = 0.656

4) F-Score = 0.711

5) Area under the curve value (from ROC) = 0.778

6) K value = 5

7) RMSE = 0.427

8) Confusion matrix = 76 predicted correctly and 17 predicted incorrectly out of 93

**B) After Hyperparameter Tuning:**

1) Accuracy = 0.827

2) Precision = 0.785

3) Recall = 0.687

4) F-Score = 0.733

5) Area under the curve value (from ROC) = 0.794

6) K value = 23

7) RMSE = 0.414

8) Confusion matrix = 77 were predicted correctly and 16 were incorrectly predicted out of 93. That is about 82.79% accurate and 17.2% inaccurate.

• In Hyperparameter Tuning, the various Hyperparameters used to improve the model were : K value, K-fold Cross validation splitting strategy (cv), weights, metrics

• In that, the following settings were K-value = 23, K-fold Cross validation (cv) = 8, weights = distance, metrics = Minkowski, n_jobs = -1.

**Explanation of each Hyperparamter:**

a) K-value is based on which the target attribute finds the k nearest neighbour, in this case, it finds the nearest 23 values and assumes the target as such.

b) K-fold cross validation splitting strategy (cv) compares multiple hyperparameter with the model and in this case splits i into 8 different models with different hyperparameter values etc.

c) weights is a functionality that gives more weightage to points in this case, for 'distance', so the closer the points, more the weightage, better the model.

d) metrics is the distance parameter the model uses, the distance 'Minkowski' selects based on Manhattan and Euclidean distance measure.

• This model has been tuned with all the features, hyperparameters, and libraries available.

**Conclusion**

**Reason why this is the best hypertuned model for KNN Classifier**

1) 1% increase in accuracy

2) 0.8% increase in Precision

3) 3.1% increase in Recall

4) 2.2% increase in F-Score

5) 1.3% decrease in RMSE (Root mean square error)

6) 1.6 % increse in AUC (Area under the curve)

**REFER TO THE NOTEBOOK FOR MORE DETAILS**

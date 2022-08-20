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

**Process continuation (Train-test split, Model Training, testing, evaluation, hyper parameter training, selection)**



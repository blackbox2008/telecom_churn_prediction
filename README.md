# telecom_churn_prediction
For the current project we use the churn database WA_Fn-UseC_-Telco-Customer-Churn.csv available on https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn 

Churn, also called attrition, is a term used usually to indicate a customer leaving the service of one company in favor of another company, 
not desired by the business owners. We use the Churn dataset to analyze the sample data to apply some descriptive statistics methods to find 
patterns and trend of churn, explore the data, the relations between predictor variables and the target variable, EDA, and other tasks such as 
prediction and classification using decision tree, logistic regression, naïve bayes methods.
all phases of data preparation, data preprocessing, exploratory data analysis, modeling and evaluation may be carried out one-button.
the churn.py script includes comments about different phases.
For configuring the parameters in several parts of the source code, please read the following comments:
 a) decision tree building
 X_train and X_test data include the predictor names in the model. you can add or delete predictors
 but remember to modify the related lines (lines 232-233) to generate the respective dummy variables.
 to run the CART or C5.0 algorithms, you may run the fitting fuction in the source code with the customized parameters.
 maxdepth, minsampleaf. maxdepth =5 is recommended.

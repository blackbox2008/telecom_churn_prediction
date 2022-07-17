# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:18:03 2022

@author: Massoud Sharifi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
import graphviz
import math

pd.options.display.max_columns=25
df = pd.read_csv(r'C:\Users\Massoud\Documents\GitHub\churn_prediction\datasets\churn.csv')
#Clean data
# drop null values 
df.TotalCharges = df.TotalCharges.replace(' ', np.nan)
df =df.dropna(how = 'any')
df["TotalCharges"]=pd.to_numeric(df.TotalCharges,errors='coerce')

# fixing MISCLASSIFICATIONS

df.MultipleLines = df.MultipleLines.apply(lambda x: "No" if x =="No phone service" else x)
for col in df:
    if (df[col].dtype=="object") and (df[col].nunique()==3):
         df[col] = df[col].apply(lambda x: "No" if x=="No internet service" else x)

# standardize the numeric fields


df['tenure_z'] = stats.zscore(df['tenure'])
df['MonthlyCharges_z'] = stats.zscore(df['MonthlyCharges'])
df['TotalCharges_z'] = stats.zscore(df['TotalCharges'])

fig,ax = plt.subplots(1,3,figsize=(15,5))
for i,x in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges']):
    ax[i].hist(df[x][df.Churn=="No"],label="Churn=0",bins=30)
    ax[i].hist(df[x][df.Churn=="Yes"],label="Churn=1",bins=30)
    ax[i].legend()
    
df.loc[df['PaymentMethod'] == 'Bank transfer (automatic)', 'PaymentMethod'] = 'Bank transfer' #to shorten value
df.loc[df['PaymentMethod'] == 'Credit card (automatic)', 'PaymentMethod'] = 'Credit card'   

# partitioning data
churn_train, churn_test = train_test_split(df, test_size = 0.40, random_state = 7)
churn_train.reset_index(inplace=True)
churn_test.reset_index(inplace=True)
del churn_train['Unnamed: 0']
del churn_train['customerID']
del churn_test['Unnamed: 0']
del churn_test['customerID']
#validating partioning
# two sample t-test for the difference in means, for numeric variables
# two-sample z-score test for the difference in proportions for categorical variables 
# with two classes (dichotomous)
# test for homogeneity of proportions for categorical variables with more than two classes (polychotomous).

alpha = 0.05
num_vars = ['tenure_z', 'MonthlyCharges_z', 'TotalCharges_z']
flag_vars = ['gender']
flag_vars2 = ['SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
multinomial_var = ['Contract', 'PaymentMethod']

def print_ht_result(variable, p_value, alpha):
    
    if p_value > alpha:
        print(f'H0 for {variable} variable cannot be rejected: 2 samples are similar')
    else:
        print(f'H0 for {variable} variable can be rejected with {(1-alpha) *100}% confidence: 2 samples are different')
#two-sample t-test
title ='Two-sample T-test results'
print(f'{title}\n-----------------------------')
for i, col in enumerate(num_vars):
    t_test, p_value = stats.ttest_ind(a=churn_train[col], b=churn_test[col], equal_var=True)
    
    print_ht_result(col, p_value, alpha)

def calc_z_test(x1, x2, n1, n2, alpha):
    p1 = x1/n1
    p2 = x2/n2
    p_pooled = (x1 + x2) /(n1 + n2)
    z_data = (p1-p2)/math.sqrt(p_pooled * (1-p_pooled)*((1/n1) +(1/n2)))
    p_value = stats.norm.sf(abs(z_data)) * 2
    return p_value
    
#two-sample z-test
title ='Two-sample Z-test results'
print(f'{title}\n-----------------------------')
for i, col in enumerate(flag_vars): #for gender attribute 
    x1 = churn_train[churn_train[col] == 'Male'][col].count()
    x2 = churn_test[churn_test[col] == 'Male'][col].count()
    n1 = churn_train[col].count()
    n2 = churn_test[col].count()
    p_value = calc_z_test(x1, x2, n1, n2, alpha)
    print_ht_result(col, p_value, alpha)

for i, col in enumerate(flag_vars2): # for other 
    x1 = churn_train[churn_train[col] == 'Yes'][col].count()
    x2 = churn_test[churn_test[col] == 'Yes'][col].count()
    n1 = churn_train[col].count()
    n2 = churn_test[col].count()
    p_value = calc_z_test(x1, x2, n1, n2, alpha)
    print_ht_result(col, p_value, alpha)
    
    
# test for the homogeneity of proportions
# Observed frequencies 
for i, col in enumerate(multinomial_var):
    print(f'Dataset: {col}')
    col_total = []
    temp_val = []
    dataset_grand_total = 0
    print(f'{"Dataset":<10}', end='')
    for j, val in enumerate(sorted(list(churn_train[col].unique()))):
        print(f'{val:>20}', end = '')
    print(f'{"Total":>20}', end='')      
    print(f'\n{"Training":<10}', end='')
    row_total = 0    
    for j, val in enumerate(sorted(list(churn_train[col].unique()))):
        temp_val.append(churn_train[churn_train[col] == val][col].count())
        row_total += temp_val[j]
        col_total.append(temp_val[j])
        print(f'{temp_val[j]:>20}', end = '')
    print(f'{row_total:>20}', end='')
    row_total = 0
    print(f'\n{"Test":<10}', end='')
    for j, val in enumerate(sorted(list(churn_test[col].unique()))):
        temp_val[j] = churn_test[churn_test[col] == val][col].count()
        col_total[j] += temp_val[j]
        row_total += temp_val[j]
        print(f'{temp_val[j]:>20}', end = '') 
    print(f'{row_total:>20}', end='')    
    print(f'\n{"Total":<10}', end='')
    for j, val in enumerate(col_total):
        dataset_grand_total += col_total[j]
        print(f'{col_total[j]:>20}', end = '')    
    print(f'{dataset_grand_total:>20}')
    print('\n'+'-'*130, end='')
    
#Modeling Phase
# CART decision tree fuction
def fitting(X_train, y_train, X_test, y_test, criterion, splitter, maxdepth, clweight, minsampleaf):
    model = DecisionTreeClassifier(criterion = criterion, 
                                   splitter=splitter,
                                   max_depth=maxdepth,
                                   class_weight = clweight,
                                   min_samples_leaf=minsampleaf)

    cart = model.fit(X_train,y_train)
    # Predict class labels on training data
    pred_labels_train = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_test = model.predict(X_test)
    
     # Tree summary and model evaluation metrics
    print('\n************** Tree Summary *****************')
    print('Classes: ', cart.classes_)
    print('Algorithm: ', 'C5.0')
    print('Tree Depth: ', cart.tree_.max_depth)
    print('No. of leaves: ', cart.tree_.n_leaves)
    print('No. of predictors: ', cart.n_features_in_)
    print('--------------------------------------------------------')
    print("")
    print('++++++++++++++++++ Evaluation on Training Data ++++++++++++++++++')
    score_train = model.score(X_train, y_train)
    print('Accuracy Score: ', score_train)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_train))
    print('***************************************************')
    print("")
    print('#################### Evaluation on Test Data ###################')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_test))
    print('--------------------------------------------------------')

    dot_data = export_graphviz(cart, out_file=None, 
     class_names=[str(list(cart.classes_)[0]), str(list(cart.classes_)[1])], # the target names.
     feature_names=X_train.columns, # the feature names.
     filled=True, # Whether to fill in the boxes with colours.
     rounded=True, # Whether to round the corners of the boxes.
     special_characters=True)
    graph = graphviz.Source(dot_data)
    return cart, graph
#Building C5.0 decision tree

y_train = churn_train.Churn.values
y_test = churn_test.Churn.values
#change Tenure to categorical column
def tenure_split(df) : 
 if df['tenure'] <= 13 :
     return '0–20'
 elif (df['tenure'] > 13) & (df['tenure'] <= 60 ):
     return '20–60'
 elif df['tenure'] > 60 :
     return '60plus'
# change TotalCharges to categorical column
def totalcharges_split(df) : 
 if df['TotalCharges'] <= 1000 :
     return '0–1k'
 elif (df['TotalCharges'] > 1000) & (df['TotalCharges'] <= 2000 ):
     return '1k-2k'
 elif (df['TotalCharges'] > 2000) & (df['TotalCharges'] <= 3000) :
     return '2k-3k'
 elif df['TotalCharges'] > 3000 :
     return '3kplus'

churn_train['tenure_group'] = churn_train.apply(lambda churn_train:tenure_split(churn_train), axis = 1)
churn_train['totalcharges_group'] = churn_train.apply(lambda churn_train:totalcharges_split(churn_train), axis = 1)
churn_test['tenure_group'] = churn_test.apply(lambda churn_test:tenure_split(churn_test), axis = 1)
churn_test['totalcharges_group'] = churn_test.apply(lambda churn_test:totalcharges_split(churn_test), axis = 1)   

# build X_train by keeping only relevant predictors
X_train = churn_train[['SeniorCitizen', 'tenure_group', 'PhoneService', 'InternetService', 'Contract', 'totalcharges_group']]
X_test = churn_test[['SeniorCitizen', 'tenure_group', 'PhoneService', 'InternetService', 'Contract', 'totalcharges_group']]



# Turn categical varibales of the training and test data sets to dummy varibales
#SeniorCitizen flag variable

X_train = pd.get_dummies(X_train, columns=['SeniorCitizen', 'tenure_group', 'PhoneService', 'InternetService', 'Contract', 'totalcharges_group'])
X_test = pd.get_dummies(X_test, columns=['SeniorCitizen', 'tenure_group', 'PhoneService', 'InternetService', 'Contract', 'totalcharges_group'])

cart, graph = fitting(X_train, y_train, X_test, y_test, 'entropy', 'best',
                      maxdepth = 5,
                      clweight = None,
                      minsampleaf = 40)



graph


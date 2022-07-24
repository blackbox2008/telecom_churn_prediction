# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:18:03 2022

@author: Massoud Sharifi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, fbeta_score
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve
import seaborn as sns
import graphviz
import math
import time
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns=25
df = pd.read_csv(r'C:\Users\Massoud\Documents\GitHub\churn_prediction\datasets\churn.csv')
#Clean data
# drop null values 
df.TotalCharges = df.TotalCharges.replace(' ', np.nan)
df = df.dropna(how = 'any')
df["TotalCharges"]=pd.to_numeric(df.TotalCharges,errors='coerce')

# fixing MISCLASSIFICATIONS
df['SeniorCitizen'] = df['SeniorCitizen'].replace({1:'Yes',0:'No'})
numurical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
df.MultipleLines = df.MultipleLines.apply(lambda x: "No" if x =="No phone service" else x)
for col in df:
    if (df[col].dtype=="object") and (df[col].nunique()==3):
         df[col] = df[col].apply(lambda x: "No" if x=="No internet service" else x)

    
df.loc[df['PaymentMethod'] == 'Bank transfer (automatic)', 'PaymentMethod'] = 'Bank transfer' #to shorten values
df.loc[df['PaymentMethod'] == 'Credit card (automatic)', 'PaymentMethod'] = 'Credit card'   
# delete non-relevant columns
del df['Unnamed: 0']
del df['customerID']


fig,ax = plt.subplots(1,3,figsize=(15,5))
for i,x in enumerate(['tenure', 'MonthlyCharges', 'TotalCharges']):
    ax[i].hist(df[x][df.Churn=="No"],label="Churn=0",bins=30, color = 'red', alpha=0.5)
    ax[i].hist(df[x][df.Churn=="Yes"],label="Churn=1", bins=30, color = 'yellow', alpha=0.7)
    ax[i].legend(shadow=True, loc=9)
    
sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], hue='Churn',
             plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)
categorical_variables = [
 'gender',
 'SeniorCitizen',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'PaymentMethod',
 'PaperlessBilling',
 'Contract' ]
ROWS, COLS = 4, 4
fig, ax = plt.subplots(ROWS, COLS, figsize=(25, 25) )
row, col = 0, 0
for i, cat_var in enumerate(categorical_variables):
 if col == COLS - 1: row += 1
 col = i % COLS
 df[df.Churn=='No'][cat_var].value_counts().plot(kind = 'bar', width=.5, ax=ax[row, col], color='blue', alpha=0.5).set_title(cat_var)
 df[df.Churn=='Yes'][cat_var].value_counts().plot(kind ='bar', width=.3, ax=ax[row, col], color='orange', alpha=0.7).set_title(cat_var)
 plt.legend(['No Churn', 'Churn'])
 fig.subplots_adjust(hspace=0.7)
 plt.tight_layout()

#Featue Engineering
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


df['tenure_group'] = df.apply(lambda df:tenure_split(df), axis = 1)
df['totalcharges_group'] = df.apply(lambda df:totalcharges_split(df), axis = 1)

df.to_csv(r'C:\Users\Massoud\Documents\GitHub\churn_prediction\datasets\df.csv', index=False)
 
# To analyse categorical feature distribution
target_var = ['Churn']
# Categorical variables having 4 or less classes
cat_vars = df.nunique()[df.nunique() <= 4].keys().tolist()
cat_vars = [x for x in cat_vars if x not in target_var]
# Numerical variables
num_vars   = [x for x in df.columns if x not in cat_vars + target_var]
# Flag Variables (having 2 classes)
flag_vars   = df.nunique()[df.nunique() == 2].keys().tolist()
#Multinomial Variables ( having more than 2 values)
multinom_vars = [i for i in cat_vars if i not in flag_vars]

#Label encoding Binary columns
le = LabelEncoder()
for i in flag_vars :
    df[i] = le.fit_transform(df[i])
#Duplicating columns for multi value columns
df = pd.get_dummies(data = df, columns = multinom_vars)

#Scaling Numerical variables
std = StandardScaler()
scaled = std.fit_transform(df[num_vars])
scaled = pd.DataFrame(scaled,columns=num_vars)

#dropping original values merging scaled values for numerical columns
df1 = df.drop(columns = num_vars, axis = 1)
df1 = df1.merge(scaled, left_index=True, right_index=True, how = "left")

df1.to_csv(r'C:\Users\Massoud\Documents\GitHub\churn_prediction\datasets\df1.csv', index=False)



# partitioning data to 70:30 ratio for train/test

X = df1.drop(['Churn'], axis =1)
y = df1['Churn']
X.reset_index(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.40, random_state = 7)


#Modeling Phase


# Logistic Regression lrm

print('\n******* LogisticRegression *******')
print('\nSearch for optimal hyperparameter C in LogisticRegresssion, \
      vary C from 0.001 to 1000, using KFold(10) Cross Validation on train data')
start_time = time.time()
kf = KFold(n_splits=10, random_state=7, shuffle=True) 
score_list = []
c_list = 10**np.linspace(-3,3,200)
for c in c_list:
# C is a positive floating-point number (1.0 by default) that defines 
# the relative strength of regularization. Smaller values indicate stronger regularization.
 lrm = LogisticRegression(solver='liblinear', C = c, random_state=7)
 lrm.fit(X_train, y_train)
 cvs = (cross_val_score(lrm, X_train, y_train, cv=kf, scoring='f1')).mean()
 score_list.append(cvs)
 print(f'{cvs:.4f}', end=', ') 
print(f'\noptimal cv F1 score = {max(score_list):.4f}')
optimal_c = float(c_list[score_list.index(max(score_list))])
print(f'optimal value of C = {optimal_c:.3f}')
time1 = time.time()
lrm = LogisticRegression(solver='liblinear', C = optimal_c, random_state=7)
lrm.fit(X_train, y_train)
optimal_th = 0.5   # start with default threshold value
    
for i in range(0,3):
    score_list = []
    print(f'\nLooping decimal place {i+1}') 
    th_list = [np.linspace(optimal_th-0.4999, optimal_th+0.4999, 11), 
               np.linspace(optimal_th-0.1, optimal_th+0.1, 21), 
               np.linspace(optimal_th-0.01, optimal_th+0.01, 21)]
           
    for th in th_list[i]:
        y_pred = lrm.predict_proba(X_test)[:,1] >= th
        f1score = f1_score(y_test, y_pred)
        score_list.append(f1score)
        print(f'{th:.3f}->{f1score:.4f}' , end=',  ')   # display score in 4 decimal places
    optimal_th = float(th_list[i][score_list.index(max(score_list))])

print(f'Optimal F1 score = {max(score_list):.4f}')
print(f'Optimal threshold = {optimal_th:.3f}')

print('accuracy score is:')
print(f'Training: {100 * lrm.score(X_train, y_train):.2f}%')  # score uses accuracy
print(f'Test set: {100 * lrm.score(X_test, y_test):.2f}%')   # should use cross validation
print('***************Logistic Regression Model Summary*****************')
for i in [0.25, 0.50, 0.75, optimal_th]:
    y_pred = lrm.predict_proba(X_test)[:,1] >= i
    lrm_prec_score = precision_score(y_test, y_pred)
    lrm_recall_score = recall_score(y_test, y_pred)
    lrm_f1 = f1_score(y_test, y_pred)
    cm =  confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    lrm_auc = auc(fpr, tpr)
    lrm_logloss = log_loss(y_test, y_pred)
    lrm_roc_auc_score = roc_auc_score(y_test, y_pred)
    print(f'\nAdjust threshold to {i}:')
    if i==optimal_th:
        print('!!!!!! Optimal threshold !!!!!!')
    print(f'Precision: {lrm_prec_score:.4f},   Recall: {lrm_recall_score :.4f},   F1 score: {lrm_f1:.4f}')
    print(f'confusion matrix: \n {cm}')

print(f'Log Loss: { lrm_logloss:.4f}')
print(f'ROC-AUC Score: { lrm_roc_auc_score:.4f}')
print(f'AUC: { lrm_auc:.4f}')
print(f'Time elapsed: {(time.time() - start_time):.2f} seconds')
# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Non-churners', 'Predicted Churners'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Non-Churners', 'Actual Churnerss'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='green')
plt.show()
plt.tight_layout()
# plot the ROC curve
plt.figure(figsize = [6,6])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % lrm_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
plt.tight_layout()
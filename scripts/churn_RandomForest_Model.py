# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:18:03 2022

@author: Massoud Sharifi
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
df.SeniorCitizen = df.SeniorCitizen.apply(lambda x: "No" if x==0 else "Yes")
df.MultipleLines = df.MultipleLines.apply(lambda x: "No" if x =="No phone service" else x)
for col in df:
    if (df[col].dtype=="object") and (df[col].nunique()==3):
         df[col] = df[col].apply(lambda x: "No" if x=="No internet service" else x)


    
df.loc[df['PaymentMethod'] == 'Bank transfer (automatic)', 'PaymentMethod'] = 'Bank transfer' #to shorten value
df.loc[df['PaymentMethod'] == 'Credit card (automatic)', 'PaymentMethod'] = 'Credit card'   

# partitioning data
del df['Unnamed: 0']
del df['customerID']
df.reset_index(inplace=True)

le = LabelEncoder()
for col in df:
    if (df[col].dtype=="object") and (df[col].nunique()==2):
         df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df,columns=[i for i in df.columns if df[i].dtypes=='object'],drop_first=True)

X = df.iloc[:, :-1]
y = df.iloc[:,-1]

scaler = StandardScaler()
scaler.fit_transform(X)

X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size = 0.30, random_state = 7)



model = RandomForestClassifier(criterion='gini', 
                               bootstrap=True, # enabling bootstrapping
                               random_state=0, # random state for reproducibility
                               max_features='sqrt', # number of random features to use sqrt(n_features)
                               min_samples_leaf=70, # minimum no of observarions allowed in a leaf
                               max_depth=5, # maximum depth of the tree
                               n_estimators=100 # how many trees to build
                              )


# Fit the model
rf = model.fit(X_train, y_train)



# Predict class labels on training data
pred_labels_train = model.predict(X_train)
# Predict class labels on a test data
pred_labels_test = model.predict(X_test)



print('---------------- Random Forest Model Summary ----------------')
print('Classes: ', rf.classes_)
print('No. of outputs: ', rf.n_outputs_)
print('No. of features: ', rf.n_features_in_)
print('No. of Estimators: ', len(rf.estimators_))
print('--------------------------------------------------------')
print("")

print('*************** Evaluation on Training Data ***************')
score_tr = model.score(X_train, y_train)
print('Accuracy Score: ', score_tr)
# Look at classification report to evaluate the model
print(classification_report(y_train, pred_labels_train))


print('*************** Evaluation on Test Data ***************')
score_te = model.score(X_test, y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(y_test, pred_labels_test))
print('--------------------------------------------------------')
print("")


dot_data = export_graphviz(rf.estimators_[50], out_file=None, 
                            feature_names=X_train.columns, 
                            class_names=[str(list(rf.classes_)[0]), str(list(rf.classes_)[1])],
                            filled=True, 
                            rounded=True, 
                            rotate=False,
                           ) 
graph = graphviz.Source(dot_data)

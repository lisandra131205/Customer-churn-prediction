#customer churn means customer leaving a company like cancelling a subscription
import pandas as pd
csv=pd.read_csv("dataset.csv")
print(csv)
print(csv.isna().sum())#tells no. of missing values present
csv['gender']=csv['gender'].map({'Female':0,'Male':1})
csv['Partner']=csv['Partner'].map({'Yes':1,'No':0})
csv['Dependents']=csv['Dependents'].map({'Yes':1,'No':0})
csv['PhoneService']=csv['PhoneService'].map({'Yes':1,'No':0})
csv['OnlineSecurity']=csv['OnlineSecurity'].map({'Yes':1,'No':0})
csv['OnlineBackup']=csv['OnlineBackup'].map({'Yes':1,'No':0})
csv['DeviceProtection']=csv['DeviceProtection'].map({'Yes':1,'No':0})
csv['TechSupport']=csv['TechSupport'].map({'Yes':1,'No':0})
csv['StreamingTV']=csv['StreamingTV'].map({'Yes':1,'No':0})
csv['StreamingMovies']=csv['StreamingMovies'].map({'Yes':1,'No':0})
csv['PaperlessBilling']=csv['PaperlessBilling'].map({'Yes':1,'No':0})
"""
binary_cols = ['Partner','Dependents','PhoneService',
               'OnlineSecurity','OnlineBackup','DeviceProtection',
               'TechSupport','StreamingTV','StreamingMovies',
               'PaperlessBilling','Churn']
for col in binary_cols:
    csv[col] = csv[col].map({'Yes':1,'No':0})
"""
csv['Churn']=csv['Churn'].map({'Yes':1,'No':0})
print(csv['Contract'].unique())#to check how many categories
csv['Contract']=csv['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
csv['TotalCharges']=pd.to_numeric(csv['TotalCharges'],errors='coerce')#converts to numbers and black spaces to nan
csv['TotalCharges']=csv['TotalCharges'].fillna(csv['TotalCharges'].median())
a=csv[['tenure','MonthlyCharges','TotalCharges','Contract']]
b=csv['Churn']
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2)
m1=LogisticRegression()
m1.fit(a_train,b_train)
print("prediction=",m1.predict(a_test)[:3])#here prediction is [0 0 0] indicating customer won't churn
print("accuracy=",accuracy_score(b_test,m1.predict(a_test)))#here we are using accuracy because it is a classification problem
print(m1.coef_)
print(a.columns)
a1=confusion_matrix(b_test,m1.predict(a_test))
a2=classification_report(b_test,m1.predict(a_test))
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(a1,annot=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

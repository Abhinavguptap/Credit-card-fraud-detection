import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv('/content/creditcard2.csv')
# Display the first few rows of the dataset
data.head()
data.tail()
data.info()
data.isnull().sum()
data_filled_with_0 = data.fillna(0)
print("\nDataset after filling null values with 0:")
print(data_filled_with_0)
print("\nNull values in the dataset after filling:")
print(data_filled_with_0.isnull().sum())
data_filled_with_0['Class'].value_counts()
normal = data_filled_with_0[data_filled_with_0.Class==0]
fraud = data_filled_with_0[data_filled_with_0.Class==1]
normal_count = len(data_filled_with_0[data_filled_with_0['Class'] == 0])
fraud_count = len(data_filled_with_0[data_filled_with_0['Class'] == 1])
categories = ['Normal Transactions', 'Fraudulent Transactions']
counts = [normal_count, fraud_count]
plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['blue', 'red'], edgecolor='black', alpha=0.7)
plt.title('Count of Normal vs Fraudulent Transactions')
plt.ylabel('Number of Transactions')
plt.xlabel('Transaction Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print(normal.shape)
print(fraud.shape)
#statistical measures
normal.Amount.describe()
fraud.Amount.describe()
data_filled_with_0.groupby('Class').mean()
normal_sample = normal.sample(n=492)
n_dataset = pd.concat([normal_sample,fraud],axis=0)
n_dataset.head(n=10)
n_dataset.tail()
n_dataset['Class'].value_counts()
n_dataset.groupby('Class').mean()
A=n_dataset.drop(columns='Class',axis=1)
P=n_dataset['Class']
print(A)
print(P)
A_train,A_test,P_train,P_test=train_test_split(A,P,test_size=0.2,stratify=P,random_state=2)
print(A.shape,A_train.shape,A_test.shape)
 model = LogisticRegression()
model.fit(A_train, P_train)
A_train_prediction=model.predict(A_train)
training_data_accuracy=accuracy_score(A_train_prediction,P_train)
print("Acurracy on Training data:",training_data_accuracy)
A_test_prediction=model.predict(A_test)
test_data_accuracy=accuracy_score(A_test_prediction,P_test)
print("Accuracy score on Test Data:",test_data_accuracy)
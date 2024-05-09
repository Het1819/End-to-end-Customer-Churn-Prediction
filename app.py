import numpy as np
import pandas as pd

df = pd.read_csv('Bank Customer Churn Prediction.csv')

df.head()
Label Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['country'] = le.fit_transform(df['country'])

df.head(3)
Six Questions

df.shape
(10000, 12)

df.isnull().sum()
 
df.duplicated().sum()
0

df.info()
 
df.describe()
EDA

corr = df.corr()

corr

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True,cbar=True,cmap='coolwarm')
<AxesSubplot:>

[45]
df.info()
< 
 
[46]
sns.histplot(df['credit_card'],bins=20)
<AxesSubplot:xlabel='credit_card', ylabel='Count'>

[47]
sns.boxplot(df['age'])
<AxesSubplot:>

[51]
sns.countplot(x=df['gender'])
<AxesSubplot:xlabel='gender', ylabel='count'>

Train Test Split
[52]
from sklearn.model_selection import train_test_split
[54]
X = df.drop(['churn','customer_id'],axis=1)
y = df['churn']
[55]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Standrization
[67]
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
[68]
X_train = sclr.fit_transform(X_train)
X_test = sclr.fit_transform(X_test)
training model
[69]
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import accuracy_score
[70]
models = {
    'lg':LogisticRegression(),
    'dtc':DecisionTreeClassifier(),
    'rfc':RandomForestClassifier(),
}
[71]
for name,model in models.items():
    model.fit(X_train,y_train)
    ypred = model.predict(X_test)
    print(f"{name} with accuracy : {accuracy_score(y_test,ypred)} ")
lg with accuracy : 0.8155 
dtc with accuracy : 0.788 
rfc with accuracy : 0.871 
model selection
[75]
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
ypred = model.predict(X_test)
prediction system
[81]
def prediction(credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary):
    features = np.array([[credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary]])
    features = sclr.fit_transform(features)
    prediction = rfc.predict(features).reshape(1,-1)
    return prediction[0]

credit_score = 608
country = 2
gender = 0
age= 41
tenure= 1
balance = 83807.86
products_number= 1
credit_card = 0
active_member =1
estimated_salary = 112542.58

pred  = prediction(credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary)
[84]
if pred == 1:
    print("he left the compnay")
else:
    print("he is there still")
he is there still
[85]
import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(rfc,open('rfc.pkl','wb'))
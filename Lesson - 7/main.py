import numpy as np
import pandas as pd

data=pd.read_csv("Data.csv")
print(data.head())
print(data.info())
#INDENTFYING FEATYRES N TARGETZ

#picked everthing last
X= data.iloc[:,:-1].values
#Picked only last column
y=data.iloc[:,-1].values

print(y)
print(X)

#DATA PREPROCCESS -- WE FLL EMPTY DATA WITH AVERAGE DATA OF THE COLOMn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

X[:,1:3]=imputer.fit_transform(X[:,1:3])
print("After Imputing:\n",X)
print()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(), [0])], remainder="passthrough")
X=pd.DataFrame(ct.fit_transform(X))
print("One hot encoding:\n",X)
print()

from sklearn.preprocessing import LabelEncoder
#y-1 n - 0
le=LabelEncoder()
y=le.fit_transform(y)
print("LABEL ENCODER :\n",y)
print()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
print("XTRAIN : \n",X_train)
print("XTEST : \n",X_test)
print()

print("yTRAIN : \n",y_train)
print("yTEST : \n",y_test)
print()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train.iloc[:,1:3]=sc.fit_transform(X_train.iloc[:,1:3])
X_test.iloc[:,1:3]=sc.transform(X_test.iloc[:,1:3])

print("AFTER SCALING THE VALUES FROM -1 to 1 :\n")
print(X_train)
print(X_test)

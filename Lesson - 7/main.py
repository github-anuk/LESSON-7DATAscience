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
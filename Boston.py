
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

lr=LinearRegression()

df=pd.read_csv("C:/Users/Riya/Downloads/HousingData.csv")

#print(df)

df["INDUS"].fillna(0, inplace = True)
df["NOX"].fillna(0, inplace = True)
df["RM"].fillna(0, inplace = True)
df["DIS"].fillna(0, inplace = True)
df["RAD"].fillna(0, inplace = True)
df["TAX"].fillna(0, inplace = True)
df["PTRATIO"].fillna(0, inplace = True)
df["B"].fillna(0, inplace = True)
df["AGE"].fillna(0, inplace = True)
df["LSTAT"].fillna(0, inplace = True)
df["CRIM"].fillna(0, inplace = True)
df["MEDV"].fillna(0.0, inplace = True)
df["ZN"].fillna(0, inplace = True)
df["CHAS"].fillna(0, inplace = True)
#print(df)

#print(df.describe())
#print(df.loc[:,["CHAS"]])

x=df.drop("MEDV",axis=1)
y=df["MEDV"]

#print(x)
#print(y)


x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, test_size=0.2)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)


lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)
#print(y_pred)

print(mean_squared_error(y_test, y_pred))

#print(df)
#print(df.dtypes)

'''
OUTPUT :  
36.478011752517276

Process finished with exit code 0
'''

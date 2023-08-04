#Black Friday Sales Prediction

import pandas as pd
import numpy as np

df=pd.read_csv("C:/Users/Riya/Downloads/train.csv")
#df.info()
#print(df.describe())

#print(round((df.isnull().sum() / df.shape[0]) * 100, 2).astype(str))

df['Age'] = df['Age'].apply(lambda x : str(x).replace('55+', '55'))
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].apply(lambda x : str(x).replace('4+', '4'))

df.drop('Product_Category_3', axis = 1, inplace = True)
df.drop('User_ID', axis = 1, inplace = True)
df.drop('Product_ID', axis = 1, inplace = True)

from sklearn.preprocessing import LabelEncoder
leg = LabelEncoder()
df['Gender'] = leg.fit_transform(df['Gender'])
lea = LabelEncoder()
df['Age'] = lea.fit_transform(df['Age'])
lec = LabelEncoder()
df['City_Category'] = lec.fit_transform(df['City_Category'])

df['Product_Category_2'].fillna(df['Product_Category_2'].median(), inplace = True)
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype('int')

x = df.drop("Purchase", axis = 1)
y = df["Purchase"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1 ,test_size = 0.2)
#print("x_train shape:", x_train.shape)
#print("x_test shape:", x_test.shape)
#print("y_train shape:", y_train.shape)
#print("y_test shape:", y_test.shape)

#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# lr = LinearRegression()
# lr.fit(x_train, y_train)
# y_pred= lr.predict(x_test)


# dt = DecisionTreeRegressor()
# dt.fit(x_train, y_train)
# y_predd= dt.predict(x_test)

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_predf = rf.predict(x_test)

# print("Linear Regression ")
# print("Root mean square error : ",np.sqrt(mean_squared_error(y_test, y_pred)))
# print("R2 score:", r2_score(y_test, y_pred))

# print("Decision tree regression ")
# print("Root mean square error : ",np.sqrt(mean_squared_error(y_test, y_predd)))
# print("R2 score:", r2_score(y_test, y_predd))

print("Random forest regression ")
print("Root mean sqaure error : ",np.sqrt(mean_squared_error(y_test, y_predf)))
print("R2 score:", r2_score(y_test, y_predf))

'''
OUTPUT 
Random forest regression 
Root mean sqaure error :  3000.550507920106
R2 score: 0.6444868034759783

Process finished with exit code 0
'''

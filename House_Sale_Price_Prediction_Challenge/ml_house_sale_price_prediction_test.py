import os
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(housing_csv):
  csv_path = os.path.join('dataset/',housing_csv)
  return pd.read_csv(csv_path)

train_originData = load_data('train-v3.csv')
valid_originData = load_data('valid-v3.csv')
test_originData = load_data('test-v3.csv')

train_Data = train_originData
valid_Data = valid_originData
test_Data = test_originData

train_originData.head(5)

train_Data.loc[train_Data['yr_renovated'] >0 ,'yr_built'] = train_Data['yr_renovated']
valid_Data.loc[valid_Data['yr_renovated'] >0 ,'yr_built'] = valid_Data['yr_renovated']
test_Data.loc[test_Data['yr_renovated'] >0 ,'yr_built'] = test_Data['yr_renovated']

train_Data = train_Data[train_Data['bedrooms'] < 11]
train_Data = train_Data[train_Data['sqft_living'] < 12000]
train_Data = train_Data[train_Data['sqft_lot15'] < 800000]

#dropData = {'id','sale_yr','sale_month','sale_day','zipcode','view','waterfront','yr_renovated'}
dropData = {'id'}

train_Data = train_originData.drop(dropData,axis=1)
valid_Data = valid_originData.drop(dropData,axis=1)
test_Data = test_originData.drop(dropData,axis=1)

x_train = train_Data.drop(['price'],axis=1)
y_train = train_Data['price'].values
x_valid = valid_Data.drop(['price'],axis=1)
y_valid = valid_Data['price'].values
x_test = test_Data

train_Data.corr()["price"].sort_values(ascending=False)

scaler = StandardScaler().fit(x_train)
X_train = scaler.transform(x_train)
X_valid = scaler.transform(x_valid)
X_test = scaler.transform(x_test)

filepath = "./bestModel.h5"
model = load_model(filepath)

final_result=model.predict(X_test)

final_result=np.reshape(final_result,(len(test_Data),))
print(final_result)

with open('submit_results.csv', 'w') as f:
    f.write('Id,price\n')
    for i in range(len(final_result)):
        f.write(str(i+1) + ',' + str(final_result[i]) + '\n')

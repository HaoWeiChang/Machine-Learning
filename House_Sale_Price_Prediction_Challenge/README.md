# 房價預測-110368003 張浩維

## 目錄

- [房價預測-110368003 張浩維](#房價預測-110368003-張浩維)
  - [目錄](#目錄)
  - [檔案介紹](#檔案介紹)
  - [使用模組](#使用模組)
  - [train.sh 、 test.sh 內容講解](#trainsh--testsh-內容講解)
  - [.ipynb 執行方式](#ipynb-執行方式)
- [房價預測-講解](#房價預測-講解)
  - [載入資料](#載入資料)
  - [圖像處理](#圖像處理)
  - [資料預處理](#資料預處理)
  - [建立模型](#建立模型)
  - [訓練結果與驗證曲線](#訓練結果與驗證曲線)
  - [最終預測 Test 資料](#最終預測-test-資料)
  - [繳交作業預測成績](#繳交作業預測成績)

---

## 檔案介紹

1. ML_House_Sale_Price_Prediction.ipynb --(使用 Colab 執行)
2. ml_house_sale_price_prediction.py --(負責 Training)
3. ml_house_sale_price_prediction_test.py --(Training 產出的 bestModel.h5，執行 Test 預測)
4. requirement.txt --(本次運用到的模組)
5. train.sh --(執行自動安裝 python3、pip3、module，並執行 Training python)
6. test.sh --(執行 ml_xxx_XX_test.py，產出預測結果 csv 檔)

## 使用模組

<h3>requirement.txt </h3>

> pandas 、 numpy 、 tensorflow 、 matplotlib  
> keras 、 scikit-learn 、 mlxtend 、 seaborn

## train.sh 、 test.sh 內容講解

<h3>train.sh</h3>

```sh
#!/bin/bash
#下載python3
sudo apt-get install python3.7
python --version
#下載pip3
sudo apt install python3-pip
pip3 --version
#下載module
pip3 install pandas
pip3 install numpy
pip3 install tensorflow
pip3 install matplotlib
pip3 install seaborn
pip3 install keras
pip3 install scikit-learn
pip3 install mlxtend
#執行房價預測Training
python3 ml_house_sale_price_prediction.py
```

<h3>test.sh</h3>

```sh
#!/bin/bash
#執行房價Test結果
python3 ml_house_sale_price_prediction_test.py
```

## .ipynb 執行方式

> 如何執行.ipynb  
> 進入 https://colab.research.google.com/?utm_source=scs-index  
> 上傳 ML_House_Sale_Price_Prediction.ipynb  
> 將 dataset 資料夾上傳至 Google Drive
> 依據資料夾的路徑更改.ipynb 內讀取資料的路徑

```python
#Ex : 依據dataset存放路徑，修改路徑
def load_data(housing_csv):
  csv_path = os.path.join('/content/drive/MyDrive/MachineLearn/house-regression/',housing_csv)
  return pd.read_csv(csv_path)
```

# 房價預測-講解

> 前言:  
> 　這次房價預測皆使用 Colab 執行房價預測，  
> 　對於電腦資源的使用，尚未到一定程度且 colab 具備方便套件，  
> 　對於新手來說非常輕鬆上手，包含一些資料分析顯示。

## 載入資料

> .ipynb 使用的是針對 google drive 路徑修改。  
> .py 使用同層路徑下資料夾路徑。

```python
#.ipynb
def load_data(housing_csv):
  csv_path = os.path.join('/content/drive/MyDrive/MachineLearn/house-regression/',housing_csv)
  return pd.read_csv(csv_path)
#.py
def load_data(housing_csv):
  csv_path = os.path.join('dataset/',housing_csv)
  return pd.read_csv(csv_path)

train_originData = load_data('train-v3.csv')
valid_originData = load_data('valid-v3.csv')
test_originData = load_data('test-v3.csv')
```

## 圖像處理

```python
train_originData.head()
```

<img src="README_img\2021-11-28 161940.jpg" />

```python
plt.figure(figsize = (8,10))
train_originData.corr()["price"].sort_values().drop("price").plot(kind = "barh");
```

<img src="README_img\2021-11-28 163220.jpg" />

```python
plt.hist(train_originData['price'],bins=30)
plt.show()
```

<img src="README_img\2021-11-28 163446.jpg" />

```python
var = 'sqft_living'
data = pd.concat([train_originData['price'], train_originData[var]], axis=1)
data.plot.scatter(x=var, y='price',xlim=(0,25000), ylim=(0,8500000));
```

<img src="README_img\2021-11-28 163503.jpg"/>

```python
var = 'zipcode'
data = pd.concat([train_originData['price'], train_originData[var]], axis=1)
data.plot.scatter(x=var, y='price',xlim=(97990,98210), ylim=(0,8000000));
```

<img src="README_img\2021-11-28 163545.jpg"/>

```python
var = 'long'
data = pd.concat([train_originData['lat'], train_originData[var]], axis=1)
data.plot.scatter(x=var, y='lat',xlim=(-122.55,-121.28), ylim=(47.1,47.85),alpha = 0.1);
```

<img src="README_img\2021-11-28 164029.jpg"/>

```python
sns.boxplot(x = 'sale_month', y = 'price', data = train_originData);
```

<img src="README_img\2021-11-28 164123.jpg"/>

```python
train_originData.groupby('sale_month')['price'].mean().plot();
```

<img src="README_img\2021-11-28 164137.jpg"/>

```python
cols = ['sale_yr','sale_month','sale_day','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15','price']
scatterplotmatrix(train_originData[cols].values.astype(float), figsize=(50,50), names=cols, alpha=0.5)
plt.tight_layout()
plt.show()
```

<img scr="README_img\2021-11-28 164200.jpg" />

## 資料預處理

> 將整修年份與建造年份合併，有整修的年份則覆蓋建造年份，讓那筆資料看似年份較新。

```python
train_Data.loc[train_Data['yr_renovated'] >0 ,'yr_built'] = train_Data['yr_renovated']
valid_Data.loc[valid_Data['yr_renovated'] >0 ,'yr_built'] = valid_Data['yr_renovated']
test_Data.loc[test_Data['yr_renovated'] >0 ,'yr_built'] = test_Data['yr_renovated']
```

> 藉由上面的圖像處理，能觀察出 bedroom 欄位，有一筆 33 的值，  
> 在此我們先預估為錯誤值，為了資料的訓練，我將它移除。  
> spft_living、spft_lot15 也皆為相同情形，因此都先將錯誤值那列移除。

```python
train_Data = train_Data[train_Data['bedrooms'] < 11]
train_Data = train_Data[train_Data['sqft_living'] < 12000]
train_Data = train_Data[train_Data['sqft_lot15'] < 800000]
```

> 原先將一些看似無幫助的資訊移除，但對於訓練來說，看似容易過早 overfitting，
> 訓練出來的結果皆不竟理想，因此最終僅移除 id 來做訓練。

```python
#dropData = {'id','sale_yr','sale_month','sale_day','zipcode','view','waterfront','yr_renovated'}
dropData = {'id'}

train_Data = train_originData.drop(dropData,axis=1)
valid_Data = valid_originData.drop(dropData,axis=1)
test_Data = test_originData.drop(dropData,axis=1)
```

> 將訓練資料與預測價格答案分別開

```python
x_train = train_Data.drop(['price'],axis=1)
y_train = train_Data['price'].values
x_valid = valid_Data.drop(['price'],axis=1)
y_valid = valid_Data['price'].values
x_test = test_Data
```

> 正規化處理  
> 使用 StandardScaler，此正規化是為了能讓數值範圍差距較大的，  
> 縮小為 0~1 之間，讓 Training，能準確預估。

```python
scaler = StandardScaler().fit(x_train)
X_train = scaler.transform(x_train)
X_valid = scaler.transform(x_valid)
X_test = scaler.transform(x_test)
```

## 建立模型

> 本次使用 checkpoint、EarlyStopping
>
> 1. checkpoint：可以將每次的 Epoch 訓練最好的預測準確 Model 儲存起來。
> 2. EarlyStopping：本次訓練將 epoch 設為 1000，
>    倘若 Val_loss 開始不再下降時，可以提早結束

```python
filepath="./bestModel.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
es = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
```

> 建立模型，使用 relu 作為激活函數

```python
model = keras.Sequential()
model.add(layers.Dense(64,kernel_initializer='normal',activation='relu',input_shape=(X_train.shape[1],)))
model.add(layers.Dense(48,kernel_initializer='normal',activation='relu'))
model.add(layers.Dense(32,kernel_initializer='normal',activation='relu'))
model.add(layers.Dense(1,kernel_initializer='normal'))
model.summary()
```

<img src="README_img\2021-11-28 214906.jpg">

## 訓練結果與驗證曲線

<img src="README_img\2021-11-28 215021.jpg">

## 最終預測 Test 資料

> 最後匯入最好的 Model.h5，進行 Test 預測，將結果輸出成 CSV，就可以繳交本次成績

```python
from keras.models import load_model
filepath = "./bestModel.h5"
model = load_model(filepath)

final_result=model.predict(X_test)

final_result=np.reshape(final_result,(len(test_Data),))
print(final_result)

with open('submit_results.csv', 'w') as f:
    f.write('Id,price\n')
    for i in range(len(final_result)):
        f.write(str(i+1) + ',' + str(final_result[i]) + '\n')
```

## 繳交作業預測成績

> 此為 Private Leaderboard 顯示成績
> <img src="README_img\2021-11-28 215322.jpg">

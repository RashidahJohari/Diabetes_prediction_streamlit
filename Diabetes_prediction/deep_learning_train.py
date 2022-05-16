# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:48:18 2022

@author: Acer

Classification problem: Diabetes or Not Diabetes

"""

#%% Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense,Flatten,Dropout,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import datetime
import pickle


#%%
# statics/constants here
DATASET_PATH = os.path.join(os.getcwd(),'Database','diabetes.csv') 
LOG_PATH = os.path.join(os.getcwd(),'log_diabetes')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_path','model.h5')
MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_path','mms_scaler.pkl')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'saved_path', 'ohe.pkl')
#%% EDA
# Step 1) Data loading
# To load data
df = pd.read_csv(DATASET_PATH)

# Step 2) Data interpretation/inspection
# to check first 10 rows
print(df.head(10))

# to check number of entries
print (df.shape)

# to check outlier (mean & median comparison)
print(df.describe().T)
#print(df.groupby('Outcome').hist(figsize=(9,9)))

# to check null
print(df.info())

# to check missing data
#print(df.isna().sum())
print(df.isnull().sum())

# to check duplicate values
print(df[df.duplicated()])

# to sort values
print(np.sort(df))

# Step 3) Data cleaning
# to check duplicate value
#print(df.drop_duplicates())
# inplace=True to remove duplicates from the original DataFrame.
print(df.drop_duplicates(inplace=True))
# to check back if still containing duplicate valueS
print(df.duplicated().sum())

# a) to convert all string data into numerical data
# b) to remove NaN and to impute using some approaches
# b(i) drop NaN data
# b(ii) imputed NaN with mean/median/interpolation : median(recommended)
dummy_df = df.copy()
imputer= SimpleImputer(strategy='median')
dummy_df_imputed  = imputer.fit_transform(dummy_df)

# b(iii) imputed NaN using KNN approach
# dummy_df = df.copy()
# imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
# dummy_df_imputed = imputer.fit_transform(dummy_df)

# to visualize NaN
# converting numpy array back to DataFrame
dummy_df_imputed = pd.DataFrame(dummy_df_imputed, columns=df.columns) #convert numpy array to DF
msno.matrix(dummy_df_imputed)

# To check if all NaNs have removed
dummy_df_imputed.isna().sum() 
dummy_df_imputed.describe().T
dummy_df_imputed.info()

#%% Step 4) Features selection (correlation/lasso:regression/pca:classification)
# Method a : Selecting features using correlation values
# a) Correlation
# data = pd.DataFrame(dummy_df_imputed)
# cor = data.corr()

# plt.figure(figsize=(12,10))
# sns.heatmap(cor,annot=True, cmap=plt.cm.Reds)
# plt.show()

# Conclusion : Glucose, BMI features have relation with target
#%% Step 5) Data preprocessing (keluar data yg nak guna shj & concat)
# i-convert all non-numerical/string to number using:
# a) np.to_numeric
# b) label encoder
# c) one hot encoder
# ii-Scaling data using:
# a) standardscaler()
# b) MinMaxScaler
# c) Robustscaler

X = dummy_df_imputed.iloc[:,:-1]
y = dummy_df_imputed.iloc[:,-1]

# by using MinMaxScaler
mms_scaler = MinMaxScaler()
X_scaled = mms_scaler.fit_transform(X)
# save the scaler
pickle.dump(mms_scaler, open(MMS_SAVE_PATH, 'wb'))

# one hot encoding for the target
ohe_scaler = OneHotEncoder(sparse=False)
y_ohe_scaler = ohe_scaler.fit_transform(np.expand_dims(y,axis=-1)) # y train
# save the one hot encoding
pickle.dump(ohe_scaler, open(OHE_SAVE_PATH, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_ohe_scaler,test_size=0.2)

#%% Model training (deep learning/machine learning)
# Deep learning (Sequential/ functional api)

# Method 1: Sequential api
model = Sequential(name=('diabetes'))

# placing items into your container
model.add(Input(shape=(8), name='input_layer')) # 
model.add(Dense(128, activation='relu', name='hidden_layer_1')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', name='hidden_layer_2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax', name='output_layer')) #relu: ada value
model.summary()



#%% Step Callbacks
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# Tensorboard Callbacks
tensorboard_callback = TensorBoard(log_dir=log_files)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3) 

#%% compile and model fitting
# to wrap the container
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

hist = model.fit(X_train,y_train,epochs=100, validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])

#%% visualize graph using matplotlib

print(hist.history.keys())
keys = [i for i in hist.history.keys()]

training_loss = hist.history[keys[0]] # training loss
training_metric = hist.history[keys[1]] # training metric
validation_loss = hist.history[keys[2]]
validation_metric = hist.history[keys[3]]

plt.figure()
plt.plot(training_loss,label='training {}'.format(keys[0])) # during training
plt.plot(validation_loss,label='training {}'.format(keys[2])) # validation loss
plt.title('training {} and validation {}'.format(keys[0], keys[2]))
plt.xlabel('epoch')
plt.ylabel(keys[0])
plt.legend()
plt.show()

plt.figure()
plt.plot(training_metric, label='training {}'.format(keys[1])) # during training
plt.plot(validation_metric,label='training {}'.format(keys[3])) # validation loss
plt.title('training {} and validation {}'.format(keys[1], keys[3]))
plt.xlabel('epoch')
plt.ylabel(keys[1])
plt.legend()
plt.show()

#%% Model evaluation
pred_x = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(pred_x, axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true, y_pred))

#%% Model deployment
model.save(MODEL_SAVE_PATH)

# deeplearning: model.save(PATH)
# machinelearning: pickle.dump(model,open(MODEL_SAVE_PATH))

#%%
# tensorboard --logdir C:\Users\Acer\Desktop\RNN\sentiment_analysis\log
# localhost:6006




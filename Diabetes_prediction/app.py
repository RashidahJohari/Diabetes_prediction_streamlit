# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:00:10 2022

@author: Acer
"""

from tensorflow.keras.models import load_model
import os
import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('CPU') # GPU
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
    
#%% statics/constants here
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_path','model.h5')
MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_path','mms_scaler.pkl')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'saved_path', 'ohe.pkl')

#%% loading of settings or models
ohe_scaler = pickle.load(open(OHE_SAVE_PATH, 'rb'))
mms_scaler = pickle.load(open(MMS_SAVE_PATH, 'rb'))

# if you are using machine learning
# model = pickle.load(open(PATH))

# if you are using deep learning
model = load_model(MODEL_SAVE_PATH)
model.summary()

diabetes_chance = {0:'negative', 1:'positive'}

#%% test deployment

patient_info = np.array([5,116,74,0,0,25.6,0.201,30]) # true label=0
patient_info_scaled = mms_scaler.transform(np.expand_dims(patient_info,axis=0))

outcome = model.predict(patient_info_scaled)
# print(np.argmax(outcome))
# print(diabetes_chance[np.argmax(outcome)])

# another approach
if np.argmax(outcome) ==1:
    outcome = [0,1]
    print(ohe_scaler.inverse_transform(np.expand_dims(patient_info,axis=0)))
else:
    outcome = [1,0]
    print(ohe_scaler.inverse_transform(np.expand_dims(outcome,axis=0)))
          
#%% build your app using streamlit
with st.form('Diabetes Prediction Form'):
    st.write("Patient's info")
    pregnancies = int(st.number_input('Insert Time of Pregnancies'))
    glucose = st.number_input('Glucose')
    bp = st.number_input('Blood Pressure')
    skin_thick = st.number_input('Skin Thickness')
    insulin_level = st.number_input('Insulin level')
    bmi = st.number_input('bmi')
    diabetes = st.number_input('diabetes')
    age = int(st.number_input('age'))
    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        patient_info = np.array([pregnancies,glucose,bp,skin_thick,
                                 insulin_level,bmi,diabetes,age])
        patient_info_scaled = mms_scaler.transform(np.expand_dims(patient_info,
                                                                  axis=0)) 
        outcome = model.predict(patient_info_scaled)
        
        st.write(diabetes_chance[np.argmax(outcome)])
        
        if np.argmax(outcome)==1:
            st.warning('You going to get diabetes soon, GOOD LUCK')
        else:
            st.balloons()
            st.success('YEAH, you are diabetic free')

        
    
    
    
    
    
    
    
    
    







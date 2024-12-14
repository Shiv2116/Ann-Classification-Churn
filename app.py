import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
## load the model

model = tf.keras.models.load_model('model.h5')

##load encoders and scalers

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

with open('onehotencoder_geography.pkl','rb') as f:
    onehotencoder_geography = pickle.load(f)

with open('label_encoder_gender.pkl','rb') as f:
    label_encoder = pickle.load(f)


# streamlit app
st.title('Churn Prediction App')

## create a form 

geography = st.selectbox('Geography',onehotencoder_geography.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.slider('CreditScore',0,900)
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': label_encoder.transform([gender]),
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## One-hot encode Geography

geo_encoded = onehotencoder_geography.transform([[(geography)]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded, columns = onehotencoder_geography.get_feature_names_out(['Geography']))

## combine

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

## scale the data
input_data_scaled = scaler.transform(input_data)

## make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write('Churn Probability: {:.2f}%'.format(prediction_proba * 100))
    st.write('Customer is likely to Churn')
else:
    st.write('Churn Probability: {:.2f}%'.format(prediction_proba * 100))
    st.write('Customer is not likely to Churn')
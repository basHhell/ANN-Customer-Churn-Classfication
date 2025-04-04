import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import joblib


# Model 

pred_model=load_model("new_model.keras")

#Load Encoder and Scaler

with open("label_encoder.pkl",'rb') as file:
    label_encoder=pickle.load(file)

with open("one_hot_encoder.pkl",'rb') as file:
    one_hot_encoder=pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file) 

title=st.title("Customer Churn Prediction")


gender=st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('CreditScore')
estimated_salary=st.number_input('EstimatedSalary')
tenure=st.slider('Tenure',0,10)
no_of_product=st.slider('NumOfProducts',1,4)
has_cred_card=st.selectbox('HasCrcard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])


input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[no_of_product],
    'HasCrCard':[has_cred_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]

})

label_encoder=LabelEncoder()
input_data["Gender"]=label_encoder.fit_transform(input_data["Gender"])
input_data
# geo_encoder=one_hot_encoder.fit_transform([["Geography"]]).toarray()
# geo_encoder

# # one_hot_encoder.get_feature_names_out(["Geography"])
# after_encoding=pd.DataFrame(geo_encoder,columns=one_hot_encoder.get_feature_names_out(["Geography"]))

# new_data=pd.concat([input_data.reset_index(drop=True),after_encoding],axis=1)
# new_data=pd.concat([input_data.drop('Geography',axis=1),after_encoding],axis=1)


scaled_data=scaler.transform(input_data)

prediction=pred_model.predict(scaled_data)
prediction=prediction[0][0]
st.write(prediction)

if prediction > 0.5:
    st.write("Customer will churn")
else:
    st.write("Customer will not churn")







































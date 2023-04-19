import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np

model = pickle.load(open('model.sav', 'rb'))

st.title('Stroke Prediction')
st.sidebar.header('Paitent Stroke Data')

# FUNCTION
def user_report():
  Gender = st.sidebar.slider('Gender', 'Male' == 0,'Female' == 1, 1 )
  Age = st.sidebar.slider('Age', 0,100, 1 )
  hypertension = st.sidebar.slider('hypertension', 0,1, 1 )
  heart_disease = st.sidebar.slider('heart_disease', 0,1, 1 )
  ever_married = st.sidebar.slider('ever_married', 0,2, 1 )
  work_type = st.sidebar.slider('work_type', 0,4, 1)
  Residence_type = st.sidebar.slider('Residence_type', 1,3, 1)
  avg_glucose_level = st.sidebar.slider('avg_glucose_level', 1,500, 1)
  bmi = st.sidebar.slider('bmi', 0,50, 1 )
  smoking_status = st.sidebar.slider('smoking_status', 0,3, 1 )


  user_report_data = {
      'Gender':Gender,
      'Age':Age,
      'hypertension':hypertension,
      'heart_disease':heart_disease,
      'ever_married':ever_married,
      'work_type':work_type,
      'Residence_type':Residence_type,
      'avg_glucose_level':avg_glucose_level,
      'bmi':bmi,
      'smoking_status':smoking_status	
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Paitent Data')
st.write(user_data)

stroke = model.predict(user_data)
st.subheader('Stroke Prediciton')
st.subheader('$'+str(np.round(stroke[0], 2)))
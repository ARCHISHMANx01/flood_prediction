# Import necessary libraries
import streamlit as st
import numpy as np
import pandas  as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import pickle
#from sklearn.linear_model import LogisticRegression

# Logo [optionnal]
#st.image("gambar anime.webp", use_column_width=True)

# create streamlit interface, some info about the app
# '''

st.write("""
         ### To predict your heart disease status:
         ###### 1- Enter parameters according to your health condition in this sidebar.
         ###### 2- Press the "Predict" button and wait for the results.
         """)

#st.subheader("Created by Wahyu Ikbal Maulana From SDT B")

# st.write(BMIdata)

# Sidebar input
# -------------------------------------------------------------------------
st.sidebar.title('Answer the following questions')

BMI=st.sidebar.selectbox("What is your BMI?", ("Normal weight BMI  (18.5-25)", 
                             "under-weight BMI (< 18.5)" ,
                             "over-weight BMI (25-30)",
                             "Obese BMI (> 30)"))
Age=st.sidebar.selectbox("Select year range", 
                            ("18-24", 
                             "25-29" ,
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "80 or older"))

Race=st.sidebar.selectbox("Choose your race", ("Asian", 
                             "Black" ,
                             "Hispanic",
                             "American Indian/Alaskan Native",
                             "White",
                             "Other"
                             ))

Gender=st.sidebar.selectbox("Enter your gender", ("Female", 
                             "Male" ))
Smoking = st.sidebar.selectbox("Have you smoked more than 100 cigarettes in"
                          " your entire life ?)",
                          options=("No", "Yes"))
alcoholDink = st.sidebar.selectbox("Do u drink alcohol very often?", options=("No", "Yes"))
stroke = st.sidebar.selectbox("Ever had a stroke?", options=("No", "Yes"))

sleepTime = st.sidebar.number_input("How many hours do you sleep a day?", 0, 24, 7) 

genHealth = st.sidebar.selectbox("Your health?",
                             options=("Good","Excellent", "Fair", "Very good", "Poor"))

physHealth = st.sidebar.number_input("What is your physical score? (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
mentHealth = st.sidebar.number_input("Your Mental health  (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
physAct = st.sidebar.selectbox("Do you exercise often?"
                           , options=("No", "Yes"))

diffWalk = st.sidebar.selectbox("Are you having difficulty walking"
                            "or climb the stairs?", options=("No", "Yes"))
diabetic = st.sidebar.selectbox("Ever had diabetes?",
                           options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
asthma = st.sidebar.selectbox("Have a history of asthma?", options=("No", "Yes"))
kidneyDisease= st.sidebar.selectbox("Have a history of kidney disease?", options=("No", "Yes"))
skinCancer = st.sidebar.selectbox("Have a history of cancer?", options=("No", "Yes"))

dataToPredic = pd.DataFrame({
   "BMI": [BMI],
   "Smoking": [Smoking],
   "AlcoholDrinking": [alcoholDink],
   "Stroke": [stroke],
   "PhysicalHealth": [physHealth],
   "MentalHealth": [mentHealth],
   "DiffWalking": [diffWalk],
   "Sex": [Gender],
   "AgeCategory": [Age],
   "Race": [Race],
   "Diabetic": [diabetic],
   "PhysicalActivity": [physAct],
   "GenHealth": [genHealth],
   "SleepTime": [sleepTime],
   "Asthma": [asthma],
   "KidneyDisease": [kidneyDisease],
   "SkinCancer": [skinCancer]
 })

# Mapping the data as explained in the script above

dataToPredic.replace("under-weight BMI (< 18.5)",0,inplace=True)
dataToPredic.replace("Normal weight BMI  (18.5-25)",1,inplace=True)
dataToPredic.replace("over-weight BMI (25-30)",2,inplace=True)
dataToPredic.replace("Obese BMI (> 30)",3,inplace=True)

dataToPredic.replace("Yes",1,inplace=True)
dataToPredic.replace("No",0,inplace=True)
dataToPredic.replace("18-24",0,inplace=True)
dataToPredic.replace("25-29",1,inplace=True)
dataToPredic.replace("30-34",2,inplace=True)
dataToPredic.replace("35-39",3,inplace=True)
dataToPredic.replace("40-44",4,inplace=True)
dataToPredic.replace("45-49",5,inplace=True)
dataToPredic.replace("50-54",6,inplace=True)
dataToPredic.replace("55-59",7,inplace=True)
dataToPredic.replace("60-64",8,inplace=True)
dataToPredic.replace("65-69",9,inplace=True)
dataToPredic.replace("70-74",10,inplace=True)
dataToPredic.replace("75-79",11,inplace=True)
dataToPredic.replace("80 or older",13,inplace=True)


dataToPredic.replace("No, borderline diabetes",2,inplace=True)
dataToPredic.replace("Yes, during pregnancy",3,inplace=True)


dataToPredic.replace("Excellent",0,inplace=True)
dataToPredic.replace("Good",1,inplace=True)
dataToPredic.replace("Fair",2,inplace=True)
dataToPredic.replace("Very good",3,inplace=True)
dataToPredic.replace("Poor",4,inplace=True)


dataToPredic.replace("White",0,inplace=True)
dataToPredic.replace("Other",1,inplace=True)
dataToPredic.replace("Black",2,inplace=True)
dataToPredic.replace("Hispanic",3,inplace=True)
dataToPredic.replace("Asian",4,inplace=True)
dataToPredic.replace("American Indian/Alaskan Native",5,inplace=True)


dataToPredic.replace("Female",0,inplace=True)
dataToPredic.replace("Male",1,inplace=True)

# Load the previously saved machine learning model
filename='LogRegModel.pkl'
loaded_model= pickle.load(open(filename, 'rb'))
Result=loaded_model.predict(dataToPredic)
ResultProb= loaded_model.predict_proba(dataToPredic)
ResultProb1=round(ResultProb[0][1] * 100, 2)

#  # Calculate the probability of getting heart disease
# if st.button('PREDICT'):
#  # st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
#  if (ResultProb1>30):
#   st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
#  else:
#   st.write('You have a', ResultProb1, '% chance of getting a heart disease' )

# Calculate the probability of getting heart disease
if st.button('PREDICT'):
    # If predicted probability is greater than or equal to 0.5, classify as "Yes"
    if ResultProb[0][1] >= 0.5:
        st.write('Prediction: **Yes**, you may have heart disease.')
    else:
        st.write('Prediction: **No**, you are less likely to have heart disease.')


  

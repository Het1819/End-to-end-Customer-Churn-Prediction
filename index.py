import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

sclr = StandardScaler()
# 526,France,Male,52,8,93590.47,1,0,1,21228.71,1
# loading models
df = pickle.load(open('df.pkl', 'rb'))
rfc = pickle.load(open('rfc.pkl', 'rb'))

def prediction(country_1, country_2, country_3, credit_score, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary):
    # Check for empty strings and handle accordingly
    if country_1 == '':
        st.error("Please provide a valid country position 1.")
        return None
    if country_2 == '':
        st.error("Please provide a valid country position 2.")
        return None
    if country_3 == '':
        st.error("Please provide a valid country position 3.")
        return None
    if credit_score == '':
        st.error("Please provide a valid credit score.")
        return None
    if gender == '':
        st.error("Please provide a valid gender.")
        return None
    if age == '':
        st.error("Please provide a valid age.")
        return None
    if tenure == '':
        st.error("Please provide a valid tenure.")
        return None
    if balance == '':
        st.error("Please provide a valid balance.")
        return None
    if products_number == '':
        st.error("Please provide a valid number of products.")
        return None
    if credit_card == '':
        st.error("Please provide a valid credit card status.")
        return None
    if active_member == '':
        st.error("Please provide a valid active member status.")
        return None
    if estimated_salary == '':
        st.error("Please provide a valid estimated salary.")
        return None

    features = np.array([[country_1, country_2, country_3, int(credit_score), gender, int(age), int(tenure), float(balance), int(products_number), int(credit_card), int(active_member), float(estimated_salary)]])
    features = sclr.fit_transform(features)  # Scale the country column
    prediction = rfc.predict(features).reshape(1, -1)
    return prediction

# web app
st.title('Bank Customer Churn Prediction')

st.caption('If Country France is then Country position 1 = "1", Country position 2 = "0", Country position 3 = "0" ')
st.caption('If Country Germany is then Country position 1 = "0", Country position 2 = "1", Country position 3 = "0" ')
st.caption('If Country Spain is then Country position 1 = "0", Country position 2 = "0", Country position 3 = "1" ')

country_1 = st.text_input('Country position 1')
country_2 = st.text_input('Country position 2')
country_3 = st.text_input('Country position 3')

credit_score = st.number_input('Credit Score')

st.caption('Male = "1" , Female = "0"')
gender = st.text_input('Gender')

age = st.number_input('Age')
tenure = st.number_input('Tenure')
balance = st.number_input('Balance')
products_number = st.number_input('Products Number')
credit_card = st.number_input('Credit Card')
active_member = st.number_input('Active Member')
estimated_salary = st.number_input('Estimated Salary')

if st.button('Predict'):
    pred = prediction(country_1, country_2, country_3,credit_score, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary)

    if pred is not None:
        if pred == 1:
            st.write("The customer has left.")
        else:
            st.write("The customer is still active.")


# 703,France,Female,51,3,0,3,1,1,77294.56,1
# 609,Spain,Male,61,1,0,1,1,0,22447.85,1
# 629,France,Female,37,10,99546.25,3,0,1,25136.95,1
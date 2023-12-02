import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go
import matplotlib.pyplot as plt
colors = px.colors.sequential.Plasma_r

import pickle

# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc

# Warnings
import warnings 
warnings. filterwarnings('ignore')

#Streamlit
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)



# Loading the data
df = pd.read_csv("/Users/bhavyakalra/Desktop/Informatics/Final Project/CVD_cleaned.csv")

# Display a subset of the data using st.dataframe
st.title('Cardiovascular Disease Dataset')
subset_size = st.slider('Number of Rows to Display', 1, len(df), value=2)
st.dataframe(df.head(subset_size))



st.title("Identification of Risk Factors")


# 1. Health assesment
st.header('1. Health Assessment')

fig1 = px.histogram(df, x="General_Health", color = 'General_Health', color_discrete_sequence = colors, title="1. Distribution of General Health")
fig1.update_layout(plot_bgcolor='white')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(df, x="General_Health", color = 'Heart_Disease', color_discrete_sequence = colors, barmode = 'group', title="2. General Health with respect to Heart Disease")
fig2.update_layout(plot_bgcolor='white')
st.plotly_chart(fig2, use_container_width=True)


# 2. Gender analysis
st.header('2. Gender Analysis on Health')

sex_counts = df['Sex'].value_counts()
age_category_counts = df['Age_Category'].value_counts()

fig3 = px.bar(x=sex_counts.index, y=sex_counts.values, color=sex_counts.index, color_discrete_sequence = colors, labels={'x': 'Sex', 'y': 'Count'})
fig3.update_layout(title="1. Distribution of gender in the Dataset", xaxis_title="", yaxis_title="Count", plot_bgcolor='white')
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.histogram(df, x="Sex", color='Heart_Disease', barmode='group', color_discrete_sequence= colors, title="2. Checking which gender is more susceptible to Heart Disease?")
fig4.update_layout(xaxis_title="Gender", yaxis_title="Count", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
st.plotly_chart(fig4, use_container_width=True)

grouped_data = df.groupby(['Sex', 'Heart_Disease'], as_index=False)['BMI'].median()
fig = px.bar(grouped_data, x='Sex', y='BMI', color='Heart_Disease', color_discrete_sequence = colors, barmode = 'group', title="3. Checking  gender and their average bmi based on heart disease?")
fig.update_layout(xaxis_title="Gender", yaxis_title="Average BMI", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)


# 3. Analysis on Age dependancies

st.header('3. Analysis on Age dependancies')

fig5 = px.bar(x=age_category_counts.index, y=age_category_counts.values, color=age_category_counts.index, color_discrete_sequence = colors ,labels={'x': 'Age Category', 'y': 'Count'})
fig5.update_layout(title="1. Distribution of Age Categories in the Dataset", xaxis_title="", yaxis_title="Count", plot_bgcolor='white')
st.plotly_chart(fig5, use_container_width=True)

fig6 = px.histogram(df, x="Age_Category", color='Heart_Disease', barmode='group', color_discrete_sequence= colors, title="2. Checking which age group is more susceptible to Heart Disease?")
fig6.update_layout(xaxis_title="Age_Category", yaxis_title="Count", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
st.plotly_chart(fig6, use_container_width=True)

grouped_data = df.groupby(['Age_Category', 'Heart_Disease'], as_index=False)['BMI'].median()
fig7 = px.bar(grouped_data, x='Age_Category', y='BMI', color='Heart_Disease', color_discrete_sequence = colors, barmode = 'group', title="3. Checking  age groups and their average bmi based on heart disease?")
fig7.update_layout(xaxis_title="Age Group", yaxis_title="Average BMI", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
st.plotly_chart(fig7, use_container_width=True)



# 4. Impact of lifestyle analysis
st.header('4. Impact of lifestyle analysis')

def create_bar_chart(data, x_col, y_col, color_col, title, x_label, y_label):
    grouped_data = data.groupby([x_col, color_col]).size().reset_index(name='count')
    fig8 = px.bar(grouped_data, x=x_col, y=y_col, color=color_col, color_discrete_sequence=colors, title=title, labels={x_col: x_label, y_col: y_label}, barmode='group', category_orders={x_col: ["No", "Yes"], color_col: ["No", "Yes"]} )
    fig8.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig8, use_container_width=True)
    
create_bar_chart(df, 'Exercise', 'count', 'Heart_Disease', '1. Impact of Exercise on Heart Disease', 'Exercise', 'Count')
create_bar_chart(df, 'Smoking_History', 'count', 'Heart_Disease', '2. Impact of Smoking on Heart Disease', 'Smoking History', 'Count')

columns = ['Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
titles = ["Alcohol Consumption", "Fruit Consumption", "Vegetables Consumption", "Potato Consumption"]

for i, col in enumerate(columns):
    grouped_data = df.groupby(['Age_Category', 'Heart_Disease'], as_index=False)[col].median()  # Use median here
    fig9 = px.bar(grouped_data, x='Age_Category', y=col, color='Heart_Disease', color_discrete_sequence=colors, barmode='group', title=f"{i + 4}. Impact of {titles[i]} on Heart Disease")
    fig9.update_layout(xaxis_title="Age Group", yaxis_title=f"Median {titles[i]}", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
    st.plotly_chart(fig9, use_container_width=True)




# 5. Correlation analysis
st.header('5. Correlation Analysis')

fig10 = px.histogram(df, x="Depression", color="Heart_Disease", barmode = 'group', color_discrete_sequence=colors, title="1. Correlation between Depression and Heart Disease")
fig10.update_layout(plot_bgcolor='white')
st.plotly_chart(fig10, use_container_width=True)

fig11 = px.histogram(df, x="Diabetes", color="Heart_Disease", barmode = 'group', color_discrete_sequence=colors, title="2. Correlation between Diabetes and Heart Disease")
fig11.update_layout(plot_bgcolor='white')
st.plotly_chart(fig11, use_container_width=True)

fig12 = px.box(df, x="Diabetes", y="BMI", title="3. BMI levels of poeple dealing with diabetes and heart disease", color="Heart_Disease", color_discrete_sequence=colors)
fig12.update_layout(xaxis_title= "Diabetes Status", yaxis_title= "BMI", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
st.plotly_chart(fig12, use_container_width=True)



# Prediction model using Random Forest classification

loaded_model = pickle.load(open('Model_rf.pickle', 'rb'))

def CVD_prediction(input_data):

    input_data_np = np.asarray(input_data)

    input_data_reshaped = input_data_np.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if (prediction[0] == 0):
        return 'Person is not prone to Heart Disease'
    else:
        return 'Person is prone to Heart disease'



def main():

    st.title("Cardiovascular Disease Prediction model")


    General_Health = st.text_input("General Health")
    Checkup = st.text_input("Number of Checkup")
    Exercise = st.text_input("Frequency of exercise")
    Skin_Cancer = st.text_input("Have/had skin cancer")
    Other_Cancer = st.text_input("Have/had any cancer")
    Depression = st.text_input("Have/had Depression")
    Diabetes = st.text_input("Have/had Diabetes")
    Arthritis = st.text_input("Have/had Arthritis")
    Age_Category = st.text_input("Age Category of the person")
    Height = st.text_input("Height in Centimeters")
    Weight = st.text_input("Weight in Kilograms")
    BMI = st.text_input("BMI")
    Alcohol_Consumption = st.text_input("Alcohol Consumption")
    Fruit_Consumption = st.text_input("Fruit Consumption")
    Green_Vegetables_Consumption = st.text_input("Green Vegetables Consumption")
    FriedPotato_Consumption = st.text_input("Fried Potato Consumption")
    bmi_group = st.text_input("BMi Group according to Age")
    Sex_Female = st.text_input("Gender Female")
    Sex_Male = st.text_input("Gender Male")
    Smoking_History_No = st.text_input("Previous Smoking History")
    Smoking_History_Yes = st.text_input("Current Smoking History")



    # Diagnosis
    Diagnosis = ''

    if st.button('CVD result'):
        Diagnosis = CVD_prediction([General_Health, Checkup, Exercise, Skin_Cancer, Other_Cancer, Depression, Diabetes, Arthritis, Age_Category, Height, Weight, BMI, Alcohol_Consumption, Fruit_Consumption, Green_Vegetables_Consumption, FriedPotato_Consumption, bmi_group, Sex_Female, Sex_Male, Smoking_History_No, Smoking_History_Yes])
        

    st.success(Diagnosis)



if __name__ == '__main__':
    main()





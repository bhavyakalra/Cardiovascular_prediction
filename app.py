import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
df = pd.read_csv("/Users/bhavyakalra/Desktop/Informatics/Final Project/CVD_cleaned.csv")
plt.figure(figsize=(12, 8))
sns.countplot(x='Age_Category', hue='Heart_Disease', data=df, palette='Set2')
plt.title('Heart Disease Distribution Across Age Categories')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.legend(title='Heart Disease', loc='upper right')


# Streamlit App
st.title('Simple Streamlit Web App')

# Display the scatter plot
st.pyplot()

# Display a subset of the data using st.dataframe
st.title('Subset of Cardiovascular Disease Dataset')
subset_size = st.slider('Number of Rows to Display', 1, len(df), value=2)
st.dataframe(df.head(subset_size))


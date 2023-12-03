# Cardiovascular Diseases Risk Prediction

## Link to the web app: https://cardiovascularprediction.streamlit.app

#### Github does not permit file size more than 100mb, my model_rf.zip file has the model saved which can be run on local streamlit server but cannot run on community cloud.



## Problem Statement :

- Heart disease continues to be a significant global health concern, impacting millions of lives and posing a substantial burden on healthcare systems. Early detection and accurate prediction of heart disease play a crucial role in managing and preventing its onset, as well as improving patient outcomes. Over the years, advancements in medical research, technology, and data analysis have significantly enhanced our understanding of heart disease prediction. This brief introduction aims to provide an overview of the progress made in this field, highlighting key methodologies and approaches utilized in predicting heart disease. By exploring the various risk factors, diagnostic tools, and emerging trends, we can gain valuable insights into the evolving landscape of heart disease prediction, paving the way for more proactive and personalized healthcare strategies.

### Dataset Attributes :

- General_Health
- Checkup
- Exercise
- Heart_Disease
- Skin_Cancer
- Other_Cancer
- Depression
- Diabetes
- Arthritis
- Sex
- Age_Category
- Height_(cm)
- Weight_(kg)
- BMI
- Smoking_History
- Alcohol_Consumption
- Fruit_Consumption
- Green_Vegetables_Consumption
- FriedPotato_Consumption

### Notebook Contents :

- Dataset Information
  - Data Cleaning
- Exploratory Data Analysis (EDA)
  - Feature Engineering
- Modeling
- Conclusion

### PROJECT OBJECTIVES:

- Analyze the data
- Extract the hidden trends and patterns
- Group the reviewers based upon health conditions and risk
- Analyze Each clusters and find out the nature for better medical recommendations
- Examine which Machine Learning Algorithm is most suitable for the dataset

## Data cleaning and aftermath

- From the boxplot as well as the code its very clear that there are outliers, while comparing the mean value and maximum value,the maximum value is very far from the mean value .The double cross checking is implemented because this dataset is related to health,extreme values may be possibile due to extreme medical conditions .At the same time there is another possibility that people may answer false or assumptions to the phone call based review system. Its impossible to guarantee 100 percentage that the answers may be exact.

After removing the outliers 
- Mean values of all the features for cases of heart disease and no heart disease.
- The average weight of people with heart disease is slightly higher than people without heart disease.
- The average consumption of vegetables in people with heart disease is lower than people without heart disease.


## Analysis from Data visualization

- The risk factor analysis revealed that people who have a poor health have more chances of heart disease.

- In terms of gender, the data shows a high ratio of male diagnosed with heart disease and they have a bit high BMI as compared to female.

- According to age groups, the elderly poeple have the highest count for heart disease. However, mid aged people having heart disease tend to have a higher BMI

- In terms of exercise, it doesn't play a significant role. However, smoking is an important factor for heart disease poeple.

- The alcohol consumption has a significant impact on heart disease, however, it doesn't affect much when a person is in young age. Furthermore, the analysis revaled that, fruit consumption isn't having much effect on poeple with heart disease.

- Similarly, green vegetables consumption has a significant impact on heart disease, specially, it affects more when a person is in adult age. Furthermore, the analysis revaled that, fried potato consumption isn't having any effect on poeple with heart disease.

- The correlation analysis revealed that people who have a depression have more chances of heart disease.

- Similarly, diabetes has a significant impact on heart disease, specially, people with diabetes and heart disease have a higher bmi on average.

## Prediction Model

- Between Random forest classifier, Logistic regression and XGBoost classifier. We achieve the maximum accuracy from Random forest classifier. 


## Conclusion Heart Disease Insights:

- To sum up, the dataset analysis produced a number of insightful findings. Data preprocessing involved altering column names, values, and data types, and null values were handled well. 'bmi_group,' a new feature, was developed using BMI bins and redesigned for data type and position. The majority of the population, according to the dataset's general health profile, had frequent exercise, a normal weight, and recent physicals. In addition, major risk factors for heart disease were found, such as poor health, a higher BMI, gender, age group, smoking, drinking alcohol, depression, and diabetes. Furthermore, Random Forest machine learning analysis performed the best, showing an amazing F1-score and AUC of 0.93, indicating that it may accurately predict heart disease in this context.

### Future work

- These findings highlight the value of risk assessment and tailored interventions in public health initiatives. We can improve disease prevention and treatment by incorporating these observations into clinical practices, which will ultimately result in improved health outcomes on a larger scale.



## Algorithm behind the streamlit web app is based on the findings from Exploratory Data Analysis, and feature engineering done on the BMI column. It also showcases the prediction model running on the Random Forest classifier to identity whether a person is prone to heart disease or not by taking all the factors in affect.


## Tools Used: 
Streamlit Installation: I installed Streamlit to create a user-friendly web interface for my project.

Pandas and NumPy Usage: I used Pandas for data manipulation and NumPy for numerical operations on the dataset.

Choice of Regression Model Library: I selected a regression model library such as Scikit-learn, Logistic regression,Random forest classifier to build my prediction model.

Data Visualization with Matplotlib/plotly: I employed Matplotlib and plotly for effective data visualization, enabling me to present insights and results clearly.

Dataset Integration: I loaded my dataset into the app, ensuring it was in a format compatible with my chosen regression model.

Implementing User Input Handling: I developed features for users to input values interactively, facilitating dynamic model predictions.

Regression Model Training and Evaluation: I trained my regression model using the dataset and evaluated its performance, showcasing relevant metrics to users.

Displaying Model Predictions: I allowed users to see model predictions based on their input, providing a practical application of the regression model.

Documentation Creation: I provided clear comments in my code and, if necessary, created a readme file to explain how to run and interact with my app.










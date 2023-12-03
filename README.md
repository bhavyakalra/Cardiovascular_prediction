# Cardiovascular Diseases Risk Prediction

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

- In terms of gender, the data shows a high ratio of male diagnosed with heart disease and they have a bit hight bmi as compared to females

- According to age groups, the elderly poeple have the highest count for heart disease. However, mid aged people having heart disease tend to have a higher bmi

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












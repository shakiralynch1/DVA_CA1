#Mutlivariate Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('modified_student_scores.csv')


#initialize the independent variables
gender = df['gender_female']
part_time_job = df['part_time_job_True']
weekly_self_study_hours = df['weekly_self_study_hours']
extracurricular_activities = df['extracurricular_activities_True']

# Scale the weekly_self_study_hours
scaler = StandardScaler()
scaled_weekly_self_study_hours = scaler.fit_transform(weekly_self_study_hours.values.reshape(-1, 1))


#intialize the dependent variables
math_score = df['math_score']
history_score = df['history_score']
physics_score = df['physics_score']
chemistry_score = df['chemistry_score']
biology_score = df['biology_score']
english_score = df['english_score']
geography_score = df['geography_score']

# Scale all the scores

scaled_math_score = scaler.fit_transform(math_score.values.reshape(-1, 1))
scaled_history_score = scaler.fit_transform(history_score.values.reshape(-1, 1))
scaled_physics_score = scaler.fit_transform(physics_score.values.reshape(-1, 1))
scaled_chemistry_score = scaler.fit_transform(chemistry_score.values.reshape(-1, 1))
scaled_biology_score = scaler.fit_transform(biology_score.values.reshape(-1, 1))
scaled_english_score = scaler.fit_transform(english_score.values.reshape(-1, 1))
scaled_geography_score = scaler.fit_transform(geography_score.values.reshape(-1, 1))

#IV TO IV SCATTER PLOTS

# IV TO IV SCATTER PLOTS

# correlation between gender and part_time_job
plt.scatter(gender, part_time_job)
plt.xlabel('gender')
plt.ylabel('part_time_job')
plt.title('Relationship between Gender and Part-Time Job')
plt.show()

# correlation between gender and extracurricular_activities
plt.scatter(gender, extracurricular_activities)
plt.xlabel('gender')
plt.ylabel('extracurricular_activities')
plt.title('Relationship between Gender and Extracurricular Activities')
plt.show()

# correlation between gender and weekly_self_study_hours
plt.scatter(gender, scaled_weekly_self_study_hours)
plt.xlabel('gender')
plt.ylabel('weekly_self_study_hours')
plt.title('Relationship between Gender and Weekly Self Study Hours')
plt.show()

# correlation between part_time_job and extracurricular_activities
plt.scatter(part_time_job, extracurricular_activities)
plt.xlabel('part_time_job')
plt.ylabel('extracurricular_activities')
plt.title('Relationship between Part-Time Job and Extracurricular Activities')
plt.show()

# correlation between part_time_job and weekly_self_study_hours
plt.scatter(part_time_job, scaled_weekly_self_study_hours)
plt.xlabel('part_time_job')
plt.ylabel('weekly_self_study_hours')
plt.title('Relationship between Part-Time Job and Weekly Self Study Hours')
plt.show()

# correlation between extracurricular_activities and weekly_self_study_hours
plt.scatter(extracurricular_activities, scaled_weekly_self_study_hours)
plt.xlabel('extracurricular_activities')
plt.ylabel('weekly_self_study_hours')
plt.title('Relationship between Extracurricular Activities and Weekly Self Study Hours')
plt.show()

##No correlation between IV TO IV




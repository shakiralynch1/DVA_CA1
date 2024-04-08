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


#####################################

#IV TO DV SCATTER PLOTS

######################################

# correlation between gender and math_score
plt.scatter(gender, scaled_math_score)
plt.xlabel('gender')
plt.ylabel('math_score')
plt.title('Relationship between Gender and Math Score')
plt.show()
plt.scatter(gender, scaled_math_score)
plt.xlabel('gender')
plt.ylabel('math_score')
plt.title('Relationship between Gender and Math Score')
plt.show()

# correlation between gender and history_score
plt.scatter(gender, scaled_history_score)
plt.xlabel('gender')
plt.ylabel('history_score')
plt.title('Relationship between Gender and History Score')
plt.show()

# correlation between gender and physics_score
plt.scatter(gender, scaled_physics_score)
plt.xlabel('gender')
plt.ylabel('physics_score')
plt.title('Relationship between Gender and Physics Score')
plt.show()

# correlation between gender and chemistry_score
plt.scatter(gender, scaled_chemistry_score)
plt.xlabel('gender')
plt.ylabel('chemistry_score')
plt.title('Relationship between Gender and Chemistry Score')
plt.show()

# correlation between gender and biology_score
plt.scatter(gender, scaled_biology_score)
plt.xlabel('gender')
plt.ylabel('biology_score')
plt.title('Relationship between Gender and Biology Score')
plt.show()

# correlation between gender and english_score
plt.scatter(gender, scaled_english_score)
plt.xlabel('gender')
plt.ylabel('english_score')
plt.title('Relationship between Gender and English Score')
plt.show()

# correlation between gender and geography_score
plt.scatter(gender, scaled_geography_score)
plt.xlabel('gender')
plt.ylabel('geography_score')
plt.title('Relationship between Gender and Geography Score')
plt.show()

# correlation between part_time_job and math_score
plt.scatter(part_time_job, scaled_math_score)
plt.xlabel('part_time_job')
plt.ylabel('math_score')
plt.title('Relationship between Part-Time Job and Math Score')
plt.show()

# correlation between part_time_job and history_score
plt.scatter(part_time_job, scaled_history_score)
plt.xlabel('part_time_job')
plt.ylabel('history_score')
plt.title('Relationship between Part-Time Job and History Score')
plt.show()

# correlation between part_time_job and physics_score
plt.scatter(part_time_job, scaled_physics_score)
plt.xlabel('part_time_job')
plt.ylabel('physics_score')
plt.title('Relationship between Part-Time Job and Physics Score')
plt.show()

# correlation between part_time_job and chemistry_score
plt.scatter(part_time_job, scaled_chemistry_score)
plt.xlabel('part_time_job')
plt.ylabel('chemistry_score')
plt.title('Relationship between Part-Time Job and Chemistry Score')
plt.show()

# correlation between part_time_job and biology_score
plt.scatter(part_time_job, scaled_biology_score)
plt.xlabel('part_time_job')
plt.ylabel('biology_score')
plt.title('Relationship between Part-Time Job and Biology Score')
plt.show()

# correlation between part_time_job and english_score
plt.scatter(part_time_job, scaled_english_score)
plt.xlabel('part_time_job')
plt.ylabel('english_score')
plt.title('Relationship between Part-Time Job and English Score')
plt.show()

# correlation between part_time_job and geography_score
plt.scatter(part_time_job, scaled_geography_score)
plt.xlabel('part_time_job')
plt.ylabel('geography_score')
plt.title('Relationship between Part-Time Job and Geography Score')
plt.show()

# correlation between extracurricular_activities and math_score
plt.scatter(extracurricular_activities, scaled_math_score)
plt.xlabel('extracurricular_activities')
plt.ylabel('math_score')
plt.title('Relationship between Extracurricular Activities and Math Score')
plt.show()

# correlation between extracurricular_activities and history_score
plt.scatter(extracurricular_activities, scaled_history_score)
plt.xlabel('extracurricular_activities')
plt.ylabel('history_score')
plt.title('Relationship between Extracurricular Activities and History Score')
plt.show()

# correlation between extracurricular_activities and physics_score
plt.scatter(extracurricular_activities, scaled_physics_score)
plt.xlabel('extracurricular_activities')
plt.ylabel('physics_score')
plt.title('Relationship between Extracurricular Activities and Physics Score')
plt.show()

# correlation between extracurricular_activities and chemistry_score
plt.scatter(extracurricular_activities, scaled_chemistry_score)
plt.xlabel('extracurricular_activities')
plt.ylabel('chemistry_score')
plt.title('Relationship between Extracurricular Activities and Chemistry Score')
plt.show()

# correlation between extracurricular_activities and biology_score
plt.scatter(extracurricular_activities, scaled_biology_score)
plt.xlabel('extracurricular_activities')
plt.ylabel('biology_score')
plt.title('Relationship between Extracurricular Activities and Biology Score')
plt.show()

# correlation between extracurricular_activities and english_score
plt.scatter(extracurricular_activities, scaled_english_score)
plt.xlabel('extracurricular_activities')
plt.ylabel('english_score')
plt.title('Relationship between Extracurricular Activities and English Score')
plt.show()

# correlation between extracurricular_activities and geography_score
plt.scatter(extracurricular_activities, scaled_geography_score)
plt.xlabel('extracurricular_activities')
plt.ylabel('geography_score')
plt.title('Relationship between Extracurricular Activities and Geography Score')
plt.show()


# correlation between weekly_self_study_hours and math_score
plt.scatter(scaled_weekly_self_study_hours, scaled_math_score)
plt.xlabel('weekly_self_study_hours')
plt.ylabel('math_score')
plt.title('Relationship between Weekly Self Study Hours and Math Score')
plt.show()

# correlation between weekly_self_study_hours and history_score
plt.scatter(scaled_weekly_self_study_hours, scaled_history_score)
plt.xlabel('weekly_self_study_hours')
plt.ylabel('history_score')
plt.title('Relationship between Weekly Self Study Hours and History Score')
plt.show()

# correlation between weekly_self_study_hours and physics_score
plt.scatter(scaled_weekly_self_study_hours, scaled_physics_score)
plt.xlabel('weekly_self_study_hours')
plt.ylabel('physics_score')
plt.title('Relationship between Weekly Self Study Hours and Physics Score')
plt.show()

# correlation between weekly_self_study_hours and chemistry_score
plt.scatter(scaled_weekly_self_study_hours, scaled_chemistry_score)
plt.xlabel('weekly_self_study_hours')
plt.ylabel('chemistry_score')
plt.title('Relationship between Weekly Self Study Hours and Chemistry Score')
plt.show()

# correlation between weekly_self_study_hours and biology_score
plt.scatter(scaled_weekly_self_study_hours, scaled_biology_score)
plt.xlabel('weekly_self_study_hours')
plt.ylabel('biology_score')
plt.title('Relationship between Weekly Self Study Hours and Biology Score')
plt.show()

# correlation between weekly_self_study_hours and english_score
plt.scatter(scaled_weekly_self_study_hours, scaled_english_score)
plt.xlabel('weekly_self_study_hours')
plt.ylabel('english_score')
plt.title('Relationship between Weekly Self Study Hours and English Score')
plt.show()

# correlation between weekly_self_study_hours and geography_score
plt.scatter(scaled_weekly_self_study_hours, scaled_geography_score)
plt.xlabel('weekly_self_study_hours')
plt.ylabel('geography_score')
plt.title('Relationship between Weekly Self Study Hours and Geography Score')
plt.show()

#Strong correlation between weekly_self_study_hours and math_score, physics_score, chemistry_score, biology_score, english_score, geography_score






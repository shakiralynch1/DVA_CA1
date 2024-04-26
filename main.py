import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler 


# Load the data
df = pd.read_csv('student-scores.csv')
print(df.head())

# One-hot encode the data 


# Drop the columns that are not needed
df = pd.DataFrame(df)
obselete_columns = ['id','first_name','last_name','email','gender', 'career_aspiration']
df.drop(obselete_columns, axis=1, inplace=True)
print(df.head())

df = pd.get_dummies(df, columns=['part_time_job', 'extracurricular_activities'], prefix=['part_time_job', 'extracurricular_activities'], dtype=int, )
print(df.head())

obselete_columns = ['part_time_job_False','extracurricular_activities_False']
df.drop(obselete_columns, axis=1, inplace=True)

# Explode the 'weekly_self_study_hours' column
df = df.explode(['weekly_self_study_hours'])

# Convert the 'weekly_self_study_hours' column to integer
df['weekly_self_study_hours'] = df['weekly_self_study_hours'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x).astype(np.int64)

# Save the modified data
df.to_csv('modified_student_scores.csv', index=False)


#initialize the independent variables

part_time_job = df['part_time_job_True'].values
weekly_self_study_hours = df['weekly_self_study_hours'].values
missing_days = df['absence_days'].values
extracurricular_activities = df['extracurricular_activities_True'].values


#intialize the dependent variables
math_score = df['math_score'].values
history_score = df['history_score'].values
physics_score = df['physics_score'].values
chemistry_score = df['chemistry_score'].values
biology_score = df['biology_score'].values
english_score = df['english_score'].values
geography_score = df['geography_score'].values

print(type(weekly_self_study_hours[0]))
print(weekly_self_study_hours[0])



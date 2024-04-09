#Mutlivariate Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr



df = pd.read_csv('modified_student_scores.csv')


#initialize the independent variables

part_time_job = df['part_time_job_True']
weekly_self_study_hours = df['weekly_self_study_hours']
missing_days = df['absence_days']
extracurricular_activities = df['extracurricular_activities_True']

# Scale the weekly_self_study_hours
scaler = StandardScaler()
scaled_weekly_self_study_hours = scaler.fit_transform(weekly_self_study_hours.values.reshape(-1, 1))
scaled_missing_days = scaler.fit_transform(missing_days.values.reshape(-1, 1))


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



#Strong correlation between weekly_self_study_hours and math_score, physics_score, chemistry_score, biology_score, english_score, geography_score


#Pearson Correlation Coefficient

# Pearson correlation between gender and math_score
corr_missing_days_math, _ = pearsonr(missing_days, math_score)
print(f"Pearson correlation between missing days and math_score: {corr_missing_days_math}")

# Pearson correlation between gender and history_score
corr_missing_days_history, _ = pearsonr(missing_days, history_score)
print(f"Pearson correlation between missing days and history_score: {corr_missing_days_history}")

# Pearson correlation between gender and physics_score
corr_missing_days_physics, _ = pearsonr(missing_days, physics_score)
print(f"Pearson correlation between missing days and physics_score: {corr_missing_days_physics}")

# Pearson correlation between gender and chemistry_score
corr_missing_days_chemsitry, _ = pearsonr(missing_days, chemistry_score)
print(f"Pearson correlation between missing days and chemistry_score: {corr_missing_days_chemsitry}")

# Pearson correlation between gender and biology_score
corr_missing_days_biology, _ = pearsonr(missing_days, biology_score)
print(f"Pearson correlation between missing days and biology_score: {corr_missing_days_biology}")

# Pearson correlation between gender and english_score
corr_missing_days_english, _ = pearsonr(missing_days, english_score)
print(f"Pearson correlation between missing days and english_score: {corr_missing_days_english}")

# Pearson correlation between gender and geography_score
corr_missing_days_geography, _ = pearsonr(missing_days, geography_score)
print(f"Pearson correlation between missing days and geography_score: {corr_missing_days_geography}")

# Pearson correlation between part_time_job and math_score
corr_part_time_job_math, _ = pearsonr(part_time_job, math_score)
print(f"Pearson correlation between part_time_job and math_score: {corr_part_time_job_math}")

# Pearson correlation between part_time_job and history_score
corr_part_time_job_history, _ = pearsonr(part_time_job, history_score)
print(f"Pearson correlation between part_time_job and history_score: {corr_part_time_job_history}")

# Pearson correlation between part_time_job and physics_score
corr_part_time_job_physics, _ = pearsonr(part_time_job, physics_score)
print(f"Pearson correlation between part_time_job and physics_score: {corr_part_time_job_physics}")

# Pearson correlation between part_time_job and chemistry_score
corr_part_time_job_chemistry, _ = pearsonr(part_time_job, chemistry_score)
print(f"Pearson correlation between part_time_job and chemistry_score: {corr_part_time_job_chemistry}")

# Pearson correlation between part_time_job and biology_score
corr_part_time_job_biology, _ = pearsonr(part_time_job, biology_score)
print(f"Pearson correlation between part_time_job and biology_score: {corr_part_time_job_biology}")

# Pearson correlation between part_time_job and english_score
corr_part_time_job_english, _ = pearsonr(part_time_job, english_score)
print(f"Pearson correlation between part_time_job and english_score: {corr_part_time_job_english}")

# Pearson correlation between part_time_job and geography_score
corr_part_time_job_geography, _ = pearsonr(part_time_job, geography_score)
print(f"Pearson correlation between part_time_job and geography_score: {corr_part_time_job_geography}")

# Pearson correlation between extracurricular_activities and math_score
corr_extracurricular_math, _ = pearsonr(extracurricular_activities, math_score)
print(f"Pearson correlation between extracurricular_activities and math_score: {corr_extracurricular_math}")

# Pearson correlation between extracurricular_activities and history_score
corr_extracurricular_history, _ = pearsonr(extracurricular_activities, history_score)
print(f"Pearson correlation between extracurricular_activities and history_score: {corr_extracurricular_history}")

# Pearson correlation between extracurricular_activities and physics_score
corr_extracurricular_physics, _ = pearsonr(extracurricular_activities, physics_score)
print(f"Pearson correlation between extracurricular_activities and physics_score: {corr_extracurricular_physics}")

# Pearson correlation between extracurricular_activities and chemistry_score
corr_extracurricular_chemistry, _ = pearsonr(extracurricular_activities, chemistry_score)
print(f"Pearson correlation between extracurricular_activities and chemistry_score: {corr_extracurricular_chemistry}")

# Pearson correlation between extracurricular_activities and biology_score
corr_extracurricular_biology, _ = pearsonr(extracurricular_activities, biology_score)
print(f"Pearson correlation between extracurricular_activities and biology_score: {corr_extracurricular_biology}")

# Pearson correlation between extracurricular_activities and english_score
corr_extracurricular_english, _ = pearsonr(extracurricular_activities, english_score)
print(f"Pearson correlation between extracurricular_activities and english_score: {corr_extracurricular_english}")

# Pearson correlation between extracurricular_activities and geography_score
corr_extracurricular_geography, _ = pearsonr(extracurricular_activities, geography_score)
print(f"Pearson correlation between extracurricular_activities and geography_score: {corr_extracurricular_geography}")

# Pearson correlation between weekly_self_study_hours and math_score
corr_weekly_self_study_math, _ = pearsonr(weekly_self_study_hours, math_score)
print(f"Pearson correlation between weekly_self_study_hours and math_score: {corr_weekly_self_study_math}")

# Pearson correlation between weekly_self_study_hours and history_score
corr_weekly_self_study_history, _ = pearsonr(weekly_self_study_hours, history_score)
print(f"Pearson correlation between weekly_self_study_hours and history_score: {corr_weekly_self_study_history}")

# Pearson correlation between weekly_self_study_hours and physics_score
corr_weekly_self_study_physics, _ = pearsonr(weekly_self_study_hours, physics_score)
print(f"Pearson correlation between weekly_self_study_hours and physics_score: {corr_weekly_self_study_physics}")

# Pearson correlation between weekly_self_study_hours and chemistry_score
corr_weekly_self_study_chemistry, _ = pearsonr(weekly_self_study_hours, chemistry_score)
print(f"Pearson correlation between weekly_self_study_hours and chemistry_score: {corr_weekly_self_study_chemistry}")

# Pearson correlation between weekly_self_study_hours and biology_score
corr_weekly_self_study_biology, _ = pearsonr(weekly_self_study_hours, biology_score)
print(f"Pearson correlation between weekly_self_study_hours and biology_score: {corr_weekly_self_study_biology}")

# Pearson correlation between weekly_self_study_hours and english_score
corr_weekly_self_study_english, _ = pearsonr(weekly_self_study_hours, english_score)
print(f"Pearson correlation between weekly_self_study_hours and english_score: {corr_weekly_self_study_english}")

# Pearson correlation between weekly_self_study_hours and geography_score
corr_weekly_self_study_geography, _ = pearsonr(weekly_self_study_hours, geography_score)
print(f"Pearson correlation between weekly_self_study_hours and geography_score: {corr_weekly_self_study_geography}")



#highest correlation is between weekly_self_study_hours and math_score, physics_score, chemistry_score, biology_score, english_score, geography_score

#Pearson correlation between weekly_self_study_hours and math_score: 0.39356929824986275
#Pearson correlation between weekly_self_study_hours and history_score: 0.27623076103095523
#Pearson correlation between weekly_self_study_hours and physics_score: 0.20211984851555595
#Pearson correlation between weekly_self_study_hours and chemistry_score: 0.20134020274229125
#Pearson correlation between weekly_self_study_hours and biology_score: 0.19048082333736108
#Pearson correlation between weekly_self_study_hours and english_score: 0.2477960153508276
#Pearson correlation between weekly_self_study_hours and geography_score: 0.15362244292021798



















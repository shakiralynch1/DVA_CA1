#Mutlivariate Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


df = pd.read_csv('modified_student_scores.csv')

#initialize the independent variables

part_time_job = df['part_time_job_True'].values
weekly_self_study_hours = df['weekly_self_study_hours'].values
missing_days = df['absence_days'].values
extracurricular_activities = df['extracurricular_activities_True'].values

print(type(extracurricular_activities[0]))
print(extracurricular_activities[0])


#intialize the dependent variables
math_score = df['math_score'].values
history_score = df['history_score'].values
physics_score = df['physics_score'].values
chemistry_score = df['chemistry_score'].values
biology_score = df['biology_score'].values
english_score = df['english_score'].values
geography_score = df['geography_score'].values


# IV TO IV SCATTER PLOTS

# Scatter plot between part_time_job and weekly_self_study_hours
plt.scatter(part_time_job, weekly_self_study_hours)
plt.xlabel('Part Time Job')
plt.ylabel('Weekly Self Study Hours')
plt.title('Scatter Plot: Part Time Job vs Weekly Self Study Hours')
plt.show()

# Scatter plot between extracurricular_activities and weekly_self_study_hours
plt.scatter(extracurricular_activities, weekly_self_study_hours)
plt.xlabel('Extracurricular Activities')
plt.ylabel('Weekly Self Study Hours')
plt.title('Scatter Plot: Extracurricular Activities vs Weekly Self Study Hours')
plt.show()

# Scatter plot between missing_days and weekly_self_study_hours
plt.scatter(missing_days, weekly_self_study_hours)
plt.xlabel('Missing Days')
plt.ylabel('Weekly Self Study Hours')
plt.title('Scatter Plot: Missing Days vs Weekly Self Study Hours')
plt.show()

# Scatter plot between part_time_job and missing_days
plt.scatter(part_time_job, missing_days)
plt.xlabel('Part Time Job')
plt.ylabel('Missing Days')
plt.title('Scatter Plot: Part Time Job vs Missing Days')
plt.show()

# Scatter plot between extracurricular_activities and missing_days
plt.scatter(extracurricular_activities, missing_days)
plt.xlabel('Extracurricular Activities')
plt.ylabel('Missing Days')
plt.title('Scatter Plot: Extracurricular Activities vs Missing Days')
plt.show()

# Scatter plot between part_time_job and extracurricular_activities
plt.scatter(part_time_job, extracurricular_activities)
plt.xlabel('Part Time Job')
plt.ylabel('Extracurricular Activities')
plt.title('Scatter Plot: Part Time Job vs Extracurricular Activities')
plt.show()

# IV TO DV SCATTER PLOTS

# Scatter plot between missing_days and math_score
plt.scatter(missing_days, math_score)
plt.xlabel('Missing Days')
plt.ylabel('Math Score')
plt.title('Scatter Plot: Missing Days vs Math Score')
plt.show()

# Scatter plot between missing_days and history_score
plt.scatter(missing_days, history_score)
plt.xlabel('Missing Days')
plt.ylabel('History Score')
plt.title('Scatter Plot: Missing Days vs History Score')
plt.show()

# Scatter plot between missing_days and physics_score
plt.scatter(missing_days, physics_score)
plt.xlabel('Missing Days')
plt.ylabel('Physics Score')
plt.title('Scatter Plot: Missing Days vs Physics Score')
plt.show()

# Scatter plot between missing_days and chemistry_score
plt.scatter(missing_days, chemistry_score)
plt.xlabel('Missing Days')
plt.ylabel('Chemistry Score')
plt.title('Scatter Plot: Missing Days vs Chemistry Score')
plt.show()

# Scatter plot between missing_days and biology_score
plt.scatter(missing_days, biology_score)
plt.xlabel('Missing Days')
plt.ylabel('Biology Score')
plt.title('Scatter Plot: Missing Days vs Biology Score')
plt.show()

# Scatter plot between missing_days and english_score
plt.scatter(missing_days, english_score)
plt.xlabel('Missing Days')
plt.ylabel('English Score')
plt.title('Scatter Plot: Missing Days vs English Score')
plt.show()

# Scatter plot between missing_days and geography_score
plt.scatter(missing_days, geography_score)
plt.xlabel('Missing Days')
plt.ylabel('Geography Score')
plt.title('Scatter Plot: Missing Days vs Geography Score')
plt.show()

# Scatter plot between part_time_job and math_score
plt.scatter(part_time_job, math_score)
plt.xlabel('Part Time Job')
plt.ylabel('Math Score')
plt.title('Scatter Plot: Part Time Job vs Math Score')
plt.show()

# Scatter plot between part_time_job and history_score
plt.scatter(part_time_job, history_score)
plt.xlabel('Part Time Job')
plt.ylabel('History Score')
plt.title('Scatter Plot: Part Time Job vs History Score')
plt.show()

# Scatter plot between part_time_job and physics_score
plt.scatter(part_time_job, physics_score)
plt.xlabel('Part Time Job')
plt.ylabel('Physics Score')
plt.title('Scatter Plot: Part Time Job vs Physics Score')
plt.show()

# Scatter plot between part_time_job and chemistry_score
plt.scatter(part_time_job, chemistry_score)
plt.xlabel('Part Time Job')
plt.ylabel('Chemistry Score')
plt.title('Scatter Plot: Part Time Job vs Chemistry Score')
plt.show()

# Scatter plot between part_time_job and biology_score
plt.scatter(part_time_job, biology_score)
plt.xlabel('Part Time Job')
plt.ylabel('Biology Score')
plt.title('Scatter Plot: Part Time Job vs Biology Score')
plt.show()

# Scatter plot between part_time_job and english_score
plt.scatter(part_time_job, english_score)
plt.xlabel('Part Time Job')
plt.ylabel('English Score')
plt.title('Scatter Plot: Part Time Job vs English Score')
plt.show()

# Scatter plot between part_time_job and geography_score
plt.scatter(part_time_job, geography_score)
plt.xlabel('Part Time Job')
plt.ylabel('Geography Score')
plt.title('Scatter Plot: Part Time Job vs Geography Score')
plt.show()

# Scatter plot between extracurricular_activities and math_score
plt.scatter(extracurricular_activities, math_score)
plt.xlabel('Extracurricular Activities')
plt.ylabel('Math Score')
plt.title('Scatter Plot: Extracurricular Activities vs Math Score')
plt.show()

# Scatter plot between extracurricular_activities and history_score
plt.scatter(extracurricular_activities, history_score)
plt.xlabel('Extracurricular Activities')
plt.ylabel('History Score')
plt.title('Scatter Plot: Extracurricular Activities vs History Score')
plt.show()

# Scatter plot between extracurricular_activities and physics_score
plt.scatter(extracurricular_activities, physics_score)
plt.xlabel('Extracurricular Activities')
plt.ylabel('Physics Score')
plt.title('Scatter Plot: Extracurricular Activities vs Physics Score')
plt.show()

# Scatter plot between extracurricular_activities and chemistry_score
plt.scatter(extracurricular_activities, chemistry_score)
plt.xlabel('Extracurricular Activities')
plt.ylabel('Chemistry Score')
plt.title('Scatter Plot: Extracurricular Activities vs Chemistry Score')
plt.show()

# Scatter plot between extracurricular_activities and biology_score
plt.scatter(extracurricular_activities, biology_score)
plt.xlabel('Extracurricular Activities')
plt.ylabel('Biology Score')
plt.title('Scatter Plot: Extracurricular Activities vs Biology Score')
plt.show()

# Scatter plot between extracurricular_activities and english_score
plt.scatter(extracurricular_activities, english_score)
plt.xlabel('Extracurricular Activities')
plt.ylabel('English Score')
plt.title('Scatter Plot: Extracurricular Activities vs English Score')
plt.show()

# Scatter plot between extracurricular_activities and geography_score
plt.scatter(extracurricular_activities, geography_score)
plt.xlabel('Extracurricular Activities')
plt.ylabel('Geography Score')
plt.title('Scatter Plot: Extracurricular Activities vs Geography Score')
plt.show()

# Scatter plot between weekly_self_study_hours and math_score
plt.scatter(weekly_self_study_hours, math_score)
plt.xlabel('Weekly Self Study Hours')
plt.ylabel('Math Score')
plt.title('Scatter Plot: Weekly Self Study Hours vs Math Score')
plt.show()

# Scatter plot between weekly_self_study_hours and history_score
plt.scatter(weekly_self_study_hours, history_score)
plt.xlabel('Weekly Self Study Hours')
plt.ylabel('History Score')
plt.title('Scatter Plot: Weekly Self Study Hours vs History Score')
plt.show()

# Scatter plot between weekly_self_study_hours and physics_score
plt.scatter(weekly_self_study_hours, physics_score)
plt.xlabel('Weekly Self Study Hours')
plt.ylabel('Physics Score')
plt.title('Scatter Plot: Weekly Self Study Hours vs Physics Score')
plt.show()

# Scatter plot between weekly_self_study_hours and chemistry_score
plt.scatter(weekly_self_study_hours, chemistry_score)
plt.xlabel('Weekly Self Study Hours')
plt.ylabel('Chemistry Score')
plt.title('Scatter Plot: Weekly Self Study Hours vs Chemistry Score')
plt.show()

#IV TO IV SCATTER PLOTS
# IV TO IV SCATTER PLOTS



#somehwhat of a  correlation between weekly_self_study_hours and math_score, physics_score, chemistry_score, biology_score, english_score, geography_score

#Pearson Correlation Coefficient

#Pearson correlation between IV and IV

# Pearson correlation between part_time_job and weekly_self_study_hours
corr_part_time_job_weekly_self_study, _ = pearsonr(part_time_job, weekly_self_study_hours)
print(f"Pearson correlation between part_time_job and weekly_self_study_hours: {corr_part_time_job_weekly_self_study}")

# Pearson correlation between extracurricular_activities and weekly_self_study_hours
corr_extracurricular_weekly_self_study, _ = pearsonr(extracurricular_activities, weekly_self_study_hours)
print(f"Pearson correlation between extracurricular_activities and weekly_self_study_hours: {corr_extracurricular_weekly_self_study}")

# Pearson correlation between missing_days and weekly_self_study_hours
corr_missing_days_weekly_self_study, _ = pearsonr(missing_days, weekly_self_study_hours)
print(f"Pearson correlation between missing_days and weekly_self_study_hours: {corr_missing_days_weekly_self_study}")

# Pearson correlation between part_time_job and missing_days
corr_part_time_job_missing_days, _ = pearsonr(part_time_job, missing_days)
print(f"Pearson correlation between part_time_job and missing_days: {corr_part_time_job_missing_days}")

# Pearson correlation between extracurricular_activities and missing_days
corr_extracurricular_missing_days, _ = pearsonr(extracurricular_activities, missing_days)
print(f"Pearson correlation between extracurricular_activities and missing_days: {corr_extracurricular_missing_days}")

# Pearson correlation between part_time_job and extracurricular_activities
corr_part_time_job_extracurricular, _ = pearsonr(part_time_job, extracurricular_activities)
print(f"Pearson correlation between part_time_job and extracurricular_activities: {corr_part_time_job_extracurricular}")

##############################################################################################
#Pearson correlation between part_time_job and missing_days: 0.20636065645123192 MUTLICONLINEARITY!!!!!!!!!!!!!





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
#The variables that affect the performance in scores is weekly_self_study_hours
#The variables that affect the performance in scores is not part_time_job, extracurricular_activities, missing_days
#The variables that affect the performance 


#Pearson correlation between weekly_self_study_hours and math_score: 0.39356929824986275
#Pearson correlation between weekly_self_study_hours and history_score: 0.27623076103095523
#Pearson correlation between weekly_self_study_hours and physics_score: 0.20211984851555595
#Pearson correlation between weekly_self_study_hours and chemistry_score: 0.20134020274229125
#Pearson correlation between weekly_self_study_hours and biology_score: 0.19048082333736108
#Pearson correlation between weekly_self_study_hours and english_score: 0.2477960153508276
#Pearson correlation between weekly_self_study_hours and geography_score: 0.15362244292021798




#modeling bivariate relationships
model = LinearRegression()
weekly_self_study_hours_arr = np.array(weekly_self_study_hours).reshape(-1,1)
model.fit(weekly_self_study_hours_arr, math_score)
r_sq= model.score(weekly_self_study_hours_arr, math_score)
print(r_sq)

#R^2 value is 0.154

model_weekly_self_study_hours = sm.add_constant(weekly_self_study_hours)
model =sm.OLS(math_score, model_weekly_self_study_hours)
result=model.fit()
print(result.summary())
#line below is for the standard error
print("Standard Error: ")
print(result.scale**0.5)


model_weekly_self_study_hours = sm.add_constant(weekly_self_study_hours)
model =sm.OLS(history_score, model_weekly_self_study_hours)
result=model.fit()
print(result.summary())
#line below is for the standard error
print("Standard Error: ")
print(result.scale**0.5)


model_weekly_self_study_hours = sm.add_constant(weekly_self_study_hours)
model =sm.OLS(physics_score, model_weekly_self_study_hours)
result=model.fit()
print(result.summary())
#line below is for the standard error
print("Standard Error: ")
print(result.scale**0.5)



model_weekly_self_study_hours = sm.add_constant(weekly_self_study_hours)
model =sm.OLS(chemistry_score, model_weekly_self_study_hours)
result=model.fit()
print(result.summary())
#line below is for the standard error
print("Standard Error: ")
print(result.scale**0.5)

model_weekly_self_study_hours = sm.add_constant(weekly_self_study_hours)
model =sm.OLS(biology_score, model_weekly_self_study_hours)
result=model.fit()
print(result.summary())
#line below is for the standard error
print("Standard Error: ")
print(result.scale**0.5)


model_weekly_self_study_hours = sm.add_constant(weekly_self_study_hours)
model =sm.OLS(english_score, model_weekly_self_study_hours)
result=model.fit()
print(result.summary())
#line below is for the standard error
print("Standard Error: ")
print(result.scale**0.5)


model_weekly_self_study_hours = sm.add_constant(weekly_self_study_hours)
model =sm.OLS(geography_score, model_weekly_self_study_hours)
result=model.fit()
print(result.summary())
#line below is for the standard error
print("Standard Error: ")
print(result.scale**0.5)




print(weekly_self_study_hours[0])

#visualising the correlation between weekly study hours and mathscore with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title(" math score vs Weekly study hours ")
plt.xlabel("Mathscore")
plt.ylabel("weekly study hours")
scatter = plt.scatter( math_score,weekly_self_study_hours, c= part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names,title="Part Time Job")
#plt.show()

#From the graph I can see that students who have a part time job have a higher math score than students who do not have a part time job 

#visualising the correlation between weekly study hours and history score with if the student is working part time or not

# Visualizing the correlation between weekly study hours and history score with if the student is working part time or not



names = ["Working", "Not Working"]
plt.title("History score vs Weekly study hours")
plt.xlabel("History score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(history_score, weekly_self_study_hours, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

# Visualizing the correlation between weekly study hours and physics score with if the student is working part time or not

plt.title("Physics score vs Weekly study hours")
plt.xlabel("Physics score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(physics_score, weekly_self_study_hours, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

# Visualizing the correlation between weekly study hours and chemistry score with if the student is working part time or not

plt.title("Chemistry score vs Weekly study hours")
plt.xlabel("Chemistry score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(chemistry_score, weekly_self_study_hours, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

# Visualizing the correlation between weekly study hours and biology score with if the student is working part time or not

plt.title("Biology score vs Weekly study hours")
plt.xlabel("Biology score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(biology_score, weekly_self_study_hours, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

# Visualizing the correlation between weekly study hours and English score with if the student is working part time or not

plt.title("English score vs Weekly study hours")
plt.xlabel("English score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(english_score, weekly_self_study_hours, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

# Visualizing the correlation between weekly study hours and geography score with if the student is working part time or not

plt.title("Geography score vs Weekly study hours")
plt.xlabel("Geography score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(geography_score, weekly_self_study_hours, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()



print("After")

print(type(weekly_self_study_hours[0]))
print(weekly_self_study_hours[0])
print(type(extracurricular_activities[0]))
print(extracurricular_activities[0])
print(len(extracurricular_activities))
print(type(missing_days[0]))
print(missing_days[0])
print(type(math_score[0]))
print(math_score[0])



#plotting the correlation between weekly study hours and math score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Math score vs Weekly study hours")
plt.xlabel("Math score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(math_score, weekly_self_study_hours, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between weekly study hours and history score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("History score vs Weekly study hours")
plt.xlabel("History score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(history_score, weekly_self_study_hours, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between weekly study hours and physics score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Physics score vs Weekly study hours")
plt.xlabel("Physics score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(physics_score, weekly_self_study_hours, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between weekly study hours and chemistry score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Chemistry score vs Weekly study hours")
plt.xlabel("Chemistry score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(chemistry_score, weekly_self_study_hours, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between weekly study hours and biology score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Biology score vs Weekly study hours")
plt.xlabel("Biology score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(biology_score, weekly_self_study_hours, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between weekly study hours and english score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("English score vs Weekly study hours")
plt.xlabel("English score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(english_score, weekly_self_study_hours, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between weekly study hours and geography score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Geography score vs Weekly study hours")
plt.xlabel("Geography score")
plt.ylabel("Weekly study hours")
scatter = plt.scatter(geography_score, weekly_self_study_hours, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()


#####CORRIELATION BETWEEN MISSING DAYS AND SCORES WITH IF THE STUDENT IS WORKING PART TIME OR NOT



#plotting the correlation between missing days and math score with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title("Math score vs Missing days")
plt.xlabel("Math score")
plt.ylabel("Missing days")
scatter = plt.scatter(math_score, missing_days, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

#plotting the correlation between missing days and history score with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title("History score vs Missing days")
plt.xlabel("History score")
plt.ylabel("Missing days")
scatter = plt.scatter(history_score, missing_days, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

#plotting the correlation between missing days and physics score with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title("Physics score vs Missing days")
plt.xlabel("Physics score")
plt.ylabel("Missing days")
scatter = plt.scatter(physics_score, missing_days, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

#plotting the correlation between missing days and chemistry score with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title("Chemistry score vs Missing days")
plt.xlabel("Chemistry score")
plt.ylabel("Missing days")
scatter = plt.scatter(chemistry_score, missing_days, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

#plotting the correlation between missing days and biology score with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title("Biology score vs Missing days")
plt.xlabel("Biology score")
plt.ylabel("Missing days")
scatter = plt.scatter(biology_score, missing_days, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

#plotting the correlation between missing days and english score with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title("English score vs Missing days")
plt.xlabel("English score")
plt.ylabel("Missing days")
scatter = plt.scatter(english_score, missing_days, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
#plt.show()

#plotting the correlation between missing days and geography score with if the student is working part time or not
names = ["Working", "Not Working"]
plt.title("Geography score vs Missing days")
plt.xlabel("Geography score")
plt.ylabel("Missing days")
scatter = plt.scatter(geography_score, missing_days, c=part_time_job, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Part Time Job")
plt.show()

#####CORRIELATION BETWEEN MISSING DAYS AND SCORES WITH IF THE STUDENT HAS EXTRA CURRICULAR OR NOT

#plotting the correlation between missing days and math score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Math score vs Missing days")
plt.xlabel("Math score")
plt.ylabel("Missing days")
scatter = plt.scatter(math_score, missing_days, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
plt.show()

#plotting the correlation between missing days and history score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("History score vs Missing days")
plt.xlabel("History score")
plt.ylabel("Missing days")
scatter = plt.scatter(history_score, missing_days, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between missing days and physics score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Physics score vs Missing days")
plt.xlabel("Physics score")
plt.ylabel("Missing days")
scatter = plt.scatter(physics_score, missing_days, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
##plt.show()

#plotting the correlation between missing days and chemistry score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Chemistry score vs Missing days")
plt.xlabel("Chemistry score")
plt.ylabel("Missing days")
scatter = plt.scatter(chemistry_score, missing_days, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between missing days and biology score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Biology score vs Missing days")
plt.xlabel("Biology score")
plt.ylabel("Missing days")
scatter = plt.scatter(biology_score, missing_days, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between missing days and english score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("English score vs Missing days")
plt.xlabel("English score")
plt.ylabel("Missing days")
scatter = plt.scatter(english_score, missing_days, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()

#plotting the correlation between missing days and geography score with if the student is doing extracurricular activities or not
names = ["Doing", "Not Doing"]
plt.title("Geography score vs Missing days")
plt.xlabel("Geography score")
plt.ylabel("Missing days")
scatter = plt.scatter(geography_score, missing_days, c=extracurricular_activities, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Extracurricular Activities")
#plt.show()



fig = plt.figure(figsize=(14, 12))
ax = plt.axes(projection='3d')
x = np.array(math_score)
y = np.array(weekly_self_study_hours)
z = np.array(part_time_job)
#z = [row[0] for row in z]
ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), linewidth=0.2, antialiased=True)
#plt.show()

fig = plt.figure(figsize=(14, 12))
ax = plt.axes(projection='3d')
x = np.array(history_score)
y = np.array(weekly_self_study_hours)
z = np.array(part_time_job)
#z = [row[0] for row in z]
ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), linewidth=0.2, antialiased=True)
#plt.show()

fig = plt.figure(figsize=(14, 12))
ax = plt.axes(projection='3d')
x = np.array(physics_score)
y = np.array(weekly_self_study_hours)
z = np.array(part_time_job)
#z = [row[0] for row in z]
ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), linewidth=0.2, antialiased=True)
#plt.show()

fig = plt.figure(figsize=(14, 12))
ax = plt.axes(projection='3d')
x = np.array(biology_score)
y = np.array(weekly_self_study_hours)  
z = np.array(part_time_job)
#z = [row[0] for row in z]
ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), linewidth=0.2, antialiased=True)
#plt.show()

fig = plt.figure(figsize=(14, 12))
ax = plt.axes(projection='3d')
x = np.array(chemistry_score)
y = np.array(weekly_self_study_hours)
z = np.array(part_time_job)
#z = [row[0] for row in z]
ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), linewidth=0.2, antialiased=True)
#plt.show()

fig = plt.figure(figsize=(14, 12))
ax = plt.axes(projection='3d')
x = np.array(geography_score)
y = np.array(weekly_self_study_hours)
z = np.array(part_time_job)
#z = [row[0] for row in z]
ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), linewidth=0.2, antialiased=True)
#plt.show()

import statsmodels.api as sm

X = df[['weekly_self_study_hours', 'part_time_job']]
X = sm.add_constant(X)
y = df['math_score']

model = sm.OLS(y, X).fit()
print(model.summary())

for score in scores:
    y = df[score]
    model = sm.OLS(y, X).fit()
    print(f"Summary for {score}:")
    print(model.summary())
    print("\n")
    
    
X = df[['missing_days', 'part_time_job_True']]
X = sm.add_constant(X)

scores = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']

for score in scores:
    y = df[score]
    model = sm.OLS(y, X).fit()
    print(f"Summary for {score}:")
    print(model.summary())
    print("\n")
    
X = df[['weekly_self_study_hours', 'extracurricular_activities_True']]
X = sm.add_constant(X)

scores = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']

for score in scores:
    y = df[score]
    model = sm.OLS(y, X).fit()
    print(f"Summary for {score}:")
    print(model.summary())
    print("\n")
       
       
X = df[['missing_days', 'extracurricular_activities_True']]
X = sm.add_constant(X)

scores = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']

for score in scores:
    y = df[score]
    model = sm.OLS(y, X).fit()
    print(f"Summary for {score}:")
    print(model.summary())
    print("\n")
    
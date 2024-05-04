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
df['extracurricular_activities_True'] = df['extracurricular_activities_True'].astype('int64')
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
print(type(extracurricular_activities[0]))
print(extracurricular_activities[0])


# Calculate the distribution of scores
math_score_dist = np.histogram(math_score, bins=10)
history_score_dist = np.histogram(history_score, bins=10)
physics_score_dist = np.histogram(physics_score, bins=10)
chemistry_score_dist = np.histogram(chemistry_score, bins=10)
biology_score_dist = np.histogram(biology_score, bins=10)
english_score_dist = np.histogram(english_score, bins=10)
geography_score_dist = np.histogram(geography_score, bins=10)

# Plot the distribution of scores
plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.hist(math_score, bins=10)
plt.title('Math Score Distribution')

plt.subplot(2, 4, 2)
plt.hist(history_score, bins=10)
plt.title('History Score Distribution')

plt.subplot(2, 4, 3)
plt.hist(physics_score, bins=10)
plt.title('Physics Score Distribution')

plt.subplot(2, 4, 4)
plt.hist(chemistry_score, bins=10)
plt.title('Chemistry Score Distribution')

plt.subplot(2, 4, 5)
plt.hist(biology_score, bins=10)
plt.title('Biology Score Distribution')

plt.subplot(2, 4, 6)
plt.hist(english_score, bins=10)
plt.title('English Score Distribution')

plt.subplot(2, 4, 7)
plt.hist(geography_score, bins=10)
plt.title('Geography Score Distribution')

plt.tight_layout()
plt.show()

# Calculate the distribution of missing days
missing_days_dist = np.histogram(missing_days, bins=10)

# Plot the distribution of missing days
plt.figure(figsize=(6, 4))
plt.hist(missing_days, bins=10)
plt.title('Missing Days Distribution')
plt.xlabel('Number of Missing Days')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of part-time job
plt.figure(figsize=(6, 4))
plt.hist(part_time_job, bins=2)
plt.title('Part-Time Job Distribution')
plt.xlabel('Part-Time Job')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Plot the distribution of weekly self-study hours
plt.figure(figsize=(6, 4))
plt.hist(weekly_self_study_hours, bins=10)
plt.title('Weekly Self-Study Hours Distribution')
plt.xlabel('Number of Weekly Self-Study Hours')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of extracurricular activities
plt.figure(figsize=(6, 4))
plt.hist(extracurricular_activities, bins=2)
plt.title('Extracurricular Activities Distribution')
plt.xlabel('Extracurricular Activities')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

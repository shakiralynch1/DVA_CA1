import pandas as pd

df = pd.read_csv('student-scores.csv')
print(df.head())

encoded_df = pd.get_dummies(df, columns=['gender', 'part_time_job', 'extracurricular_activities'], prefix=['gender', 'part_time_job', 'extracurricular_activities'], dtype=int, )

print(encoded_df.head())




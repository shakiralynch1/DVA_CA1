import pandas as pd

df = pd.read_csv('student-scores.csv')
print(df.head())

encoded_df = pd.get_dummies(df, columns=['gender', 'part_time_job', 'extracurricular_activities'], prefix=['gender', 'part_time_job', 'extracurricular_activities'], dtype=int, )

print(encoded_df.head())


df = pd.DataFrame(encoded_df)
obselete_columns = ['id','first_name','last_name','email', 'career_aspiration']
df.drop(obselete_columns, axis=1, inplace=True)

print(df.head())

df.to_csv('modified_student_scores.csv', index=False)


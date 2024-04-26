import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


print(df)
# Save the modified data
df.to_csv('modified_student_scores.csv', index=False)











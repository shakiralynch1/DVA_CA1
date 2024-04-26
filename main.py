import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 


# Load the data
df = pd.read_csv('student-scores.csv')
print(df.head())

# One-hot encode the data 
encoded_df = pd.get_dummies(df, columns=['part_time_job', 'extracurricular_activities'], prefix=['part_time_job', 'extracurricular_activities'], dtype=int, )
print(encoded_df.head())

# Drop the columns that are not needed
df = pd.DataFrame(encoded_df)
obselete_columns = ['id','first_name','last_name','email', 'career_aspiration', 'part_time_job_False', 'extracurricular_activities_False']
df.drop(obselete_columns, axis=1, inplace=True)

print(df.head())

# Save the modified data
df.to_csv('modified_student_scores.csv', index=False)











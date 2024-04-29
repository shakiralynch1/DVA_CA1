import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



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

data = df[['part_time_job_True', 'weekly_self_study_hours', 'absence_days', 'extracurricular_activities_True',
                'math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score',
                'geography_score']]
column_names = np.array(data.columns)

# Standardize the variables
# Standardize the variables
scaled_data = StandardScaler().fit_transform(data)



# Perform PCA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
print(per_var)


# Get the loading scores
loading_scores = pca.components_.T * np.sqrt(pca.explained_variance_)

#plotting scree plot
labels=['PC'+str(i) for i in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal Components")
plt.title('Scree Plot')
plt.show()







import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



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


# Standardize the variables
math_score_scaled = StandardScaler().fit_transform(df['math_score'].values.reshape(-1, 1))
history_score_scaled = StandardScaler().fit_transform(df['history_score'].values.reshape(-1, 1))
physics_score_scaled = StandardScaler().fit_transform(df['physics_score'].values.reshape(-1, 1))
chemistry_score_scaled = StandardScaler().fit_transform(df['chemistry_score'].values.reshape(-1, 1))
biology_score_scaled = StandardScaler().fit_transform(df['biology_score'].values.reshape(-1, 1))
english_score_scaled = StandardScaler().fit_transform(df['english_score'].values.reshape(-1, 1))
geography_score_scaled = StandardScaler().fit_transform(df['geography_score'].values.reshape(-1, 1))

data = df[['part_time_job_True', 'weekly_self_study_hours', 'absence_days', 'extracurricular_activities_True',
                'math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score',
                'geography_score']]
column_names = np.array(data.columns)


# Standardize the variables
scaled_data = StandardScaler().fit_transform(data)



# Perform PCA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
print(per_var)



loading_scores = pca.components_.T * np.sqrt(pca.explained_variance_)

#plotting scree plot
labels=['PC'+str(i) for i in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal Components")
plt.title('Scree Plot')
plt.show()



loading_scores = pd.Series(loading_scores[0], index=column_names)
print(loading_scores)

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_genes = sorted_loading_scores[0:10].index.values
print(top_10_genes)

# Get the top 10 genes that contribute to pc1
print(loading_scores[top_10_genes])

# Split the data into independent variables (X) and dependent variable (y)

X = df[['part_time_job_True']]
y = df['biology_score']


###########MODEL ON PREDICTING ON SCORES####################


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

 


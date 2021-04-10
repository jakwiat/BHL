from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


df = pd.read_csv('final_train.csv')
df = df.iloc[:,1:]
klasy = df.loc[:,"Activity"]
print(f" Classes: {set(klasy)} ")
print(klasy.value_counts(), end="\n\n")
time_series_cols = [col for col in df if col.startswith('t') or col == "Activity"]
print(df[time_series_cols].head())
colnames = []
for col in df[time_series_cols].columns:
    colnames.append(col)
print(df.isna().sum().sum())
df = df.fillna(df.mean())
print(df.isna().sum().sum())
vals = df.iloc[:,1:]
tags = df.loc[:, "Activity"]
print(vals, tags)
v_train, v_test, t_train, t_test = train_test_split(vals, tags, test_size=0.3)
# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(v_train, t_train)
t_pred = clf.predict(v_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(t_test, t_pred))


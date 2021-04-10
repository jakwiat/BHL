from data_reader import prepare_data
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

v_train, v_test, t_train, t_test, colnames = prepare_data("final_train.csv", best_cols=50)

print("data prepared")

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(v_train, t_train)
t_pred = clf.predict(v_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(t_test, t_pred))

feature_imp = pd.Series(clf.feature_importances_, index=colnames).sort_values(ascending=False)
print(feature_imp.head(20))

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
# plt.show()

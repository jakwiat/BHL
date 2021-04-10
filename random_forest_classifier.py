from data_reader import prepare_data
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

v_train, v_test, t_train, t_test, colnames = prepare_data("final_train.csv", best_cols=50)

print("data prepared")

clf = RandomForestClassifier(n_estimators=100)
clf.fit(v_train, t_train)
t_pred = clf.predict(v_test)

print("Accuracy:", metrics.accuracy_score(t_test, t_pred))

feature_imp = pd.Series(clf.feature_importances_, index=colnames).sort_values(ascending=False)
print(feature_imp.head(20))

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

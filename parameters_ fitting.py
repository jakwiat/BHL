from data_reader import prepare_data
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


df_sizes = []
acc = []
for i in range(50, 501, 50):
    df_sizes.append(i)
    v_train, v_test, t_train, t_test, colnames = prepare_data("final_train.csv", best_cols=i)

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(v_train, t_train)
    t_pred = clf.predict(v_test)
    x = metrics.accuracy_score(t_test, t_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", x)
    acc.append((x))

print(df_sizes, acc)
plt.plot(df_sizes, acc)
#feature_imp = pd.Series(clf.feature_importances_, colnames).sort_values(ascending=False)
#print(feature_imp.head(30))
#feature_imp.to_csv("important_fts.csv")

# Creating a bar plot
#sns.barplot(x=feature_imp, y=feature_imp.index)
#plt.xlabel('Feature Importance Score')
#plt.ylabel('Features')
#plt.title("Visualizing Important Features")
#plt.legend()
plt.show()


df_sizes = []
acc = []
for i in range(1, 52, 10):
    df_sizes.append(i)
    v_train, v_test, t_train, t_test, colnames = prepare_data("final_train.csv", best_cols=i)

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(v_train, t_train)
    t_pred = clf.predict(v_test)
    x = metrics.accuracy_score(t_test, t_pred)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", x)
    acc.append((x))

print(df_sizes, acc)
plt.plot(df_sizes, acc)
plt.show()

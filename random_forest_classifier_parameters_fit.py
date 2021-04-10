from data_reader import prepare_data
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


df_sizes = []
acc = []
for i in range(50, 501, 50):
    df_sizes.append(i)
    v_train, v_test, t_train, t_test, colnames = prepare_data("final_train.csv", best_cols=i)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(v_train, t_train)
    t_pred = clf.predict(v_test)
    x = metrics.accuracy_score(t_test, t_pred)
    print("Accuracy:", x)
    acc.append((x))

print(df_sizes, acc)
plt.plot(df_sizes, acc)
plt.show()


df_sizes = []
acc = []
for i in range(1, 52, 10):
    df_sizes.append(i)
    v_train, v_test, t_train, t_test, colnames = prepare_data("final_train.csv", best_cols=i)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(v_train, t_train)
    t_pred = clf.predict(v_test)
    x = metrics.accuracy_score(t_test, t_pred)
    print("Accuracy:", x)
    acc.append((x))

print(df_sizes, acc)
plt.plot(df_sizes, acc)
plt.show()

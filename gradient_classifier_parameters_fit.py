from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from data_reader import prepare_data
import matplotlib.pyplot as plt


df_sizes = []
acc = []
for i in range(50, 501, 50):
    df_sizes.append(i)
    v_train, v_test, t_train, t_test, colnames = prepare_data("final_train.csv", best_cols=i)

    scaler = MinMaxScaler()
    v_train = scaler.fit_transform(v_train)
    v_test = scaler.transform(v_test)

    print("data normalized")

    gbc = GradientBoostingClassifier()
    gbc.fit(v_train, t_train)

    gbc_pred = gbc.predict(v_test)
    x = metrics.accuracy_score(t_test, gbc_pred)
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

    scaler = MinMaxScaler()
    v_train = scaler.fit_transform(v_train)
    v_test = scaler.transform(v_test)

    print("data normalized")

    gbc = GradientBoostingClassifier()
    gbc.fit(v_train, t_train)

    gbc_pred = gbc.predict(v_test)
    x = metrics.accuracy_score(t_test, gbc_pred)
    print("Accuracy:", x)
    acc.append((x))

print(df_sizes, acc)
plt.plot(df_sizes, acc)
plt.show()


from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from data_reader import prepare_data
from joblib import load


v_train, v_test, t_train, t_test, _ = prepare_data("final_train.csv", best_cols=200)

print("data prepared")

scaler = MinMaxScaler()
v_train = scaler.fit_transform(v_train)
v_test = scaler.transform(v_test)

print("data normalized")

gbc = load('filename.joblib')

gbc_pred = gbc.predict(v_test)

print("Accuracy:", metrics.accuracy_score(t_test, gbc_pred))

from sklearn.model_selection import train_test_split
import pandas as pd


def read_file(filename, best_cols):
    df = pd.read_csv(filename)
    df = df.iloc[:, 1:]
    if best_cols != 0:
        i_f = pd.read_csv('important_fts.csv')
        wazne = []
        wazne.append("Activity")
        wazne.extend(list(i_f.head(best_cols).iloc[:, 0]))
        print(wazne)
        df = df[wazne]
    klasy = df.loc[:, "Activity"]
    return df, klasy


def get_column_names(df):
    colnames = []
    for col in df.columns:
        if col != "Activity":
            colnames.append(col)
    return colnames


def prepare_data(filename, test_size=0.3, best_cols=0):
    df, klasy = read_file(filename, best_cols)
    df = df.fillna(df.mean())
    vals = df.iloc[:, 1:]
    tags = df.loc[:, "Activity"]
    columns = get_column_names(df)
    v_train, v_test, t_train, t_test = train_test_split(vals, tags, test_size=test_size)
    return v_train, v_test, t_train, t_test, columns

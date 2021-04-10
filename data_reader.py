import csv
import numpy as np
import pandas as pd


CLASSES = ["STANDING", "WALKING_UPSTAIRS", "LAYING", "WALKING", "SITTING", "WALKING_DOWNSTAIRS"]


def data_reader(filename):
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        data = []
        for row in csv_reader:
            if line_count == 0:
                attribute_names = row[1:]
            else:
                data.append(row[1:])
            line_count += 1
        print(line_count)
        print(attribute_names)
        print(data[0])

        return attribute_names, data


df = pd.read_csv('final_train.csv')
#attribute_names, data = data_reader('final_train.csv')
df = df.iloc[:,1:]
klasy = df.loc[:,"Activity"]
print(f" Classes: {set(klasy)} ")
print(klasy.value_counts(), end="\n\n")
time_series_cols = [col for col in df if col.startswith('t') or col == "Activity"]
print(df[time_series_cols].head())



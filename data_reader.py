import csv
import numpy as np
import pandas as pd

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


dane = pd.read_csv('final_train.csv')
#attribute_names, data = data_reader('final_train.csv')
dane = dane.iloc[:,1:]
klasy = dane.loc[:,"Activity"]
print(f" Classes: {set(klasy)} ")
print(klasy.value_counts(), end="\n\n")
print(dane.head())

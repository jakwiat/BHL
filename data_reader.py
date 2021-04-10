import csv

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

        return attribute_names, data


attribute_names, data = data_reader('final_train.csv')

print(attribute_names)
print(data[0])

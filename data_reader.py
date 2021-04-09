import csv

with open('final_train.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    line_count = 0
    data = []
    for row in csv_reader:
        if line_count == 0:
            z = f'Column names are {", ".join(row)}'

            line_count += 1
        else:
            data.append(row)
            line_count += 1
        print(line_count)
    print(z)
    print(f'Processed {line_count} lines.')
    print(data[0])

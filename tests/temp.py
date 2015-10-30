import csv
filename = '../data/20151023/good-rants.csv'
with open(filename, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',', quotechar="'")
    headers = next(reader)  # skip headers
    print(headers)

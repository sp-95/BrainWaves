from sklearn import linear_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numpy import array, reshape
import csv

path_to_csv = './soc_gen_data/train.csv'
data = []
with open(path_to_csv, newline='') as f:
    csv_reader = csv.reader(f)
    csv_headings = next(csv_reader)
    for row in csv_reader:
        data.append([float(x) for x in row])
    data = array(data)

clf = linear_model.Ridge(alpha = .8)
i = 1
for column in data[:, 1:-1].T:
    clf.fit(data[:, 0].reshape(-1, 1), column)
    print("X" + str(i) + ":" + str(clf.coef_[0]) + ":" + str(column[-1]))
    i += 1
from sklearn import *
import matplotlib.pyplot as plt
from numpy import *
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
x = []
for column in data[:, 1:-1].T:
    clf.fit(data[:, 0].reshape(-1, 1), column)
    x.append([column[-1], clf.coef_[0]])

x = array(x)
CENTROIDS = 10
prediction = cluster.KMeans(n_clusters=CENTROIDS).fit_predict(x)

plt.subplot(221)
plt.scatter(x[:,0], x[:,1], c=prediction)
plt.show()

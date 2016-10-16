from sklearn import linear_model, cluster
import matplotlib.pyplot as plt
from numpy import *
import csv
from io import open

path_to_csv = './soc_gen_data/train.csv'
data = []
with open(path_to_csv, newline='') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        data.append([float(x) for x in row])
    data = array(data)

sample_size = 100
clf = linear_model.Ridge(alpha = .8)
x = []
for column in data[:, 1:sample_size+1].T:
    clf.fit(data[:, 0].reshape(-1, 1), column)
    x.append([column[-1], clf.coef_[0]])

x = array(x)
CENTROIDS = 10
prediction = cluster.KMeans(n_clusters=CENTROIDS).fit_predict(x)

for i in range(sample_size):
    print("{},{}".format(header[i+1], prediction[i]+1))

plt.subplot(111)
for i in range(len(data)):
    plt.scatter(repeat(array(i), sample_size), data[i, 1:sample_size+1], c=prediction)
plt.show()
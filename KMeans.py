from sklearn import linear_model, cluster
from matplotlib import pyplot, cm
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

start = 1
end = 100
size = end - start + 1

clf = linear_model.Ridge(alpha = .8)
x = []
for column in data[:, start:end+1].T:
    clf.fit(data[:, 0].reshape(-1, 1), column)
    x.append([column[-1], clf.coef_[0]])

x = array(x)
CENTROIDS = 10
prediction = cluster.KMeans(n_clusters=CENTROIDS).fit_predict(x)

for i in range(size):
    print("{},{}".format(header[i+start], prediction[i]+1))

cmap = cm.jet
pyplot.subplot(111)
for i in range(size):
    pyplot.plot(data[:, i+start], color=cmap(prediction[i]/CENTROIDS), label=header[i+start])
pyplot.show()
from sklearn import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

clf = linear_model.Ridge(alpha = .8)
x = []
for column in data[:, 1:-1].T:
    clf.fit(data[:, 0].reshape(-1, 1), column)
    x.append([column[-1], clf.coef_[0]])

x = array(x)
CENTROIDS = 10
prediction = cluster.KMeans(n_clusters=CENTROIDS).fit_predict(x)

for i in range(100):
    print("{},{}".format(header[i+1], prediction[i]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.scatter(x[:,0], x[:,1], c=prediction)
for x in data[:, 1:-1]:
    ax.scatter(range(1, 101), range(1, 101), x, c=prediction)
plt.show()
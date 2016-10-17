from sklearn import linear_model, cluster
from matplotlib import pyplot, cm
import numpy as np
import csv
from io import open

training_csv = './soc_gen_data/train.csv'
with open(training_csv, newline='') as file:
    reader = csv.reader(file)
    header = next(reader)
    training_data = []
    for row in reader:
        training_data.append([float(x) for x in row])
    training_data = np.array(training_data)

start = 1
end = 100
size = end - start + 1
clf = linear_model.Ridge(alpha = .8)
x = []
for column in training_data[:, start:end+1].T:
    clf.fit(training_data[:, 0].reshape(-1, 1), column)
    x.append([column[-1], clf.coef_[0]])
x = np.array(x)

CENTROIDS = 10
kmeans = cluster.KMeans(n_clusters=CENTROIDS).fit(x)
clusters = kmeans.labels_

# print("Asset,Cluster")
# for i in range(size):
#     print("{},{}".format(header[i+start], clusters[i]+1))

# with open('result.txt', 'w') as result:
#     result.write("Asset,Cluster\n")
#     for i in range(size):
#         result.write("{},{}\n".format(header[i+start], clusters[i]+1))

cmap = cm.jet
pyplot.subplot(211)
for i in range(size):
    pyplot.plot(training_data[:, 0], training_data[:, i+start], color=cmap(clusters[i]/CENTROIDS))
pyplot.title("Training Data")
pyplot.ylabel("Stock Price")
pyplot.xlabel("Time")

test_csv = './soc_gen_data/test.csv'
with open(test_csv, newline='') as file:
    reader = csv.reader(file)
    header = next(reader)
    test_data = []
    for row in reader:
        test_data.append([float(x) for x in row])
    test_data = np.array(test_data)

x = []
for column in test_data[:, start:end+1].T:
    clf.fit(test_data[:, 0].reshape(-1, 1), column)
    x.append([column[-1], clf.coef_[0]])
x = np.array(x)

predictions = kmeans.predict(x)

# print("Asset,Cluster")
# for i in range(size):
#     print("{},{}".format(header[i+start], predictions[i]+1))

with open('result.txt', 'w') as result:
    result.write("Asset,Cluster\n")
    for i in range(size):
        result.write("{},{}\n".format(header[i+start], predictions[i]+1))

cmap = cm.jet
pyplot.subplot(212)
for i in range(size):
    pyplot.plot(test_data[:, 0], test_data[:, i+start], color=cmap(predictions[i]/CENTROIDS))
pyplot.title("Test Data")
pyplot.ylabel("Stock Price")
pyplot.xlabel("Time")
pyplot.show()
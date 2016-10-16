from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numpy import genfromtxt

path_to_csv = './soc_gen_data/train.csv'
data = genfromtxt(path_to_csv, dtype=float, delimiter=',', names=True)
print(data)
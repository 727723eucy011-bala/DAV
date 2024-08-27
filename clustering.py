import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random as r
import pandas as p 
x = []
for i in range(0,11):
    x.append(r.randint(1,2000))
y = []
for i in range(0,11):
    y.append(r.randint(1,2000))
data = list(zip(x, y))
inertias = []

df=p.read_csv('datasets.csv')
X = df[['population','longitude']]
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

plt.scatter(X['population'],X['longitude'],c=kmeans.labels_)
plt.title('Elbow method')
plt.xlabel('population')
plt.ylabel('longitude')
plt.show()
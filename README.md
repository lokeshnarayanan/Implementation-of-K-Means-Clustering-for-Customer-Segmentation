# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import all necessary packages.

2.Upload the appropiate dataset to perform K-Means Clustering.

3.Perform K-Means Clustering on the requried dataset.

4.Plot graph and display the clusters.

## Program:


Program to implement the K Means Clustering for Customer Segmentation.

Developed by: LOKESH N

RegisterNumber:  212222100023

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
```
```python
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
```
```python
X=data[['Annual Income (k$)','Spending Score (1-100)']]
X
```
```python
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()
```
```python
k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
```
```python
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
```
```python
colors=['r','g','b','c','m']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
### DATASET:
![Screenshot 2024-04-16 205704](https://github.com/chandrumathiyazhagan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393023/283e7366-6c48-4675-9d25-6090d9b9410e)
### GRAPH:
![Screenshot 2024-04-16 205549](https://github.com/chandrumathiyazhagan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393023/ca2882d2-2325-4edb-bab7-7e910238e7b7)
### CENTROID VALUE:
![Screenshot 2024-04-16 204953](https://github.com/chandrumathiyazhagan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393023/0d1c7485-eb52-42b3-ae02-ccc84406f8a4)
### K-MEANS CLUSTER:
![Screenshot 2024-04-16 204944](https://github.com/chandrumathiyazhagan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393023/aba371be-8e70-4794-a614-64647d7fa302)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

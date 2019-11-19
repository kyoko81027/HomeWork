import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csvtest as csvt

csvtemp = csvt.loadCsv


df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

df = pd.DataFrame(csvtemp.loadCSVfile2()[0])



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)#分三群
kmeans.fit(df)
y_pred = kmeans.fit_predict(df)#分成3種顏色

plt.figure(figsize=(10, 6))
plt.xlabel('')
plt.ylabel('')
plt.scatter(df[:, 0], df[:, 1], c=y_pred) #C是第三維度 顏色做維度

centroids = kmeans.cluster_centers_ #各群中心點(X,Y)的位置
plt.scatter(centroids[:, 0], centroids[:, 1],c='red') #標群心

plt.show()

kmeans.cluster_centers_ #各群中心點(X,Y)的位置
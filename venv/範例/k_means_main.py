import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tmp = np.loadtxt("dataset.csv", dtype=np.str, delimiter=",")
data = tmp[1:,1:].astype(np.float)#讀取數據

df = pd.DataFrame(data[0:,:7]) #將前7個欄位做一次分群

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)  # 分X群
kmeans.fit(df)
y_pred = kmeans.fit_predict(df)  # 將群分類


df = pd.DataFrame({'x':y_pred,'y':data[0:,7]})#在和錄取率做分群
y_pred = kmeans.fit_predict(df)  # 將群分類
centroids = kmeans.cluster_centers_  # 各群中心點(X,Y)的位置
plt.figure(figsize=(10, 6))
plt.ylabel('Chance of Admit ')
plt.xlabel('cluster')
plt.xticks(range(len(y_pred)))#設定為整數
plt.scatter(df['x'],df['y'], c=y_pred) #C是第三維度 顏色做維度
plt.scatter(centroids[:, 0], centroids[:, 1],c='red') #標群心
plt.show()
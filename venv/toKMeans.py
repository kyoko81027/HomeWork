import pandas as pd
import numpy as np

class toKMeans:
    def toKMeamsList(df,clusters):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=clusters)  # 分X群
        kmeans.fit(df)
        y_pred = kmeans.fit_predict(df)  # 將群分類
        centroids = kmeans.cluster_centers_  # 各群中心點(X,Y)的位置
        return y_pred,centroids

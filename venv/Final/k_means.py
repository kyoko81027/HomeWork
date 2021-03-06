import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csvtest as csvt
import toKMeans as km
kms = km.toKMeans
csvtemp = csvt.loadCsv

list = csvtemp.loadCSVfile2()#讀資料

df = pd.DataFrame(list[0:,:7]) #將前7個欄位做一次分群

reArr = kms.toKMeamsList(df,3)
cl_pred = reArr[0]#分群結果


df = pd.DataFrame({'x':cl_pred,'y':list[0:,7]})#在和錄取率做分群
reArr = kms.toKMeamsList(df,3)
y_pred = reArr[0]#分群結果
centroids = reArr[1]  #各群中心點(X,Y)的位置
plt.figure(figsize=(10, 6))
plt.ylabel('Chance of Admit ')
plt.xlabel('cluster')
plt.xticks(range(len(y_pred)))#設定為整數
plt.scatter(df['x'],df['y'], c=y_pred) #C是第三維度 顏色做維度
plt.scatter(centroids[:, 0], centroids[:, 1],c='red') #標群心
plt.show()

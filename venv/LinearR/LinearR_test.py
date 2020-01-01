import warnings
from collections import OrderedDict
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
warnings.filterwarnings('ignore')
df2=pd.read_csv("dataset.csv",encoding="big5")
df3=df2[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research','Chance of Admit ']]
df3.head()
exam_X=df3[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
exam_y=df3[['Chance of Admit ']]

for i in exam_y:
    exam_y.loc[ exam_y[i]>=0.7,i]=1
    exam_y.loc[ exam_y[i]<0.7,i]=0
#print(exam_y)

examDict=df3
examOrderedDict=OrderedDict(examDict)
examDf=pd.DataFrame(examOrderedDict)
examDf.head()
examDf.describe()
from sklearn.model_selection  import train_test_split
train_X,test_X,train_y,test_y =train_test_split(exam_X,exam_y,train_size=0.8)
print('訓練集大小',train_X.shape,train_y.shape)
print('測試集大小',test_X.shape,test_y.shape)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(train_X,train_y)
print('模型得分為',round(model.score(test_X,test_y),4))
rDf=examDf.corr()
print(rDf)

# set figure size
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (9, 9)

import seaborn as sns
sns.heatmap(rDf, square=True ,vmax=1.0, linecolor='white', annot=True)
plt.show()
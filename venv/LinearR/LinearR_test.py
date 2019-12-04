import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn import preprocessing, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')
plt.rcParams['font.family']='SimHei' #⿊體
df2=pd.read_csv("dataset.csv",encoding="big5")
df3=df2[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research','Chance of Admit ']]
df3.head()

x=df3[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
y=df3[['Chance of Admit ']]

#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20191120) #random_state 種子值

#from sklearn.utils import shuffle
#X_shuffle, y_shuffle = shuffle(x, y)

#from sklearn.preprocessing  import StandardScaler
#sc=StandardScaler()
#sc.fit(X_shuffle)

#x_train_nor=sc.transform(X_shuffle)
#x_test_nor=sc.transform(x_test)
#y_shuffle = np.array(y_shuffle, dtype=int).ravel()
y = np.array(y, dtype=int).ravel()
#y = y.ravel()
from sklearn.linear_model  import LinearRegression
from sklearn.feature_selection  import f_regression
#lr=LogisticRegression()
lr=LinearRegression()
lr.fit(x,y)

# 印出係數
print(lr.coef_)
# 印出截距
print(lr.intercept_ )

print(f_regression(x,y)[1])

predictions=lr.predict(x)

from sklearn import metrics
#Mean Absolute Error (MAE)代表平均誤差，公式為所有實際值及預測值相減的絕對值平均。
print(metrics.mean_absolute_error(y,predictions))
#Mean Squared Error (MSE)比起MSE可以拉開誤差差距，算是蠻常用的指標，公式所有實際值及預測值相減的平方的平均
print(metrics.mean_squared_error(y,predictions))
#Root Mean Squared Error (RMSE)代表MSE的平方根。比起MSE更為常用，因為更容易解釋y。
print(np.sqrt(metrics.mean_squared_error(y,predictions)))
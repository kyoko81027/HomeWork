import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn import preprocessing, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df2=pd.read_csv("dataset.csv",encoding="big5")
df3=df2[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research','Chance of Admit ']]
df3.head()

x=df3[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
y=df3[['Chance of Admit ']]


#y[y['Chance of Admit '] >= 0.7] = 1
#y[y['Chance of Admit '] < 0.7] = 0
y = np.where(y>=0.6,1,0).tolist()


#print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20191120) #random_state 種子值

#print(y_train)
#print(y_test)

from sklearn.utils import shuffle
X_shuffle, y_shuffle = shuffle(x, y)

from sklearn.preprocessing  import StandardScaler
sc=StandardScaler()
sc.fit(X_shuffle)

x_train_nor=sc.transform(X_shuffle)
x_test_nor=sc.transform(x_test)
y_shuffle = np.array(y_shuffle, dtype=int).ravel()


from sklearn.linear_model  import LogisticRegression
lr=LogisticRegression()
#lr=LinearRegression()
lr.fit(x_train_nor,y_shuffle)
predictions = lr.predict(x_test)
# 印出係數
#print(lr.coef_)
# 印出截距
#print(lr.intercept_ )


from sklearn import metrics
lr.score(x_train,y_train)
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
print("Precision:",metrics.precision_score(y_test, predictions))
print("Recall:",metrics.recall_score(y_test, predictions))

from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,predictions)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


y_pred_proba = lr.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import pyplot
import pandas as pd
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

'''Keras实现神经网络回归模型'''
# 读取数据
path = 'dataset.csv'
train_df = pd.read_csv(path)
# 删除不用字符串字段
# dataset = train_df.drop('jh',axis=1)
# df转换成array
values = train_df.values
# 原始数据标准化，为了加速收敛
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
y = scaled[:, -1]
X = scaled[:, 0:-1]
print(X)
print(y)
# 随机拆分训练集与测试集
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

# 全连接神经网络
model = Sequential()
input = X.shape[1]
# 隐藏层128
model.add(Dense(128, input_shape=(input,)))
model.add(Activation('rule'))
# Dropout层用于防止过拟合
# model.add(Dropout(0.2))
# 隐藏层128
model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
# 没有激活函数用于输出层，因为这是一个回归问题，
# 我们希望直接预测数值，而不需要采用激活函数进行变换
model.add(Dense(1))
# 使用高效的ADAM优化算法以及优化的最小均方误差损失函数
model.compile(loss='mean_squared_error', optimizer=Adam())
# early stopping
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
# 训练
history = model.fit(train_X, train_y, epochs=300, batch_size=20,
                    validation_data=(test_X, test_y), verbose=2,
                    shuffle=False, callbacks=[early_stopping])
# loss曲线
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# 预测
yhat = model.predict(test_X)
# 预测y 逆标准化
inv_yhat0 = concatenate((test_X, yhat), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat0)
inv_yhat = inv_yhat1[:, -1]
# 原始y逆标准化
test_y = test_y.reshape(len(test_y), 1)
inv_y0 = concatenate((test_X, test_y), axis=1)
inv_y1 = scaler.inverse_transform(inv_y0)
inv_y = inv_y1[:, -1]

# 计算RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y)
plt.plot(inv_yhat)
plt.show()
"""
LSTM时间序列问题预测：国际旅行人数预测
"""
import numpy as np
# import mxnet as mx
from matplotlib import pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

seed = 7
batch_size = 1
epochs = 500
# filename = 'international-airline-passengers.csv'
filename = 'data_visitors.csv'
footer = 0
look_back = 12
predict_steps = 12
timesteps = 24  # 构造x，为72个数据,表示每次用前72个数据作为一段

def create_dataset(dataset):
    # 创建数据集
    dataX, dataY = [], []
    # for i in range(len(dataset) - look_back - 1):
    for i in range(len(dataset) - look_back):
        x = dataset[i:i + look_back, 0]
        dataX.append(x)
        y = dataset[i + look_back, 0]
        dataY.append(y)
        # print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)


def build_model():
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(1, look_back)))
    # model.add(LSTM(units=4, input_shape=(look_back, 1)))
    # model.add(Dense(units=12))
    model.add(Dense(1))
    # 均方误差，也称标准差，缩写为MSE，利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率.
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train(trainsize,year):
    # 设置随机种子
    np.random.seed(seed)

    # 导入数据
    data = read_csv(filename, usecols=[1], engine='python', skipfooter=footer)  # skipfooter=10 则最后10行不读取
    dataset = data.values.astype('float32')
    # 标准化数据
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * trainsize)  # 训练集和验证集长度
    validation_size = int(len(dataset) * 0.2)
    train = dataset[0:train_size, :]
    # validation = dataset[train_size - look_back:train_size + validation_size, :]

    ####  循环测试

    # 创建dataset，使数据产生相关性
    X_train, y_train = create_dataset(train)
    # 添加到循环
    # X_validation, y_validation = create_dataset(validation)

    # 将数据转换成[样本，时间步长，特征]的形式
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # 添加到循环
    # X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))

    # 训练模型
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)  # 通过设置详细0,1或2,您只需说明您希望如何“看到”每个时期的训练进度. verbose = 0会显示任何内容(无声) verbose = 1会显示一个动画进度条,如下所示： progres_bar verbose = 2只会提到这样的纪元数：

    predict_xlist = dataset[train_size - look_back:train_size + validation_size, :]  # 添加预测x列表
    # predict_xlist = []
    predict_y = []  # 添加预测y列表
    predict_validation = []  #添加预测y列表
    # predict_xlist.extend(train.tolist())     # 已经存在的最后timesteps个数据添加进列表，预测新值
    while len(predict_y) < 12:
        i = 0
        # validation = np.array(predict_xlist[-timesteps:])
        validation = predict_xlist[-timesteps:, :]
        # 从最新的predict_xlist取出timesteps个数据，预测新的predict_steps个数据（因为每次预测的y会添加到predict_xlist列表中，为了预测将来的值，所以每次构造的x要取这个列表中最后的timesteps个数据词啊性）
        # validation = dataset[train_size - look_back:train_size + validation_size, :]
        X_validation, y_validation = create_dataset(validation)
        X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))  # 变换格式，适应LSTM模型
        # 模型预测数据
        predict_validation = model.predict(X_validation)
        # predict_xlist.extend(predict_validation[0])  # 将新预测出来的predict_steps个数据，加入predict_xlist列表，用于下次预测
        pre = predict_validation.astype('float32')# 12维
        predict_xlist = np.concatenate((predict_xlist, pre), axis=0)
        # np.concatenate((predict_xlist, pre))

        # 反标准化数据，目的是为了保证MSE的准确性
        predict_validation = scaler.inverse_transform(predict_validation)  # 预测的 19年游客的数据值
        y_validation = scaler.inverse_transform([y_validation])
        predict_y.extend(predict_validation[0])  # 预测的结果y，每次预测的1个数据，添加进去，

    #打印预测游客数量
    # print(year + '年预测游客数量:')
    # # print(predict_validation)
    # print(predict_y)

    ####循环测试

    # 评估模型
    validation_score = math.sqrt(mean_squared_error(y_validation[0], predict_validation[:, 0]))
    print('Validation Score : %.2f RMSE' % validation_score)

    # 构建通过评估数据集进行预测的图表数据
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:, :] = np.nan
    predict_validation_plot[train_size + validation_size - len(predict_validation): train_size + validation_size, :] = predict_validation

    # 图表显示
    dataset = scaler.inverse_transform(dataset)
    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.plot(dataset, color='black')
    plt.plot(predict_validation_plot, color='red')
    plt.show()

    return predict_validation

if __name__ == '__main__':
   # train(0.6,'2019')
   i = 0
   while i < 6:
    predict_validation = train(0.8,'2020')
    print('########## 2020年预测游客数量:', i)
    print(predict_validation)
    i += 1
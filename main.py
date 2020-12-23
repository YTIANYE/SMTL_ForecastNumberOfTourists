"""
LSTM时间序列问题预测：国际旅行人数预测
"""
import numpy as np
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
epochs = 2000
# filename = 'international-airline-passengers.csv'
filename = 'data_visitors.csv'
footer = 0
look_back = 12


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
    # train, validation = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    train = dataset[0:train_size, :]
    validation = dataset[train_size - look_back:train_size + validation_size, :]

    # 创建dataset，使数据产生相关性
    X_train, y_train = create_dataset(train)
    X_validation, y_validation = create_dataset(validation)

    # 将数据转换成[样本，时间步长，特征]的形式
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))

    # 训练模型
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)  # 通过设置详细0,1或2,您只需说明您希望如何“看到”每个时期的训练进度. verbose = 0会显示任何内容(无声) verbose = 1会显示一个动画进度条,如下所示： progres_bar verbose = 2只会提到这样的纪元数：

    # 模型预测数据
    # predict_train = model.predict(X_train)
    predict_validation = model.predict(X_validation)

    # 反标准化数据，目的是为了保证MSE的准确性
    # predict_train = scaler.inverse_transform(predict_train)
    # y_train = scaler.inverse_transform([y_train])
    predict_validation = scaler.inverse_transform(predict_validation)  # 预测的 19年游客的数据值
    y_validation = scaler.inverse_transform([y_validation])
    #打印预测游客数量
    print(year + '年预测游客数量:')
    print(predict_validation)

    # 评估模型
    # train_score = math.sqrt(mean_squared_error(y_train[0], predict_train[:, 0]))
    # print('Train Score: %.2f RMSE' % train_score)
    validation_score = math.sqrt(mean_squared_error(y_validation[0], predict_validation[:, 0]))
    print('Validation Score : %.2f RMSE' % validation_score)

    # 构建通过训练数据集进行预测的图表数据
    # predict_train_plot = np.empty_like(dataset)
    # predict_train_plot[:, :] = np.nan
    # predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train

    # 构建通过评估数据集进行预测的图表数据
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:, :] = np.nan
    # predict_validation_plot[len(predict_train) + look_back * 2 + 1: len(dataset) - 1 - forecast_size , :] = predict_validation
    # predict_validation_plot[len(dataset) - 1 - forecast_size - len(predict_validation) : len(dataset) - 1 - forecast_size, :] = predict_validation
    predict_validation_plot[train_size + validation_size - len(predict_validation): train_size + validation_size, :] = predict_validation

    # 图表显示
    dataset = scaler.inverse_transform(dataset)
    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.plot(dataset, color='black')
    # plt.plot(predict_train_plot, color='green')
    plt.plot(predict_validation_plot, color='red')

    plt.show()


if __name__ == '__main__':
    i = 0
    while i < 6:
        i += 1
        train(0.6,'2019')
   # train(0.8,'2020')
# Naive LSTM to learn three-char window to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
from sklearn.model_selection import train_test_split
from tensorflow import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf

numpy.random.seed(7)

seq_length = 3
dataX = []
dataY = []

Data_Test_X = []
Data_Test_Y = []

for data in open("../benchmarks/FB15K237/train2id_test.txt"):
    dataX.append([int(data.split(' ')[0].replace('\n', '')), int(data.split(' ')[1].replace('\n', '')),
                  int(data.split(' ')[2].replace('\n', ''))])
    dataY.append([int(data.split(' ')[2].replace('\n', ''))])

for data in open("../benchmarks/FB15K237/test2id_test.txt"):
    Data_Test_X.append([int(data.split(' ')[0].replace('\n', '')), int(data.split(' ')[1].replace('\n', '')),
                        int(data.split(' ')[2].replace('\n', ''))])
    Data_Test_Y.append([int(data.split(' ')[2].replace('\n', ''))])

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), 1, seq_length))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# test2id.txt
Test_X = numpy.reshape(Data_Test_X, (len(Data_Test_X), 1, seq_length))
Test_Y = np_utils.to_categorical(Data_Test_Y)

# normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30, shuffle=True)

# create and fit the model
model = Sequential()
model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

adam = optimizers.Adam(lr=0.0001)  # use Adam optimizer
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# build the model
history = model.fit(X_train, y_train, epochs=500, batch_size=200, validation_split=0.3, verbose=1, shuffle=True)
# save model
model.save(filepath="../checkpoints/LSTM_FB15K237", save_format='tf')

scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {} % '.format(scores[1]))

scores2 = model.evaluate(Test_X, Test_Y, verbose=0)
print('Accuracy on test data: {} %'.format(scores2[1]))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train accuracy', 'valid accuracy', 'train loss', 'valid loss'], loc='upper left')
plt.show()

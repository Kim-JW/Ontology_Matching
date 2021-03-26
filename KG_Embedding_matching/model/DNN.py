import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras import optimizers

X = []
y = []

for data in open("../benchmarks/FB15K237/train2id_test.txt"):
    X.append(int(data.split(' ')[0].replace('\n', '')))
    X.append(int(data.split(' ')[1].replace('\n', '')))
    y.append(int(data.split(' ')[2].replace('\n', '')))

X = numpy.array(X).reshape(272115, 2)
y = numpy.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, shuffle=True)

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=2))
model.add(Dropout(0.3))
model.add(Dense(300, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(237, activation='softmax'))

adam = optimizers.Adam(lr=0.0001)  # use Adam optimizer
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# build the model
history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.3, verbose=1, shuffle=True)

# save model
model.save(filepath="../checkpoints/DNN", save_format='tf')

scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {} % '.format(scores[1]))

scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {} %'.format(scores2[1]))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train accuracy', 'valid accuracy', 'train loss', 'valid loss'], loc='upper left')
plt.show()

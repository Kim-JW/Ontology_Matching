import numpy as np
from keras.utils import np_utils
from tensorflow import keras

model = keras.models.load_model("../checkpoints/LSTM_FB15K237")

Data_Test_X = []
Data_Test_Y = []
seq_length = 3
for data in open("../benchmarks/FB15K237/test2id_test.txt"):
    Data_Test_X.append([int(data.split(' ')[0].replace('\n', '')), int(data.split(' ')[1].replace('\n', '')),
                        int(data.split(' ')[2].replace('\n', ''))])
    Data_Test_Y.append([int(data.split(' ')[2].replace('\n', ''))])

# test2id.txt
Test_X = np.reshape(Data_Test_X, (len(Data_Test_X), 1, seq_length))
Test_Y = np_utils.to_categorical(Data_Test_Y)

scores2 = model.evaluate(Test_X, Test_Y, verbose=0)
print('Hit@1 : {} %'.format(scores2[1]))

predictions = model.predict(Test_X)
hit_3 = 0
hit_10 = 0
MR = 0
MRR = 0
keys = range(237)
for j in range(20466):
    dictionary = dict(zip(keys, predictions[j]))
    dic = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
    check_3 = [i for i in dic.keys()][234:]
    check_10 = [i for i in dic.keys()][227:]
    check_MR = [i for i in dic.keys()]
    if Data_Test_Y[j][0] in check_3:
        hit_3 = hit_3+1
    if Data_Test_Y[j][0] in check_10:
        hit_10 = hit_10+1
    if Data_Test_Y[j][0] in check_MR:
        MR = MR + check_MR.index(Data_Test_Y[j][0])
        MRR = MRR + 1/check_MR.index(Data_Test_Y[j][0])
    else:
        print("--------")
print('Hit@3 : {} %'.format(hit_3/20466))
print('Hit@10 : {} %'.format(hit_10/20466))
print('MR : {} '.format(MR/20466))
print('MRR : {} '.format(MRR))
# Generate arg maxes for predictions
# classes = np.argmax(predictions, axis=1)
# print(classes)
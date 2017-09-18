import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


with open ("kafka.txt", "r") as myfile:
    all_text=myfile.read().lower()

vocab = set(all_text)
print(len(all_text), len(vocab))

char_int_dict = dict((char, idx) for idx, char in enumerate(vocab))
int_char_dict = dict((idx, char) for idx, char in enumerate(vocab))

print(char_int_dict)
print(int_char_dict)

x = []
y = []
for i in range(len(all_text)-100):
    temp_x = all_text[i:i+100]
    temp_y = all_text[i+100]
    x.append([char_int_dict[char] for char in temp_x])
    y.append([char_int_dict[temp_y]])

train_x = np.reshape(x, (len(x), 100, 1))
train_x = train_x/float(len(all_text))
train_y = np_utils.to_categorical(y)

model = Sequential()
model.add(LSTM(256, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filename = "weights-char_lstm-24-2.952.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

seed_value = np.random.randint(0, len(x)-1)
seed = x[seed_value]
print(seed_value)
print("Seed :", [int_char_dict[_] for _ in seed])
for i in range(1000):
    test_x = np.reshape(seed, (1, len(seed), 1))
    test_x = test_x/float(len(all_text))
    prediction = model.predict(test_x, verbose=0)
    #index = np.argmax(prediction)
    index = np.random.choice(len(prediction[0]), p=prediction[0])
    result = int_char_dict[index]
    print(result, end=' ')
    seed.append(index)
    seed = seed[1:len(seed)]
print("The End")
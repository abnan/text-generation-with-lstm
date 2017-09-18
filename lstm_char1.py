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

print(char_int_dict)

x = []
y = []
for i in range(len(all_text)-100):
    temp_x = all_text[i:i+100]
    temp_y = all_text[i+100]
    x.append([char_int_dict[char] for char in temp_x])
    y.append([char_int_dict[temp_y]])

x = np.reshape(x, (len(x), 100, 1))
x = x/float(len(all_text))
y = np_utils.to_categorical(y)

model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-char_lstm-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x, y, epochs=30, batch_size=64, callbacks=callbacks_list)
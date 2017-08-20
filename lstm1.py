import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

with open ("kafka.txt", "r") as myfile:
    all_text=myfile.read().lower().replace('\n',' ').replace(',',' ,').replace('.',' .').replace('!',' !').replace('"',' "').replace('(',' (').replace(')',' )').replace('?',' ?')
words = list(set(all_text.split(' ')))

word_int={}
for idx, word in enumerate(words):
	word_int[word]=idx+1
#print(word_int)

story_int = [word_int[word] for word in all_text.split(' ')]
total_length = len(story_int)

x_train=[]
y_train=[]
for i in range(total_length-100):
	x_train.append(story_int[i:i+100])
	y_train.append(story_int[i+100])

print("Number of patterns = ", len(x_train))

x = np.reshape(x_train, (len(x_train), 100, 1))
# normalize
x = x / float(len(words))
# one hot encode the output variable
y = np_utils.to_categorical(y_train)



# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x, y, epochs=50, batch_size=64, callbacks=callbacks_list)
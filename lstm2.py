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
int_word={}
for idx, word in enumerate(words):
    word_int[word]=idx+1
    int_word[idx+1]=word
story_int = [word_int[word] for word in all_text.split(' ')]
total_length = len(story_int)

print("Number of fullstops ", story_int.count(word_int['.']))

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

filename = "weights-improvement-49-2.512.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

seed_value = np.random.randint(0, len(x_train)-1)
seed = x_train[seed_value]
print("Seed :", [int_word[x] for x in seed])
for i in range(300):
    test_x = np.reshape(seed, (1, len(seed), 1))
    test_x = test_x/float(len(words))
    prediction = model.predict(test_x, verbose=0)
    index = np.argmax(prediction)
    result = int_word[index]
    print(result, end=' ')
    seed.append(index)
    seed = seed[1:len(seed)]
print("The End")
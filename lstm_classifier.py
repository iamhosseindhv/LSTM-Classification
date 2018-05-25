import sys, os, re, csv, codecs
import numpy as np, pandas as pd
import time
# clean function for cleaning the dataset
from comment_cleaner import clean
#%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

start = time.time()

train = pd.read_csv('train.csv', encoding='latin-1')
test = pd.read_csv('test.csv', encoding='latin-1')
submission = pd.read_csv('sample_submission.csv')

print('Reading the dataset...')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"].apply(lambda comment: clean(comment))
list_sentences_test = test["comment_text"].apply(lambda comment: clean(comment))

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 100
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
# maxlen=200 as defined earlier
inp = Input(shape=(maxlen, ))

# size of the vector space
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

output_dimention = 60
x = LSTM(output_dimention, return_sequences=True,name='lstm_layer')(x)
# reduce dimention
x = GlobalMaxPool1D()(x)
# disable 10% precent of the nodes
x = Dropout(0.1)(x)
# pass output through a RELU function
x = Dense(50, activation="relu")(x)
# another 10% dropout
x = Dropout(0.1)(x)
# pass the output through a sigmoid layer, since 
# we are looking for a binary (0,1) classification 
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
# we use binary_crossentropy because of binary classification
# optimise loss by Adam optimiser
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training Model...')
start_fitting = time.time()
batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
fitting_model_time = time.time()
print('Training Model took: ', fitting_model_time - start_fitting)

print('Making Predictions...')
y_pred = model.predict(X_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('predictions.csv', index=False)
prediction_model_time = time.time()
print('Making Predictions took: ', prediction_model_time - fitting_model_time)

end = time.time()
print('TOTAL time spent', end-start)





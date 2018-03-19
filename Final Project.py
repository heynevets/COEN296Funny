"""
Team:
    Nicholas Fong
    Ting-Yu Yeh
    Nathan Kerr
    Brian Cox
COEN 296 Natural Language Processing Final Project
Created on 3/13/18

RNN from Keras to create jokes from joke training data
Training data taken from https://www.kaggle.com/abhinavmoudgil95/short-jokes/data
Another source to potentially use is https://github.com/taivop/joke-dataset

Vast majority of code copied from Trung Tran at https://github.com/ChunML/text-generator
with explanation given at https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
Thus, I must include:
MIT License

Copyright (c) 2016 Trung Tran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import print_function
import numpy as np
import math
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
import argparse

startTime = time.time()

# method for generating text
def generate_text(model, length, vocab_size, ix_to_char, inputText = ''):
	if(len(inputText) >= length):
		print("Length of output is less than the length of the inputText. Please fix.")
		return ('')
	X = np.zeros((1, length, vocab_size))
	y_char = []
	if(inputText == ''):
		# starting with random character
		ix = [np.random.randint(vocab_size)]
		y_char.append(ix_to_char[ix[-1]])
	else:
		for i in range(len(inputText)):
			ix = np.array([list(ix_to_char.keys())[list(ix_to_char.values()).index(inputText[i])]])
			y_char.append(ix_to_char[ix[-1]])
			if(i != len(inputText) - 1):
				X[0, i, :][ix[-1]] = 1
				print(ix_to_char[ix[-1]], end="")
	for i in range(len(inputText), length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		print(ix_to_char[ix[-1]], end="")
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)

# method for preparing the training data
def load_data(data_dir, seq_length):
	data = open(data_dir, 'r').read()
	chars = list(set(data))
	VOCAB_SIZE = len(chars)

	print('Data length: {} characters'.format(len(data)))
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

	X = np.zeros((math.ceil(len(data)/seq_length), seq_length, VOCAB_SIZE))
	y = np.zeros((math.ceil(len(data)/seq_length), seq_length, VOCAB_SIZE))
	for i in range(0, math.floor(len(data)/seq_length)):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='shortJokesTruncated.csv')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=500)
ap.add_argument('-nb_epoch', type=int, default=30)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
NUM_EPOCHS = args['nb_epoch']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Creating training data
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
	model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Generate some sample before training to know how bad it is!
generate_text(model, args['generate_length'], VOCAB_SIZE, ix_to_char)

if(WEIGHTS != ''):
	model.load_weights(WEIGHTS)
	nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
	nb_epoch = 0

# Training if there is no trained weights specified
if(args['mode'] == 'train' or WEIGHTS == ''):
	i = 0
	while i < NUM_EPOCHS:
		i += 1
		print('\n\nEpoch: {}\n'.format(nb_epoch))
		model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
		nb_epoch += 1
		generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
		if nb_epoch % 5 == 0:
			model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif(args['mode'] == 'generate'):
	# Loading the trained weights
	#model.load_weights(WEIGHTS)
	generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
	print('\n\n')
else:
	print('\n\nNothing to do!')

print('time taken =', round(time.time() - startTime), 'seconds')
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def make_model(input_dim=4, output_dim=2):
	model = Sequential()      
	# WRITE CODE HERE
	# Add layers to the model:
	# a fully connected layer with 10 units
	# a tanh activation
	# another fully connected layer with out_dim (the number of actions)
	# a softmax activation (so the output is a proper distribution)
	# END

	model.compile(loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])
	model.build((1, input_dim))

	# We expect the model to have four weight variables (a kernel and bias for
	# both layers)
	assert len(model.weights) == 4, 'Model should have 4 weights.'
	return model


def test_model():
	model = make_model()
	for t in range(20):
		X = np.random.normal(size=(1000, 4))  # some random data
		is_positive = np.sum(X, axis=1) > 0  # A simple binary function
		Y = np.zeros((1000, 2))
		Y[np.arange(1000), is_positive.astype(int)] = 1  # one-hot labels
		history = model.fit(X, Y, epochs=10, batch_size=256, verbose=0)
		loss = history.history['loss'][-1]
		acc = history.history['accuracy'][-1]
		print('(%d) loss= %.3f; accuracy = %.1f%%' % (t, loss, 100 * acc))

if __name__ == '__main__':
	test_model()
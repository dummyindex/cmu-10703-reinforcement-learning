import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np

class FullyConnectedModel(torch.nn.Module):
	def __init__(self, input_dim=4, output_dim=2):
		super().__init__()
		# WRITE CODE HERE
		# Add layers to the model:
		# a fully connected layer with 10 units
		# a tanh activation
		# another fully connected layer with out_dim (the number of actions)
		#
		# Do not include a softmax activation function if using Pytorch!
		# Pytorch's implementation of the CrossEnropyLoss() assumes that
		# the network has not applied a softmax to the output.
		# END

		hidden_dim = 10
		self.fc_0 = torch.nn.Linear(input_dim, hidden_dim)
		self.tanh = torch.nn.Tanh()
		self.fc_1 = torch.nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		# WRITE CODE HERE
		# END
		x = self.fc_0(x)
		x = self.tanh(x)
		x = self.fc_1(x)
		return x


def make_model(input_dim=4, output_dim=2):
	model = FullyConnectedModel(input_dim, output_dim)
	# We expect the model to have four weight variables (a kernel and bias for
	# both layers)
	assert len([p for p in model.parameters()]) == 4, 'Model should have 4 weights.'
	return model


def test_model():
	model = make_model()
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	for t in range(20):
		X = np.random.normal(size=(1000, 4))  # some random data
		is_positive = np.sum(X, axis=1) > 0  # A simple binary function
		Y = np.zeros((1000, 2))
		Y[np.arange(1000), is_positive.astype(int)] = 1 # One hot labels
		Y = np.argmax(Y, axis=1) # Class labels

		train_set = TensorDataset(torch.Tensor(X), torch.Tensor(Y).type(torch.long))
		train_loader = DataLoader(dataset=train_set, batch_size = 256, shuffle=True)

		for epoch in range(10):
			running_loss = 0
			correct = 0
			for i, data in enumerate(train_loader, 0):
				x_batch, y_batch = data

				optimizer.zero_grad()
				yhat = model(x_batch)
				loss = criterion(yhat, y_batch)
				loss.backward()
				optimizer.step()

				correct += (torch.argmax(yhat, dim=1) == y_batch).float().sum()
				running_loss += loss.item()

		acc = correct / len(train_set)
		print('(%d) loss= %.3f; accuracy = %.1f%%' % (t, loss, 100 * acc))

if __name__ == '__main__':
	test_model()

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivate_sigmoid(x):
	return x * (1 - x)

learning_rate = 0.01
momentum = 0.5
input_neurons = [0] * 3
hidden_neurons = [0] * 3
weights_ih = np.zeros(shape=(3,3))
weights_ih_delta = np.zeros(shape=(3,3))
prev_weights_ih = np.zeros(shape=(3,3))
biases_ih = [0] * 3
prev_biases_ih = [0] * 3
hidden_gradients = [0] * 3
output_neurons = [0] * 2
weights_ho = np.zeros(shape=(3,2))
weights_ho_delta = np.zeros(shape=(3,2))
prev_weights_ho = np.zeros(shape=(3,2))
biases_ho = [0] * 2
prev_biases_ho = [0] * 2
output_gradients = [0] * 2
error = 1

np.random.seed(1)

for r in range(len(input_neurons)):
	for c in range(len(hidden_neurons)):
		weights_ih[r][c] = np.random.uniform(-0.5, 0.5)

for r in range(len(hidden_neurons)):
	for c in range(len(output_neurons)):
		weights_ho[r][c] = np.random.uniform(-0.5, 0.5)

for i in range(len(biases_ih)):
	biases_ih[i] = np.random.uniform(-0.5, 0.5)

for i in range(len(biases_ho)):
	biases_ho[i] = np.random.uniform(-0.5, 0.5)

# print(weights_ih)



#wykres
xdata = []
ydata = []

plt.show()
axes = plt.gca()

iteration_num = 0

# for i in range(1000):
while error > 0.1:
	if iteration_num % 2 == 0:
		input_neurons[0] = 0.21
		input_neurons[1] = 0.31
		input_neurons[2] = 0.17
		target = [0.0, 1.0]
	else :
		input_neurons[0] = 0.74
		input_neurons[1] = 0.89
		input_neurons[2] = 0.91
		target = [1.0, 0.0]

	hidden_neurons = np.dot(input_neurons, weights_ih)
	hidden_neurons += biases_ih
	# print(hidden_neurons)

	for h in range(len(hidden_neurons)):
		hidden_neurons[h] = sigmoid(hidden_neurons[h])

	# print("Wagi i-h:", weights_ih)
	# print("Neurony ukryte:", hidden_neurons)

	output_neurons = np.dot(hidden_neurons, weights_ho)
	output_neurons += biases_ho

	# print("Wagi h-o:", weights_ho)
	for o in range(len(output_neurons)):
		output_neurons[o] = sigmoid(output_neurons[o])

	error = np.mean(np.abs(target - output_neurons))
	print("Neurony wyjsciowe:", output_neurons)
	# print("Blad:",error)



	for o in range(len(output_gradients)):
		output_gradients[o] = (target[o] - output_neurons[o])*derivate_sigmoid(output_neurons[o])

	# print("Gradienty wyjsciowe:", output_gradients)

	for h in range(len(hidden_gradients)):
		hidden_gradients = np.dot(output_gradients, weights_ho.T)
		for h in range(len(hidden_gradients)):
			hidden_gradients[h] *= derivate_sigmoid(hidden_neurons[h])
		# print("Gradienty ukryte:", hidden_gradients)

	for r in range(len(input_neurons)):
		for c in range(len(hidden_neurons)):
			weights_ih_delta[r][c] = learning_rate * hidden_gradients[c] * input_neurons[r]
			weights_ih[r][c] += weights_ih_delta[r][c]
	# 		weights_ih[r][c] -= momentum * prev_weights_ih[r][c]
	# prev_weights_ih = weights_ih
	# print(weights_ih)

	for r in range(len(hidden_neurons)):
		for c in range(len(output_neurons)):
			weights_ho_delta[r][c] = learning_rate * output_gradients[c] * hidden_neurons[r]
			weights_ho[r][c] += weights_ho_delta[r][c]
	# 		weights_ho[r][c] -= momentum * prev_weights_ho[r][c]
	# prev_weights_ho = weights_ho

	for i in range(len(biases_ih)):
		biases_ih[i] += learning_rate * hidden_gradients[i] * 1.0
	# 	biases_ih[i] -= momentum * prev_biases_ih[i]
	# prev_biases_ih = biases_ih
	# print(biases_ih)

	for i in range(len(biases_ho)):
		biases_ho[i] += learning_rate * output_gradients[i] * 1.0
	# 	biases_ho[i] -= momentum * prev_biases_ho[i] TUTAJ JEST JAKIS BLAD
	# prev_biases_ho = biases_ho 
	# print(biases_ho)
	xdata.append(iteration_num)
	ydata.append(error)
	iteration_num += 1

plt.semilogx(xdata, ydata, label='linear')
plt.show()


	


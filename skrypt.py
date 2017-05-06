import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivate_sigmoid(x):
	return x * (1 - x)

# Dane konfiguracyjne
input_neurons_cnt = 3
hidden_neurons_cnt = 3
output_neurons_cnt = 2
learning_rate = 0.01
momentum = 0.7
filename = 'data.csv'

# Wspolczynnik okreslajacy stosunek liczby wektorow danych uzytych do treningu sieci do liczby wektorow uzytych do testowania sieci
train2test_ratio = 0.1

# Poszukiwana dokladnosc pracy sieci neuronowej
desired_max_error = 0.001

# Inicjalizacja tablic i list
input_neurons = [0] * input_neurons_cnt
hidden_neurons = [0] * hidden_neurons_cnt
output_neurons = [0] * output_neurons_cnt

weights_ih = np.zeros(shape=(input_neurons_cnt,hidden_neurons_cnt))
weights_ih_delta = np.zeros(shape=(input_neurons_cnt,hidden_neurons_cnt))
prev_weights_ih_delta = np.zeros(shape=(input_neurons_cnt,hidden_neurons_cnt))
biases_ih = [0] * hidden_neurons_cnt
biases_ih_delta = [0] * hidden_neurons_cnt
prev_biases_ih_delta = [0] * hidden_neurons_cnt
hidden_gradients = [0] * hidden_neurons_cnt

weights_ho = np.zeros(shape=(hidden_neurons_cnt,output_neurons_cnt))
weights_ho_delta = np.zeros(shape=(hidden_neurons_cnt,output_neurons_cnt))
prev_weights_ho_delta = np.zeros(shape=(hidden_neurons_cnt,output_neurons_cnt))
biases_ho = [0] * output_neurons_cnt
biases_ho_delta = [0] * output_neurons_cnt
prev_biases_ho_delta = [0] * output_neurons_cnt
output_gradients = [0] * output_neurons_cnt


# Pobranie danych z pliku 
data = np.genfromtxt(filename, delimiter=',', skip_header=1, skip_footer=0, names=['target', 'x', 'y', 'z'])

# Skalowanie danych do zakresu [-1;1]
max_value = max(max(abs(data['x'])), max(abs(data['y'])), max(abs(data['z'])))
vectorX = data['x'] / max_value
vectorY = data['y'] / max_value
vectorZ = data['z'] / max_value

# Zapisanie danych treningowych do wektorow
T1X = vectorX[0:math.ceil(2499*train2test_ratio)]
T1Y	= vectorY[0:math.ceil(2499*train2test_ratio)]
T1Z	= vectorZ[0:math.ceil(2499*train2test_ratio)]
T2X	= vectorX[2500:math.ceil(2500+train2test_ratio*2499)]
T2Y	= vectorY[2500:math.ceil(2500+train2test_ratio*2499)]
T2Z	= vectorZ[2500:math.ceil(2500+train2test_ratio*2499)]

data_size = 2*len(T1X)

np.random.seed(1)

# Wypelnienie tablic z wagami wartosciami losowymi w przedziale [-0.5; 0.5]
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

# Inicjacja tablic sluzacych do sporzadzenia wykresu zaleznosci bledu od numeru iteracji
x_iteration = []
y_error = []

iteration_num = 0
error = 999

# Poczatek treningu sieci
while error > desired_max_error:
	n = 0
	error = 0
	# Przejscie w danej iteracji po wszystkich treningowych wektorach danych
	while (n < data_size):
		#Co drugi wektor danych treningowych ma taki sam target
		if n%2 == 0:
			input_neurons[0] = T1X[math.floor(n/2)]
			input_neurons[1] = T1Y[math.floor(n/2)]
			input_neurons[2] = T1Z[math.floor(n/2)]
			target = [1.0, 0.0]
		else :
			input_neurons[0] = T2X[math.floor(n/2)]
			input_neurons[1] = T2Y[math.floor(n/2)]
			input_neurons[2] = T2Z[math.floor(n/2)]
			target = [0.0, 1.0]

		# print("Neurony wejsciowe: ",input_neurons, ", Target: ", target)

		hidden_neurons = np.dot(input_neurons, weights_ih)
		hidden_neurons += biases_ih
		for h in range(len(hidden_neurons)):
			hidden_neurons[h] = sigmoid(hidden_neurons[h])

		output_neurons = np.dot(hidden_neurons, weights_ho)
		output_neurons += biases_ho
		for o in range(len(output_neurons)):
			output_neurons[o] = sigmoid(output_neurons[o])

		error += np.mean(np.abs(target - output_neurons))
		# print("Neurony wyjsciowe:", output_neurons)
		# print("Blad:",error)

		for o in range(len(output_gradients)):
			output_gradients[o] = (target[o] - output_neurons[o])*derivate_sigmoid(output_neurons[o])
		# print("Gradienty wyjsciowe:", output_gradients)

		hidden_gradients = np.dot(output_gradients, weights_ho.T)
		for h in range(len(hidden_gradients)):
			hidden_gradients[h] *= derivate_sigmoid(hidden_neurons[h])
		# print("Gradienty ukryte:", hidden_gradients)

		for r in range(len(input_neurons)):
			for c in range(len(hidden_neurons)):
				weights_ih_delta[r][c] = learning_rate * hidden_gradients[c] * input_neurons[r] + momentum * prev_weights_ih_delta[r][c]
				weights_ih[r][c] += weights_ih_delta[r][c]
		prev_weights_ih_delta = weights_ih_delta
		# print(weights_ih)

		for r in range(len(hidden_neurons)):
			for c in range(len(output_neurons)):
				weights_ho_delta[r][c] = learning_rate * output_gradients[c] * hidden_neurons[r] + momentum * prev_weights_ho_delta[r][c]
				weights_ho[r][c] += weights_ho_delta[r][c]
		prev_weights_ho_delta = weights_ho_delta

		for i in range(len(biases_ih)):
			biases_ih_delta[i] = learning_rate * hidden_gradients[i] * 1.0 + momentum * prev_biases_ih_delta[i]
			biases_ih[i] += biases_ih_delta[i]
		prev_biases_ih_delta = biases_ih_delta
		# print(biases_ih)

		for i in range(len(biases_ho)):
			biases_ho_delta[i] = learning_rate * output_gradients[i] * 1.0 + momentum * prev_biases_ho_delta[i]
			biases_ho[i] += biases_ho_delta[i]
		prev_biases_ho_delta = biases_ho_delta
		# print(biases_ho)
		n = n+1
	error /= data_size
	x_iteration.append(iteration_num)
	y_error.append(error)		
	if iteration_num % 10 == 0:
		print("Iteracja ", iteration_num, ", blad:",error)
	iteration_num += 1

print("Liczba iteracji:", iteration_num)
# plt.plot(x_iteration, y_error, label='linear')
plt.semilogx(x_iteration, y_error, label='linear')
plt.xlabel('Numer iteracji')
plt.ylabel('Blad')
plt.show()



# Testowanie sieci
# Pobierane sa pozostale wektory danych wejsciowych 
T1X = vectorX[math.ceil(2499*train2test_ratio):2499]
T1Y	= vectorY[math.ceil(2499*train2test_ratio):2499]
T1Z	= vectorZ[math.ceil(2499*train2test_ratio):2499]
T2X	= vectorX[math.ceil(2500+train2test_ratio*2499):4999]
T2Y	= vectorY[math.ceil(2500+train2test_ratio*2499):4999]
T2Z	= vectorZ[math.ceil(2500+train2test_ratio*2499):4999]


prediction = [0] * output_neurons_cnt
predicting_errors = 0
data_size = 2*len(T1X)
n = 0

# Petla sprawdzajaca wszystkie pobranych danych testowych
while (n < data_size):
	if n<data_size/2:
		input_neurons[0] = T1X[math.floor(n/2)]
		input_neurons[1] = T1Y[math.floor(n/2)]
		input_neurons[2] = T1Z[math.floor(n/2)]
		target = [1.0, 0.0]
	else :
		input_neurons[0] = T2X[math.floor(n/2)]
		input_neurons[1] = T2Y[math.floor(n/2)]
		input_neurons[2] = T2Z[math.floor(n/2)]
		target = [0.0, 1.0]	

	hidden_neurons = np.dot(input_neurons, weights_ih)
	hidden_neurons += biases_ih
	for h in range(len(hidden_neurons)):
		hidden_neurons[h] = sigmoid(hidden_neurons[h])

	output_neurons = np.dot(hidden_neurons, weights_ho)
	output_neurons += biases_ho
	for o in range(len(output_neurons)):
		output_neurons[o] = sigmoid(output_neurons[o])
		if output_neurons[o] > 0.5:
			prediction[o] = 1.0
		else :
			prediction[o] = 0.0

	print("Nerony wejsciowe:", input_neurons)
	print("Neurony wyjsciowe:", output_neurons)
	print("Predykcja:",prediction)
	print("Cel:", target)
	print("************")
		
	if target != prediction :
		predicting_errors += 1
		print("#############")
		print("Dodaje nowy blad!")
		print("#############")

	n += 1
	if n % 500 == 0 :
		print("Przetestowalem juz", n, "zestawow danych.")

	

print("Bledy predykcji klasyfikacji danych: ", predicting_errors, "/", data_size)





import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivate_sigmoid(x):
	return x * (1 - x)


#W ostatnim etapie projektu należy wykonać eksperymenty przy użyciu sieci:
#Należy sprawdzić jak zachowuje się nauka sieci dla minimum 3 różnych parametrów uczenia sieci (3 różne wartości współczynnika szybkości uczenia - learning rate)
#Należy sprawdzić jakie uzyskuje się wyniki przy użyciu minimum 3 różnych architektur sieci (różna liczba neuronów w sieci)




# Dane konfiguracyjne
input_neurons_cnt = 3
hidden_neurons_cnt = 11
output_neurons_cnt = 2
learning_rate = 0.01
momentum = 0.7
filename = 'data.csv'

# Wspolczynnik okreslajacy stosunek liczby wektorow danych uzytych do treningu sieci do liczby wektorow uzytych do testowania sieci
train2test_ratio = 0.8

# Poszukiwana dokladnosc pracy sieci neuronowej
desired_max_error = 0.1

# Inicjalizacja tablic i list, neurony wejsciowe i ukryte powiekszone o 1 poniewaz wykorzystujemy bias
input_neurons = [0] * (input_neurons_cnt + 1)
hidden_neurons = [0] * (hidden_neurons_cnt + 1)
output_neurons = [0] * output_neurons_cnt

# Wartosc "nadmiarowego" neuronu wejsciowego ustawiona na stale na 1 - bias
input_neurons[len(input_neurons) - 1] = 1.0

weights_ih = np.zeros(shape=(input_neurons_cnt + 1,hidden_neurons_cnt + 1))
weights_ih_delta = np.zeros(shape=(input_neurons_cnt + 1,hidden_neurons_cnt + 1))
prev_weights_ih_delta = np.zeros(shape=(input_neurons_cnt + 1,hidden_neurons_cnt + 1))
hidden_gradients = [0] * (hidden_neurons_cnt + 1)

weights_ho = np.zeros(shape=(hidden_neurons_cnt + 1,output_neurons_cnt))
weights_ho_delta = np.zeros(shape=(hidden_neurons_cnt + 1,output_neurons_cnt))
prev_weights_ho_delta = np.zeros(shape=(hidden_neurons_cnt + 1,output_neurons_cnt))
biases_ho = [0] * output_neurons_cnt
biases_ho_delta = [0] * output_neurons_cnt
prev_biases_ho_delta = [0] * output_neurons_cnt
output_gradients = [0] * output_neurons_cnt

# Inicjalizacja tablicy przechowujacej wynik poszczegolnych testow sieci
prediction = [0] * output_neurons_cnt

# Inicjacja tablic sluzacych do sporzadzenia wykresu zaleznosci bledu od numeru iteracji
x_iteration = []
y_error = []

# Pobranie danych z pliku 
data = np.genfromtxt(filename, delimiter=',', skip_header=1, skip_footer=0, names=['target', 'x', 'y', 'z'])

# Skalowanie danych do zakresu [-1;1]
max_value = max(max(abs(data['x'])), max(abs(data['y'])), max(abs(data['z'])))
vectorX = data['x'] / max_value
vectorY = data['y'] / max_value
vectorZ = data['z'] / max_value
print(max_value)

data_length = len(vectorX)

T1X = vectorX[0:2499]
T1Y	= vectorY[0:2499]
T1Z	= vectorZ[0:2499]
T2X	= vectorX[2500:4999]
T2Y	= vectorY[2500:4999]
T2Z	= vectorZ[2500:4999]

T1X = T1X.tolist()
T1Y = T1Y.tolist()
T1Z = T1Z.tolist()
T2X = T2X.tolist()
T2Y = T2Y.tolist()
T2Z = T2Z.tolist()


T1X_training = []
T1Y_training = []
T1Z_training = []
T2X_training = []
T2Y_training = []
T2Z_training = []

np.random.seed(1)

# Losowo wybierana jest okreslona liczba wektorow danych dla obu targetow
# W tablicach T** pozostana dane do testowania
for i in range(int(data_length * train2test_ratio / 2)):
	index = np.random.random_integers(0, len(T1X)-1)
	T1X_training.append(T1X[index])
	T1Y_training.append(T1Y[index])
	T1Z_training.append(T1Z[index])
	T2X_training.append(T2X[index])
	T2Y_training.append(T2Y[index])
	T2Z_training.append(T2Z[index])
	T1X.pop(index)
	T1Y.pop(index)
	T1Z.pop(index)
	T2X.pop(index)
	T2Y.pop(index)
	T2Z.pop(index)

# Wypelnienie tablic z wagami wartosciami losowymi w przedziale [-0.5; 0.5]
for r in range(len(input_neurons)):
	for c in range(len(hidden_neurons)):
		weights_ih[r][c] = np.random.uniform(-0.5, 0.5)

for r in range(len(hidden_neurons)):
	for c in range(len(output_neurons)):
		weights_ho[r][c] = np.random.uniform(-0.5, 0.5)


iteration_num = 0
epoch_error = 1


# Poczatek treningu sieci
while epoch_error > desired_max_error:
	n = 0
	epoch_error = 0
	# Przejscie w danej iteracji po wszystkich treningowych wektorach danych
	while (n < data_length * train2test_ratio):
		if n%2 == 0:
			input_neurons[0] = T1X_training[np.math.floor(n/2)]
			input_neurons[1] = T1Y_training[np.math.floor(n/2)]
			input_neurons[2] = T1Z_training[np.math.floor(n/2)]
			target = [1.0, 0.0]
		else :
			input_neurons[0] = T2X_training[np.math.floor(n/2)]
			input_neurons[1] = T2Y_training[np.math.floor(n/2)]
			input_neurons[2] = T2Z_training[np.math.floor(n/2)]
			target = [0.0, 1.0]

		# print("Neurony wejsciowe: ",input_neurons, ", Target: ", target)

		hidden_neurons = np.dot(input_neurons, weights_ih)
		for h in range(len(hidden_neurons)):
			hidden_neurons[h] = sigmoid(hidden_neurons[h])

		output_neurons = np.dot(hidden_neurons, weights_ho)
		for o in range(len(output_neurons)):
			output_neurons[o] = sigmoid(output_neurons[o])

		error = target - output_neurons
		epoch_error += np.sum(0.5*np.power(error, 2))

		for o in range(len(output_gradients)):
			output_gradients[o] = (target[o] - output_neurons[o])*derivate_sigmoid(output_neurons[o])

		hidden_gradients = np.dot(output_gradients, weights_ho.T)
		for h in range(len(hidden_gradients)):
			hidden_gradients[h] *= derivate_sigmoid(hidden_neurons[h])

		for r in range(len(input_neurons)):
			for c in range(len(hidden_neurons)):
				weights_ih_delta[r][c] = learning_rate * hidden_gradients[c] * input_neurons[r] + momentum * prev_weights_ih_delta[r][c]
				weights_ih[r][c] += weights_ih_delta[r][c]
		prev_weights_ih_delta = weights_ih_delta

		for r in range(len(hidden_neurons)):
			for c in range(len(output_neurons)):
				weights_ho_delta[r][c] = learning_rate * output_gradients[c] * hidden_neurons[r] + momentum * prev_weights_ho_delta[r][c]
				weights_ho[r][c] += weights_ho_delta[r][c]
		prev_weights_ho_delta = weights_ho_delta

		n = n+1
	epoch_error /= data_length * train2test_ratio
	x_iteration.append(iteration_num)
	y_error.append(epoch_error)		
	if iteration_num % 10 == 0:
		print("Iteracja ", iteration_num, ", blad:",epoch_error)
	iteration_num += 1

print("Liczba iteracji potrzebnych do osiagniecia zalozonego poziomu bledu:", iteration_num)
plt.plot(x_iteration, y_error)
plt.xlabel('Numer iteracji')
plt.ylabel('Blad')
plt.grid(True)
plt.show()

#### TESTOWANIE ####

predicting_errors = 0
data_size = 2*len(T1X)
n = 0

# Petla sprawdzajaca wszystkie pobranych danych testowych
while (n < data_size):
	if n<data_size/2:
		input_neurons[0] = T1X[n]
		input_neurons[1] = T1Y[n]
		input_neurons[2] = T1Z[n]
		target = [1.0, 0.0]
	else :
		input_neurons[0] = T2X[int(n - data_size/2)]
		input_neurons[1] = T2Y[int(n - data_size/2)]
		input_neurons[2] = T2Z[int(n - data_size/2)]
		target = [0.0, 1.0]	

	hidden_neurons = np.dot(input_neurons, weights_ih)
	for h in range(len(hidden_neurons)):
		hidden_neurons[h] = sigmoid(hidden_neurons[h])

	output_neurons = np.dot(hidden_neurons, weights_ho)
	for o in range(len(output_neurons)):
		output_neurons[o] = sigmoid(output_neurons[o])
		if output_neurons[o] > 0.5:
			prediction[o] = 1.0
		else :
			prediction[o] = 0.0

	# print("Nerony wejsciowe:", input_neurons)
	# print("Neurony wyjsciowe:", output_neurons)
	# print("Predykcja:",prediction)
	# print("Cel:", target)
	# print("************")
		
	if target != prediction :
		predicting_errors += 1

	n += 1
	if n % 500 == 0 :
		print("Przetestowalem juz", n, "zestawow danych,", predicting_errors,"/",n,"bledow.")

	

print("Bledy predykcji klasyfikacji danych: ", predicting_errors, "/", data_size)





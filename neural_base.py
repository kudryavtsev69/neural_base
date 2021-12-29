import numpy as np

def sigmoid(x):   # Формула расчета порогвых значений (сигмоида)
    return 1/(1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0, 1, 1, 0]]).T  # Значения ответов

np.random.seed(1)   # Генератор случайных значений

synaptic_weights = 2 * np.random.random((3, 1)) - 1   # Определение весов

print('Случайно инициализированные веса: ')
print(synaptic_weights)

# Метод обратного распространения
for i in range(200):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    arguments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += arguments

print('Веса после обучения:')
print(synaptic_weights)
print('Результат:')
print(outputs)
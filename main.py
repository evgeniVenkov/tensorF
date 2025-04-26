
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ----------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def one_simple_ii():
    inputs = [28, 85]
    weights = [0.5, 0.5]
    bias = -30

    weighted_inputs = [inputs[i] * weights[i] for i in range(len(inputs))]

    total_input = sum(weighted_inputs) + bias
    activated_output = sigmoid(total_input)
    print("Результат работы нейрона:", activated_output)
def tensor_fn():
    tensor1 = tf.constant([1,2,3,4])
    tensor2 = tf.constant([6,7,8,9])

    ress_add = tf.add(tensor1, tensor2)
    ress_mul = tf.multiply(tensor1, tensor2)
    ress_div = tf.divide(tensor1, tensor2)
    ress_sub = tf.subtract(tensor1, tensor2)
    ress_reshape = tf.reshape(ress_add, [2,2])

    print("Результат сложения:", ress_add)
    print("Результат умножения:", ress_mul)
    print("Результат деления:", ress_div)
    print("Результат вычитания:", ress_sub)
    print("Измененная форма тензора:", ress_reshape)
def tensor_fn_two():
    tensor1 = tf.constant([1, 2, 3, 4, 5])
    print("Тензор 1:", tensor1)

    tensor2 = tf.zeros([3, 3])
    print("Тензор 2 (ноль):", tensor2)

    tensor3 = tf.ones([2, 2])
    print("Тензор 3 (единицы):", tensor3)

    tensor4 = tf.random.normal([2, 2], mean=0, stddev=1)
    print("Тензор 4 (случайные значения):", tensor4)
def two_simple_ii():
    # эта хуета работает не правильно
    x = np.arange(1, 21, dtype=np.float32)
    y = x ** 2

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(x,y, epochs=1000, verbose=0)
    print("Обучение завершено!")

    x_test = np.array([7, 8, 9], dtype=np.float32)

    predictions = model.predict(x_test)

    print("Входные данные:", x_test)
    print("Прогнозы:", predictions.flatten())
def three_simple_ii():
    x1 = np.array([5, 10, 15, 20, 25], dtype=np.float32)
    x2 = np.array([50, 100, 150, 200, 250], dtype=np.float32)

    x = np.stack([x1, x2], axis=1)
    y = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float32)

    model = keras.Sequential([
        layers.Dense(128,activation='relu',input_shape = (2,)),
        layers.Dense(64,activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(x,y, epochs=1000, verbose=0)
    print("Обучение завершено!")

    x1_test = np.array([8, 12, 18])
    x2_test = np.array([60, 120, 180])
    x_test = np.stack([x1_test, x2_test], axis=1)

    predictions = model.predict(x_test)

    print("Входные данные:", x_test)
    print("Прогнозы:", predictions.flatten())
    print("ожидаемый выход: [800, 1200, 1800]")

gpus = tf.config.list_physical_devices('GPU')
print("Доступные GPU:", gpus)


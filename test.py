import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


x1 = np.array([5, 10, 15, 20, 25])
x2 = np.array([1, 2, 3, 4, 5])
x = np.stack([x1, x2], axis=1)
y = np.array([5, 20, 45, 80, 125])

model = keras.Sequential([
    layers.Dense(1, input_shape=(2,), use_bias=False)  # без смещения (чистая линейная зависимость)
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.fit(x,y, epochs=1000,verbose=0)

x1_test = np.array([7, 12, 18])
x2_test = np.array([2, 3, 4])
x_test = np.stack([x1_test, x2_test], axis=1)
y_pred = model.predict(x_test)

print("Входные данные:", x_test)
print("Прогнозы:", y_pred.flatten())
print("ожидаемый выход: [[14, 36, 72]]")
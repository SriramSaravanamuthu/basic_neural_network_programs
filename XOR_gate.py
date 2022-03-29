import numpy as np
import tensorflow as tf

x = np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
y = np.array([[0],[1],[1],[0]],"float32")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(4, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['binary_accuracy'])

model.fit(x,y,epochs=1000)
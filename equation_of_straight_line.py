import tensorflow as tf
import numpy as np
x=np.array([0,1,2,3,4,5,6],float)
y=2*x+3
model=tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1]),
                          tf.keras.layers.Dense(units=1)])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(x,y,epochs=500)
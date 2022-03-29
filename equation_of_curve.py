
import tensorflow as tf
import numpy as np

#x=np.array([1,2,3,4,5,6,7,8],float) #=B8^3+B8^2+B8+10
#y=np.array([13,24,49,94,165,268,409,594],float)
x=np.array([0,1,2,3,4,5],float)   #=B9^2+10
y=np.array([10,11,14,19,26,35],float)

#model = tf.keras.models.Sequential()
#model.add(keras.layers.Dense(1,[1]))
#model.add(keras,layers.Dense(2,'relu'))
#model.add(keras,layers.Dense(1,'softmax'))
#model.compile('sgd','mean_squared_error'))
model=tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1]),
                          tf.keras.layers.Dense(units=5),
                           tf.keras.layers.Dense(units=5),
                           #tf.keras.layers.Dense(units=2),
                           tf.keras.layers.Dense(units=5),
                          tf.keras.layers.Dense(units=1)])
model.compile(optimizer='adam',loss='mean_squared_logarithmic_error')

model.fit(x,y,epochs=1000)
""" neural network that predicts the price of a house according to a simple formula """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    #training data
    xs =np.array([1,2,3,4,5,6,8,9,10],dtype=float)
    ys =np.array([1,1.5,2,2.5,3,3.5,4.5,5,5.5],dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer="sgd",loss="mean_squared_error")
    model.fit(xs,ys,epochs=2000)
    return model.predict(y_new)[0]


prediction = house_model([7.0])
print(prediction)


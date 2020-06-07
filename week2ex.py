
""" Write an MNIST classifier (for handwritten digit recognition) that trains to 99% accuracy or above, and does it without a fixed number of epochs 
-- i.e. you should stop training once you reach that level of accuracy. """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

path = r'C:\Users\SIDDHI\Desktop\desktop\python\TensorflowInPractice\utf-8''mnist.npz'

def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
  class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("acc")>0.99):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True
  # YOUR CODE SHOULD END HERE

  mnist = tf.keras.datasets.mnist

  (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
  # YOUR CODE SHOULD START HERE
  x_train, x_test = x_train / 255.0, x_test / 255.0
  callbacks = myCallback()
  # YOUR CODE SHOULD END HERE
  model = tf.keras.models.Sequential([
      # YOUR CODE SHOULD START HERE
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(1024, activation=tf.nn.relu),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax),

      # YOUR CODE SHOULD END HERE
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['acc']) 

# model fitting
  history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
  
  a,b=history.epoch, history.history['acc'][-1]
  print("Number of epochs needed = "+str(len(a)))
  print("Accuracy = "+str(b))
  #Predicting 
  path1="5.png"
  img = plt.imread(path1)
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  img = image.load_img(path1,color_mode="grayscale",target_size=(28, 28))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  predict_arr=model.predict(x)
  index=np.flatnonzero(predict_arr)
  print("The number is :"+str(np.squeeze(index)))

  
train_mnist()












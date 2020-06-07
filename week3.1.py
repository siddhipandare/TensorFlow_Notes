""" Improving Computer Vision Accuracy using Convolutions """
""" Handwritten digit clasifier with CallBacks"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
img=test_images[np.random.randint(0,50)]
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0 

class MyCallBack(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if(logs.get("accuracy")>0.99):
      print("\n Reached 99% accuracy so cancelling training!")
      self.model.stop_training=True

callback1=MyCallBack()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10,callbacks=[callback1])
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Accuracy is: "+str(test_acc))



from keras.preprocessing import image
path= r"5.png"
img = image.load_img(path,color_mode="grayscale", target_size=(28, 28))
plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

index=np.argmax(model.predict(x), axis=-1)
print("The number is :"+str(np.squeeze(index)))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
plt.show()
print(training_labels[0])
print(training_images[0])

#normalizing data 
training_images  = training_images / 255.0
test_images = test_images / 255.0

print(training_labels[0])
print(f"{np.around(training_images[0],1)}")



model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation="relu"), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist=model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

print(hist.epoch[-1]+1,hist.history["accuracy"][-1])
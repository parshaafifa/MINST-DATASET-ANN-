import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
tf.random.set_seed(72)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
type(X_train)
print(Y_train.shape)
print(X_train.shape)
print(Y_train.shape)
print(X_train[10].shape)
plt.imshow(X_train[10])
print(Y_train[0])
plt.imshow(X_train[0], cmap='pink')
plt.title(f"Label: {Y_train[0]}")
plt.axis("off")  # hides the axis
plt.show()
print(np.unique(Y_test))
X_train=X_train/255
X_test=X_test/255
X_train.shape
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax') 
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
history=model.fit(X_train,Y_train,epochs=10,validation_split=0.2)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.title('model accuracy')
plt.xlabel('accuracy')
plt.ylabel('epoch')
plt.legend()
plt.show()
Y_pred=model.predict(X_test)
Y_prob = Y_pred.argmax(axis=1)
Y_prob[0]
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_prob)
plt.imshow(X_train[0])
input_sample = X_train[0].reshape(1, 28, 28)  # reshape to batch size 1
pred = model.predict(input_sample)           # get prediction (probabilities)
pred_class = pred.argmax(axis=1)             # get predicted class index

print(pred_class)
 # If pred is 2D: shape (1, num_classes)


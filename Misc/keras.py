from Misc.get_data import *
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses,optimizers,metrics

dnn_keras_model = models.Sequential()

dnn_keras_model.add(layers.Dense(units=13,input_dim=13,activation='relu'))

dnn_keras_model.add(layers.Dense(units=13,activation='relu'))
dnn_keras_model.add(layers.Dense(units=13,activation='relu'))

dnn_keras_model.add(layers.Dense(units=3,activation='softmax'))

losses.sparse_categorical_crossentropy

dnn_keras_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

dnn_keras_model.fit(scaled_x_train,y_train,epochs=50)

predictions = dnn_keras_model.predict_classes(scaled_x_test)

print(classification_report(predictions,y_test))
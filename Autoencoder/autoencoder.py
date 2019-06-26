import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

df = pd.read_csv('course/05-Autoencoders/anonymized_data.csv')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop('Label',axis=1))
num_inputs = 30  # 3 dimensional input
num_hidden = 2  # 2 dimensional representation 
num_outputs = num_inputs # Must be true for an autoencoder!

learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)

loss = tf.reduce_mean(tf.square(outputs - X))  # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
train  = optimizer.minimize( loss)

init = tf.global_variables_initializer()

num_steps = 1000

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_steps):
        sess.run(train,feed_dict={X: scaled_data})

with tf.Session() as sess:
    sess.run(init)
        
    # Now ask for the hidden layer output (the 2 dimensional output)
    output_2d = hidden.eval(feed_dict={X: scaled_data})

plt.scatter(output_2d[:,0],output_2d[:,1],c=df['Label'])

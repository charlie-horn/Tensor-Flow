import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

rand_a = np.random.uniform(0,100,(5,5))

rand_b = np.random.uniform(0,100,(5,1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_op = a + b
mul_op = a * b
with tf.Session() as sess:
    add_result = sess.run(add_op,feed_dict={a:rand_a, b:rand_b})
    mult_result = sess.run(mul_op,feed_dict={a:rand_a, b:rand_b})

## Constants

n_features = 10
n_dense_neurons = 3

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

m = tf.Variable(np.random.rand(1))
b = tf.Variable(np.random.rand(1))

error = 0

for x,y in zip(x_data, y_data):
    y_hat = m*x + b

    error += (y-y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epochs = 0
    for i in range(epochs):
        sess.run(train)

    final_slope, final_intercept = sess.run([m,b])

x_test = np.linspace(-1,11,10)
y_pred = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred,'r')
plt.plot(x_data,y_data)
plt.show()

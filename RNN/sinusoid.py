import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

class TimeSeriesData():
    def __init__(self,num_points,xmin,xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin,xmax,num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self,batch_size,steps,return_batch_ts=False):
        
        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size,1) 
        
        # Convert to be on time series
        ts_start = rand_start * (self.xmax- self.xmin - (steps*self.resolution) )
        
         # Create batch Time Series on t axis
        batch_ts = ts_start + np.arange(0.0,steps+1) * self.resolution
        
        # Create Y data for time series in the batches
        y_batch = np.sin(batch_ts)
        
        # Format for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) ,batch_ts
        
        else:
            
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 

def single_training_instance_plot():
    ts_data = TimeSeriesData(250,0,10)

    num_time_steps = 30
    y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)

    batch = plt.figure(1).gca()
    batch.plot(ts_data.x_data,ts_data.y_true,label='Sin(t)')
    batch.plot(ts.flatten()[1:],y2.flatten(),'*',label='Single Training Instance')
    batch.legend()
    #batch.tight_layout()

    train = plt.figure(2).gca()
    train_inst = np.linspace(5,5 + ts_data.resolution * (num_time_steps + 1), num_time_steps+1)

    train.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), "bo", markersize=15,alpha=0.5 ,label="instance")
    train.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), "*", markersize=7, label="target")

def plot_batch(X,Y_true,Y_pred,mse):

    plt.plot(X[1:],Y_true, "bo", markersize=15,alpha=0.5 ,label="Desired Output")
    plt.plot(X[1:],Y_pred, "*", markersize=7, label="Output")
    plt.legend()
    title = "MSE: " + str(mse)
    plt.title(title)
    plt.show()
    plt.close()

# single_training_instance_plot()

tf.reset_default_graph()

ts_data = TimeSeriesData(250,0,10)
num_inputs = 1
num_time_steps = 30
num_neurons = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = 2000
batch_size = 1

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        X_batch, y_batch, x_vals = ts_data.next_batch(batch_size, num_time_steps, return_batch_ts=True)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            plot_batch(x_vals.flatten(),y_batch.flatten(),y_pred[0,:,0],mse)

    saver.save(sess, "./rnn_time_series_model")

with tf.Session() as sess:                          
    saver.restore(sess, "./rnn_time_series_model")   
    X_batch, y_batch, x_vals = ts_data.next_batch(batch_size, num_time_steps, return_batch_ts=True)
    train_inst = np.linspace(5,5 + ts_data.resolution * (num_time_steps + 1), num_time_steps+1)
    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})

plt.title("Testing Model")

plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5, label="Training Instance")

plt.plot(train_inst[1:], np.sin(train_inst[1:]), "*", markersize=10, label="target")

plt.plot(train_inst[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")

plt.xlabel("Time")
plt.legend()
plt.tight_layout()


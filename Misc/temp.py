import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#%%
wine_data = load_wine()
feat_data = wine_data['data']
labels = wine_data['target']


#%%
X_train, X_test, y_train, y_test = train_test_split(feat_data,
                                                    labels,
                                                    test_size=0.3,
                                                   random_state=101)


#%%
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)
# ONE HOT ENCODED
onehot_y_train = pd.get_dummies(y_train).as_matrix()
one_hot_y_test = pd.get_dummies(y_test).as_matrix()

#%% [markdown]
# ### Parameters

#%%
num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
learning_rate = 0.01


#%%
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

#%% [markdown]
# ### Placeholder

#%%
X = tf.placeholder(tf.float32,shape=[None,num_feat])
y_true = tf.placeholder(tf.float32,shape=[None,3])

#%% [markdown]
# ### Activation Function

#%%
actf = tf.nn.relu

#%% [markdown]
# ### Create Layers

#%%
hidden1 = fully_connected(X,num_hidden1,activation_fn=actf)


#%%
hidden2 = fully_connected(hidden1,num_hidden2,activation_fn=actf)


#%%
output = fully_connected(hidden2,num_outputs)

#%% [markdown]
# ### Loss Function

#%%
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)

#%% [markdown]
# ### Optimizer

#%%
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#%% [markdown]
# ### Init

#%%
init = tf.global_variables_initializer()


#%%
training_steps = 1000
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(training_steps):
        sess.run(train,feed_dict={X:scaled_x_train,y_true:y_train})
        
    # Get Predictions
    logits = output.eval(feed_dict={X:scaled_x_test})
    
    preds = tf.argmax(logits,axis=1)
    
    results = preds.eval()


#%%
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(results,y_test))
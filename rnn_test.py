# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

def plot_data(data):
    y = data
    x = range(df.shape[0])  
    plt.plot(x, y)   
    plt.show()
    
    
def plot_data_cmp(x, y):
    plt.plot(y)
    plt.plot(x)
    plt.show()


def read_data(file_path="./", file_name="input.csv"):
    if file_path is None:
        return None
    if file_name is None:
        return None    
    return pd.read_csv(os.path.join(file_path, file_name)).values

    
def split_data(data, n_timesteps, n_input, n_output):
    data_len = data.shape[0]
    n_batches = int((data_len-(n_timesteps-1)-1))
    
    expected_data_size = n_batches + n_timesteps - 1 + 1
    data = data[data_len-expected_data_size:]
    
    data_list = []
    for i in range(expected_data_size):
        item = data[i][-1]
        data_list.append([item])
    data = np.array(data_list).reshape(-1, n_input)

    input_data = data[:-1]
    output_data = data[1:]
    return input_data, output_data, n_batches


def get_batches(data, n_batches, n_timesteps, n):
    data = data.tolist()
    data_flat_list = []
    for i in range(n_batches):
        subarr = data[i:i+n_timesteps]
        data_flat_list.append(subarr)
    data_batches = np.array(data_flat_list)
    
    return data_batches.reshape(-1, n_timesteps, n)


def RNN(n_timesteps, n_input, n_output, n_units, n_layers=1):    
    tf.reset_default_graph()
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_timesteps, n_input], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_timesteps, n_output], name="Y")
    
    cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_units, name="cell%d"%layer) for layer in range(n_layers)]
    
    stacked_rnn = tf.contrib.rnn.MultiRNNCell(cells)
    stacked_output, states = tf.nn.dynamic_rnn(stacked_rnn, X, dtype=tf.float32)
    stacked_output = tf.layers.dense(stacked_output, n_output)       
    return X, Y, stacked_output


def RNN_train(work_dir, market, 
              X, Y, outputs, 
              n_timesteps, n_input, n_output, n_units,
              learning_rate, n_epochs, x_train_batches, y_train_batches, useLSTM=False):   
    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=outputs, predictions=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver=tf.train.Saver(max_to_keep=1)
    save_path= os.path.join(work_dir, market)    
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            x_train_batches = np.array(x_train_batches)
            y_train_batches = np.array(y_train_batches)
            sess.run(training_op, feed_dict={X: x_train_batches, Y: y_train_batches})
                            
            if epoch % 10000 == 0:
                loss_value = loss.eval(feed_dict={X: x_train_batches, Y: y_train_batches})
                print("loss:", loss_value)
                saver.save(sess, save_path)
                
                # print variables
                variables_names =[v.name for v in tf.trainable_variables()]
                values = sess.run(variables_names)
                for k,v in zip(variables_names, values):
                    print(k)
                    print(v)
                y_pred_data = sess.run(outputs, feed_dict={X: x_train_batches})
                x_list = y_train_batches.reshape(-1, 1)
                y_list = y_pred_data.reshape(-1, 1)
                
                plot_data_cmp(x_list, y_list)

work_dir = './'
input_file = "input.csv"

n_timesteps = 1
n_input = 1
n_output = 1
n_units = 5
n_epochs = 1000000
n_layers = 1
learning_rate = 0.01

data = read_data(work_dir, input_file)
plot_data(data)
input_data, output_data, n_batches = split_data(data, n_timesteps, n_input, n_output)

# Uncomment this block for testing with data from y = 10x+5
"""
input_data = np.array([[i*10+5] for i in range(300)])
output_data = input_data.copy()
n_batches = 300
"""

x_train_batches = get_batches(input_data, n_batches, n_timesteps, n_input)
y_train_batches = get_batches(output_data.transpose()[-1], n_batches, n_timesteps, n_output)
X, Y, outputs = RNN(n_timesteps, n_input, n_output, n_units, n_layers)
RNN_train(work_dir, market, X, Y, outputs, n_timesteps, n_input, n_output, n_units, learning_rate, n_epochs,
          x_train_batches, y_train_batches, useLSTM=False)

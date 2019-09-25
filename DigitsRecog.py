# NOTE THIS IS A SMALLER TEST BUILD OF THE LARGER PROJECT.


# Import MNIST data and tensorflow

import input_data
import numpy
import tensorflow as tflow

mnist = inpute_data.read_data_sets("/tmp/data", one_hot=True)

# set inital parameters // hyper-parameters

rate_learning = 0.01
training_iter = 30
size_batch = 100
display_step = 2

# Input for graph in tensorflow

xval = tflow.placeholder("float", [None, 784]) # mnist data image of shape is 28^2 = 784
yval = tflow.placeholder("float", [None, 10]) # 10 cases as digits can range 0-9

# model creation

# inital model weights

W = tflow.Variable(tflow.zeros([784,10]))
bias = tflow.Variable(tflow.zeros[10]))

with tflow.name_scope("Wxval_bias") as scope:

    # create linear model
    model = tflow.nn.softmax(tflow.matmul(xval,W) + bias) # softmax

# summary opperations to collect data

w_h = tflow.histogram_summary("Weights", W)
bias_h = tflow.histogram_summary("Biases", bias)

# init variables

init = tflow.initialize_all_variables()

# launch graph

with tflow.Session() as session:
    session.run(init)

    # training cycle

    for iteration in range(training_iter):
        average_cost = 0
        batch_total = int(mnist.train.next_batch(size_batch))

        # loooping over all batches

        for val in range (batch_total):

            batch_xs, batch_ys = mnist.train.next_batch(size_batch)

            session.run(optimizer, feed_dict = {xval: batch_xs, yval: batch_ys})/batch_total

            









    


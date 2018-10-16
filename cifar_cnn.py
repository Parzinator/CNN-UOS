import tensorflow as tf
import numpy as np

from cifar_input import get_training_data, get_test_data

epochs = 100
batch_size = 128
keep_probability = 0.7
learning_rate = 0.001

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape = (None, 32, 32, 3), name = "input_x")
Y = tf.placeholder(tf.float32, shape = (None, 10), name = "output_y")

keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape = (3, 3, 3, 64), mean = 0, stddev = 0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape = (3, 3, 64, 128), mean = 0, stddev = 0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape = (3, 3, 128, 64), mean = 0, stddev = 0.08))

    # conv layer 1
    conv1       = tf.nn.conv2d(x, conv1_filter, strides = (1, 1, 1, 1), padding = "SAME")
    conv1       = tf.nn.relu(conv1)
    conv1_pool  = tf.nn.max_pool(conv1, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "SAME")

    # conv layer 2
    conv2       = tf.nn.conv2d(conv1_pool, conv2_filter, strides = (1, 1, 1, 1), padding = "SAME")
    conv2       = tf.nn.relu(conv2)
    conv2_pool  = tf.nn.max_pool(conv2, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "SAME")

    # conv layer 3
    conv3       = tf.nn.conv2d(conv2_pool, conv3_filter, strides = (1, 1, 1, 1), padding = "SAME")
    conv3       = tf.nn.relu(conv3)
    conv3_pool  = tf.nn.max_pool(conv3, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "SAME")

    # flatten
    # may be tf.contrib.layers for older versions
    flat = tf.layers.flatten(conv3_pool)

    # fully connected 1
    full1 = tf.contrib.layers.fully_connected(inputs = flat, num_outputs = 128, activation_fn = tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)

    # output layer
    out = tf.contrib.layers.fully_connected(inputs = full1, num_outputs = 10, activation_fn = None)

    return out

def train_on_batch(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, feed_dict = {
        X: feature_batch,
        Y: label_batch,
        keep_prob: keep_probability
    })

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = session.run(cost, feed_dict = {
        X: feature_batch,
        Y: label_batch,
        keep_prob: 1
    })

    valid_acc = session.run(accuracy, feed_dict = {
        X: feature_batch,
        Y: label_batch,
        keep_prob: 0.1
    })

    print("Loss: {:>10.4f} Validation Accuracy: {:.6f}".format(loss, valid_acc))

logits = conv_net(X, keep_prob)

# loss and optimizer
cost        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer   = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# accuracy
correct_pred    = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy        = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")

# load training data
features, labels = get_training_data()

with tf.Session() as sess:
    # initialize variables
    sess.run(tf.global_variables_initializer())

    # training
    steps_per_epoch = int(features.shape[0] / batch_size) - 1

    for epoch in range(epochs):
        for i in range(steps_per_epoch):
            features_batch  = features[i * batch_size : (i + 1) * batch_size]
            labels_batch    = labels[i * batch_size : (i + 1) * batch_size]

            train_on_batch(sess, optimizer, keep_probability, features_batch, labels_batch)

        print("Epoch {:>2}:  ".format(epoch + 1), end = "")
        print_stats(sess, features_batch, labels_batch, cost, accuracy)



    features_test, labels_test = get_test_data()

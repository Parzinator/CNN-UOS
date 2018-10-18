import tensorflow as tf
import numpy as np

import layers
from architectures import *
from cifar_input import get_training_data, get_test_data


epochs              = 100
batch_size          = 100
validation_size     = 1000
keep_probability    = 0.5
learning_rate       = 0.001

tf.reset_default_graph()

X           = tf.placeholder(tf.float32, shape = (None, 32, 32, 3), name = "input_x")
Y           = tf.placeholder(tf.float32, shape = (None, 10), name = "output_y")
keep_prob   = tf.placeholder(tf.float32, name = "keep_prob")

def train_on_batch(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, feed_dict = {
        X: feature_batch,
        Y: label_batch,
        keep_prob: keep_probability
    })

def get_batch_stats(session, features, labels):
    return session.run([cost, accuracy], feed_dict = {
        X: features,
        Y: labels,
        keep_prob: 1
    })

def get_stats(session, features, labels):
    steps = int(features.shape[0] / batch_size) - 1

    loss, acc = 0, 0

    for i in range(steps):
        features_batch  = features[i * batch_size : (i + 1) * batch_size]
        labels_batch    = labels[i * batch_size : (i + 1) * batch_size]

        step_loss, step_acc = get_batch_stats(session, features_batch, labels_batch)

        loss += step_loss / steps
        acc += step_acc / steps

    return loss, acc

def print_stats(epoch, loss, acc, mode = "train"):
    print("Epoch {} ({}):  Loss: {:.8f} Accuracy: {:.5f}".format(epoch + 1, mode, loss, acc))


logits = conv_net_2(X, keep_prob)

# loss and optimizer
cost        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
#optimizer   = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
optimizer   = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
#optimizer   = tf.train.MomentumOptimizer(learning_rate = 0.01, momentum = 0.9, use_nesterov = True).minimize(cost)

# accuracy
correct_pred    = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy        = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")

# load training data
features, labels    = get_training_data()

features_training   = features[validation_size : ]
features_validation = features[ : validation_size]

labels_training     = labels[validation_size : ]
labels_validation   = labels[ : validation_size]

with tf.Session() as sess:
    # create summary writer for TensorBoard
    writer = tf.summary.FileWriter("logs", sess.graph)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    print("Trainable Variables:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    # training
    steps_per_epoch = int(features_training.shape[0] / batch_size) - 1

    for epoch in range(epochs):
        # shuffle indices
        idx = np.random.permutation(features_training.shape[0])

        for i in range(steps_per_epoch):
            slice           = idx[i * batch_size : (i + 1) * batch_size]

            features_batch  = features_training[slice]
            labels_batch    = labels_training[slice]

            train_on_batch(sess, optimizer, keep_probability, features_batch, labels_batch)

        step_loss, step_acc = get_batch_stats(sess, features_batch, labels_batch)
        print_stats(epoch, step_loss, step_acc, mode = "train")

        # validation

        validation_loss, validation_acc = get_stats(sess, features_validation, labels_validation)

        print_stats(epoch, validation_loss, validation_acc, mode = "valid")

    print("--- TRAINING COMPLETE ---")
    print("Test stats:")

    features_test, labels_test = get_test_data()

    test_loss, test_acc = get_stats(sess, features_test, labels_test)

    print_stats(0, test_loss, test_acc, mode = "test")

    writer.close()

import tensorflow as tf

def init_xavier(n_inputs, n_outputs):
    init_range = tf.sqrt(12 / (n_inputs + n_outputs))

    return tf.random_uniform_initializer(-init_range, init_range)

def layer_fc(input_tensor, num_outputs, activation = tf.nn.relu):

    layer = tf.contrib.layers.fully_connected(inputs = input_tensor, num_outputs = num_outputs, activation_fn = activation)

    return layer

def layer_conv(input_tensor, input_channels, output_channels, kernel_size,
    strides = (1, 1, 1, 1), padding = "SAME", activation = tf.nn.leaky_relu, name = None, batch_norm = False):

    # variables = tf.Variable(tf.truncated_normal(
    #     shape = (kernel_size, kernel_size, input_channels, output_channels),
    #     mean = 0,
    #     stddev = 0.02
    # ))

    initializer = init_xavier(input_channels * kernel_size ** 2, output_channels * kernel_size ** 2)

    variables = tf.Variable(initializer((kernel_size, kernel_size, input_channels, output_channels)))

    layer = input_tensor

    if batch_norm:
        layer = tf.layers.batch_normalization(layer)

    layer = tf.nn.conv2d(layer, variables, strides = strides, padding = padding)
    layer = activation(layer)

    return layer

def layer_max_pool(input_tensor, shape = (1, 2, 2, 1), padding = "SAME"):

    layer = tf.nn.max_pool(input_tensor, ksize = shape, strides = shape, padding = padding)

    return layer

def layer_avg_pool(input_tensor, shape = (1, 2, 2, 1), padding = "SAME"):

    layer = tf.nn.avg_pool(input_tensor, ksize = shape, strides = shape, padding = padding)

    return layer

def layer_global_avg_pool(input_tensor):

    layer = tf.reduce_mean(input_tensor, axis = (1, 2))

    return layer

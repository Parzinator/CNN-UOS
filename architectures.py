import tensorflow as tf

import layers

def conv_net(x, keep_prob):
    model = layers.layer_conv(x, input_channels = 3, output_channels = 64, kernel_size = 3)
    model = layers.layer_conv(model, input_channels = 64, output_channels = 64, kernel_size = 3, batch_norm = True)

    model = layers.layer_avg_pool(model, shape = (1, 2, 2, 1))

    model = layers.layer_conv(model, input_channels = 64, output_channels = 128, kernel_size = 3, batch_norm = True)
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)

    model = layers.layer_avg_pool(model, shape = (1, 2, 2, 1))

    model = layers.layer_conv(model, input_channels = 128, output_channels = 256, kernel_size = 3, batch_norm = True)

    model = tf.layers.flatten(model)

    model = layers.layer_fc(model, num_outputs = 10)

    model = tf.nn.dropout(model, keep_prob)

    return model

def conv_net_2(x, keep_prob):
    model = layers.layer_conv(x, input_channels = 3, output_channels = 16, kernel_size = 3)
    model = layers.layer_conv(model, input_channels = 16, output_channels = 32, kernel_size = 3)

    model = layers.layer_max_pool(model, shape = (1, 2, 2, 1))

    model = layers.layer_conv(model, input_channels = 32, output_channels = 32, kernel_size = 3)
    model = layers.layer_conv(model, input_channels = 32, output_channels = 64, kernel_size = 3)

    model = layers.layer_max_pool(model, shape = (1, 2, 2, 1))

    model = tf.layers.flatten(model)
    model = layers.layer_fc(model, num_outputs = 64)

    model = tf.nn.dropout(model, keep_prob)

    model = layers.layer_fc(model, num_outputs = 10)

    return model

def simple_net(x, keep_prob):
    # 32 x 32
    model = layers.layer_conv(x, input_channels = 3, output_channels = 64, kernel_size = 3, batch_norm = True)
    model = layers.layer_conv(model, input_channels = 64, output_channels = 128, kernel_size = 3, batch_norm = True)
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)
    #model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)

    model = layers.layer_max_pool(model, shape = (1, 2, 2, 1))

    # 16 x 16
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)
    #model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)

    model = layers.layer_max_pool(model, shape = (1, 2, 2, 1))

    # 8 x 8
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)

    model = layers.layer_max_pool(model, shape = (1, 2, 2, 1))

    # 4 x 4
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 1, batch_norm = True)
    #model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 1, batch_norm = True)

    model = layers.layer_max_pool(model, shape = (1, 2, 2, 1))

    # 2 x 2
    model = layers.layer_conv(model, input_channels = 128, output_channels = 128, kernel_size = 3, batch_norm = True)

    model = layers.layer_max_pool(model, shape = (1, 2, 2, 1))

    # 1 x 1
    model = tf.layers.flatten(model)
    model = layers.layer_fc(model, num_outputs = 10)

    return model

import sys
import os

import tensorflow as tf

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10000000     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01         # Initial learning rate.

FLAGS=tf.app.flags.FLAGS
'''
def build_nn(data):
    
    layer_1 = tf.layers.Dense(FLAGS.hidden_neurals, 
                    #kernel_initializer=tf.truncated_normal_initializer(mean=0.1),
                    kernel_initializer=tf.constant_initializer(0.1),
                    activation='relu', name='layer_1')
    out_1 = layer_1.apply(data)
    layer_2 = tf.layers.Dense(5, kernel_initializer=tf.truncated_normal_initializer(mean=0.1),
                    activation='relu', name='layer_2')
    #layer_2 = tf.layers.Conv1D(filters=1, kernel_size=3, padding='same',
    #)
    out_2 = layer_2.apply(out_1)
    layer_3 = tf.layers.Dense(5, kernel_initializer=tf.truncated_normal_initializer(mean=0.1),
                    activation='relu', name='layer_3')
    out_3 = layer_3.apply(out_2)
    
    layer_out = tf.layers.Dense(1, 
                    kernel_initializer=tf.truncated_normal_initializer(),
                    #kernel_initializer=tf.constant_initializer([3,5]),
                    #bias_initializer=tf.constant_initializer(10),
                    name='layer_out')
    out = layer_out.apply(out_3)
    return out
'''
def build_nn(data):
    for i in range(FLAGS.hidden_layers):
        layer = tf.layers.Dense(FLAGS.hidden_neurals, kernel_initializer=tf.truncated_normal_initializer(mean=0.1),
                    activation='relu', name='layer_'+str(i))
        out = layer.apply(data)
        data = out
    '''
    data = tf.expand_dims(data, -1, name='expand_dim')
    # data = [batch, hidden_neurals, 1]
    layer_conv = tf.layers.Conv1D(filters=5, kernel_size=3, padding='same')
    out = layer_conv.apply(data)
    data = out
    # data = [batch, hidden_neurals, conv_filter_out]
    layer_conv = tf.layers.Conv1D(filters=1, kernel_size=3, padding='same')
    out = layer_conv.apply(data)
    data = out
    # data = [batch, hidden_neurals, 1]
    data = tf.squeeze(data, 2)
    '''
    layer_out = tf.layers.Dense(1, 
                    kernel_initializer=tf.truncated_normal_initializer(),
                    #kernel_initializer=tf.constant_initializer([3,5]),
                    #bias_initializer=tf.constant_initializer(10),
                    name='layer_out')
    out = layer_out.apply(data)
    return out

def build_regression(data, data_size=2):
    # X = [x0, x1]T
    # Y = W0 + X * W1 + x0 * x1 * W2
    x = []
    for i in range(data_size):
        x.append(tf.reshape(data[:,i], [-1,1]))

    #W0 = tf.Variable(initial_value = tf.random.truncated_normal([1,1]), name='W0')
    #W1 = tf.Variable(initial_value = tf.random.truncated_normal([data_size,1]), name='W1')
    #W2 = tf.Variable(initial_value = tf.random.truncated_normal([1,1]), name='W2')
    W0 = tf.Variable(initial_value = [[1.0]], name='W0')
    W1 = tf.Variable(initial_value = [[0.0],[0.0]], name='W1')
    W2 = tf.Variable(initial_value = [[0.0]], name='W2')
    #y1 = tf.reduce_sum(tf.multiply(data, W1))
    y1 = tf.matmul(data, W1, name='y1')
    y2 = W2
    for i in range(data_size):
        y2 = tf.math.multiply(y2 , x[i], name = 'y2_'+str(i))
    Y = W0 + y1 + y2
    return Y


def loss(labels, predicts):
    #maes = tf.losses.absolute_difference(labels, predicts)
    #maes_loss = tf.reduce_sum(maes)
    mses = tf.losses.mean_squared_error(labels, predicts)
    mape_loss = tf.keras.losses.MeanAbsolutePercentageError()
    mape = mape_loss(labels, predicts)

    # Percentage error
    #error = tf.math.subtract(predicts, labels, name = 'predict_error')
    #abs_error = tf.math.abs(error, name = 'abs_error')
    #percentage_error = tf.math.divide_no_nan(abs_error, labels, name='percentage_error')
    #loss = tf.reduce_sum(percentage_error)
    return mape
'''
def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
'''

def train_op(total_loss, global_step, train_dataset_size):
    """Train model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = train_dataset_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    '''
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    '''
    lr = FLAGS.lr

    # Generate moving averages of all losses and associated summaries.
    # loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    #with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.MomentumOptimizer(learning_rate=lr,
    #                                momentum=0.9)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    #opt = hvd.DistributedOptimizer(opt)
    grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # TODO Try to remove moving average for OPs
    '''
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op
    '''
    return apply_gradient_op, grads
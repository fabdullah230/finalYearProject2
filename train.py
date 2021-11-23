import tensorflow as tf
import data_loading
import nn
import os
import numpy as np
import shutil
import time
import timeit


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('raw_data', 
                        './data/conv2d_rand_data_analyze.csv',
                        """ The data used to train """)
tf.app.flags.DEFINE_string('main_comp_exp', 
                        '',
                        """ File of Exp of main components """)
tf.app.flags.DEFINE_integer('load', 
                        0,
                        """ Whether load checkpoint and continue train """)
tf.app.flags.DEFINE_string('checkpoint_dir', 
                        './train_checkpoints/',
                        """ The checkpoint used to train """)
tf.app.flags.DEFINE_integer('batch_size', 
                        64,
                        """ Training epochs """)                        
tf.app.flags.DEFINE_integer('train_epochs', 
                        1000,
                        """ Maximum training epochs """)
tf.app.flags.DEFINE_integer('max_epoch_no_update', 
                        20,
                        """ If recent x epoch does not meet new min loss, stop. """)
tf.app.flags.DEFINE_float('lr', 
                        0.001,
                        """ Learning rate """)
tf.app.flags.DEFINE_integer('hidden_neurals', 
                        3,
                        """ Num of hidden neurals """)
tf.app.flags.DEFINE_integer('hidden_layers', 
                        1,
                        """ Num of hidden layers """)
tf.app.flags.DEFINE_integer('std_test', 
                        0,
                        """ If 1, use standard function to generate data for testing """)
tf.app.flags.DEFINE_integer('data_col_start', 
                        0,
                        """ strat column of input data """)
tf.app.flags.DEFINE_integer('data_col_end', 
                        2,
                        """ end column of input data """)
tf.app.flags.DEFINE_integer('label_col', 
                        3,
                        """ column of output """)
tf.app.flags.DEFINE_string('loss_log', 
                        './loss_log.txt',
                        """ The file to save loss log """)
                        
tf.app.flags.DEFINE_integer('default_pc_exp', 
                        0,
                        """ if 0, then pc is 0exp(always1), else, set the pc exp """)
def train(train_data, train_label, test_data, test_label):
    '''
    About the training timeuse statistic: In each epoch, record all step times in the epoch.
    Let the statistic script to process the accumulated time use.
    '''
    config=tf.ConfigProto()
    data_len = FLAGS.data_col_end - FLAGS.data_col_start + 2 # Include a main component
    data_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,data_len])
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,1])
    global_step = tf.train.get_or_create_global_step()
    logits = nn.build_nn(data_placeholder)
    #logits = nn.build_regression(data_placeholder)
    loss = nn.loss(label_placeholder, logits)
    train_op, grads_op = nn.train_op(loss, global_step, len(train_data))
    grad_tensors = []
    for grad_tensor, grad_var in grads_op:
        grad_tensors.append(grad_tensor)
    
    
    steps = 0
    num_batches_per_epoch = len(train_data) // FLAGS.batch_size
    total_train_step = FLAGS.train_epochs * num_batches_per_epoch
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3)

    #with tf.train.MonitoredTrainingSession(        
    with tf.Session(
        # Train
        config=config) as sess:
        
        options = tf.RunOptions(output_partition_graphs=True)
        run_metadata = tf.RunMetadata()

        train_vars = tf.trainable_variables()
        
        if FLAGS.load == 0:
            # Init variables
            print("Start initialization...")
            step_data, step_label = data_loading.fetch_data(train_data,train_label)
            run_target = init_op
            run_result = sess.run(run_target,
                        feed_dict = {data_placeholder: step_data, 
                        label_placeholder: step_label},)
        else:
            print("Restoring from checkpoint in folder %s ..." % FLAGS.checkpoint_dir)
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        #vars_vals = run_result[1:]
        #for var, val in zip(train_vars, vars_vals):
        #    print("var: {}, value: {}".format(var.name, val))
        print("Start training...")
        last_min_loss = -1
        last_min_loss_step = 0
        train_loss_list = []
        epoch_avg_loss = 0.0
        epoch_cnt = 0
        epoch_timeuse = 0.0
        while steps < total_train_step:
            #run_target = [train_op, loss, data_placeholder, label_placeholder, 'layer_out/MatMul:0','layer_out/BiasAdd:0',logits] + train_vars
            step_data, step_label = data_loading.fetch_data(train_data,train_label)
            run_target = [train_op, loss, logits, data_placeholder] + train_vars
            time_start = time.time()
            run_result = sess.run(run_target,
                feed_dict = {data_placeholder: step_data, 
                            label_placeholder: step_label},
                            options = options, run_metadata = run_metadata)
            time_end = time.time()

            step_timeuse = time_end - time_start
            epoch_timeuse += step_timeuse
            output = run_result[2]
            loss_ret = run_result[1]
            data = run_result[-1]
            
            #if (steps % 10 == 0):
                #print('Step = %d: Loss = %f' % (steps, loss_ret))
                #print('output = ' + str(output))
                #print('label = ' + str(step_label))
            
            #if steps - last_min_loss_step > num_batches_per_epoch * 10:
            #    break
            
            steps += 1
            epoch_avg_loss += loss_ret

            if steps % num_batches_per_epoch == 0:
                # An epoch is finished
                epoch_avg_loss /= num_batches_per_epoch
                print('Epoch %d, loss = %f' % (epoch_cnt, epoch_avg_loss))
                # Update the last min loss epoch
                if last_min_loss == -1 or epoch_avg_loss < last_min_loss:
                    last_min_loss = epoch_avg_loss
                    last_min_loss_epoch = epoch_cnt
                train_loss_list.append(epoch_avg_loss)
                saver.save(sess, FLAGS.checkpoint_dir, global_step=steps)
                with open(FLAGS.loss_log, 'a') as fd_out:
                    fd_out.write('%f,%f\n' % (epoch_avg_loss, epoch_timeuse))
                epoch_avg_loss = 0.0
                epoch_timeuse = 0.0
                epoch_cnt += 1
                if epoch_cnt - last_min_loss_epoch > FLAGS.max_epoch_no_update:
                    break
            '''
            if steps == 1:
                for i, partition_graph_def in enumerate(run_metadata.partition_graphs):
                    meta_graph_path = 'meta_graph'
                    if not os.path.exists(meta_graph_path):
                        os.makedirs(meta_graph_path)
                    with open('%s/%d.pbtxt' % (meta_graph_path,i), 'w') as f:
                        print(partition_graph_def, file=f)
            '''
            
    
        # Validate in training set
        train_predict_list = []
        train_err_percent_list = []
        print('Start validation in training set')
        train_avg_err_percent = 0
        for i in range(len(train_data)):
            data = [train_data[i]]
            label = [train_label[i]]
            predict = sess.run(logits, 
                feed_dict = {data_placeholder: data, 
                            label_placeholder: label})
            predict = float(predict)
            err = abs((predict - label[0][0]))
            err_percent = err / label[0][0] * 100.0
            #result_format = 'predict = %.6f, real = %.6f, err = %.6f, err%% = %.6f'
            #print(result_format % (predict, label[0][0], err, err_percent))
            train_avg_err_percent += err_percent
            train_predict_list.append(predict)
            train_err_percent_list.append(err_percent)
        train_avg_err_percent /= len(train_data)
        
        # Test
        test_predict_list = []
        test_err_percent_list = []
        print('Start validation in testing set')
        test_avg_err_percent = 0
        for i in range(len(test_data)):
            data = [test_data[i]]
            label = [test_label[i]]
            predict = sess.run(logits, 
                feed_dict = {data_placeholder: data, 
                            label_placeholder: label})
            predict = float(predict)
            err = abs((predict - label[0][0]))
            err_percent = err / label[0][0] * 100.0
            result_format = 'predict = %.6f, real = %.6f, err = %.6f, err%% = %.6f'
            #print(result_format % (predict, label[0][0], err, err_percent))
            test_avg_err_percent += err_percent
            test_predict_list.append(predict)
            test_err_percent_list.append(err_percent)
            
        test_avg_err_percent /= len(test_data)
        print('Train set average err(%%) = %.6f%%' % train_avg_err_percent)
        print('Test set average err(%%) = %.6f%%' % test_avg_err_percent)
    return train_predict_list, train_err_percent_list, test_predict_list, test_err_percent_list

def linear_regression(train_data, train_label,test_data,test_label):
    learning_rate = 0.00000001
    training_epochs = 1

    X1 = tf.placeholder("float")
    X2 = tf.placeholder("float")
    #X3 = tf.placeholder("float")
    Y = tf.placeholder("float")
    W1 = tf.Variable(0.0,name='W1')
    #W2 = tf.Variable(np.random.randn(),name='W2')
    #W3 = tf.Variable(np.random.randn(),name='W3')
    b = tf.Variable(0.0,name='b')
    n = len(train_label)
    print("Data len: ", n)

    #hypothesis
    x0 = tf.multiply(X1,X2)
    x1 = tf.multiply(x0,W1)
    #x2 = tf.multiply(X2,W2)
    #x3 = tf.multiply(tf.multiply(X1,X2),W3)
    #y_pred = tf.add(x1,b)
    y_pred = tf.add(x1,b)
    #cost = tf.reduce_sum(tf.pow(y_pred-Y,2)) / (2*n)
    mape_loss = tf.keras.losses.MeanAbsolutePercentageError()
    cost = mape_loss(Y, y_pred)
    #cost = tf.losses.mean_squared_error(Y, y_pred)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    predict_list = []
    err_percent_list = []
    func = ''

    total_train_step = int(training_epochs * len(train_data) / FLAGS.batch_size)
    #config=tf.ConfigProto()
    #global_step = tf.train.get_or_create_global_step()
    #hooks=[]
    with tf.Session() as sess:
        sess.run(init)       
        for step in range(total_train_step):
            step_data, step_label = data_loading.fetch_data(train_data,train_label)
            for (x,y) in zip(step_data,step_label):
                sess.run(optimizer, 
                        feed_dict = {X1: x[0],
                            X2: x[1],
                            #X3: x[2],
                            Y: y})
            print("Step: %d: W1: %f, b: %f" % (step+1, sess.run(W1),
                        sess.run(b))) 
        
        w1_pred = sess.run(W1)
        b_pred  = sess.run(b)
        print("Final result: Y = %f*x1x2 + %f" % (w1_pred, b_pred))
        func = 'Y = %.8f*X1*X2 + %.8f'% (w1_pred,b_pred)
      
        # Validate in train dataset
        print('Validate in train dataset')
        avg_err_percent=0
        for i in range(len(train_data)):
            data = train_data[i]
            label = train_label[i][0]
            predict = sess.run(y_pred, 
                feed_dict = {X1: data[0], 
                            X2: data[1],
                            #X3: data[2],
                            })
            #print(predict)
            err = abs((predict - label))
            err_percent = err / label * 100.0
            result_format = 'predict = %.6f, real = %.6f, err = %.6f, err%% = %.6f'
            #print(result_format % (predict, label, err, err_percent))
            avg_err_percent += err_percent
            predict_list.append(predict)
            err_percent_list.append(err_percent)
        avg_err_percent /= len(train_data)
        print('Train Avg err%% = %.6f' % avg_err_percent)

        # Test
        print("Start testing...")
        avg_err_percent=0
        for i in range(len(test_data)):
            data = test_data[i]
            label = test_label[i][0]
            predict = sess.run(y_pred, 
                feed_dict = {X1: data[0], 
                            X2: data[1],
                            #X3: data[2],
                            })
            #print(predict)
            err = abs((predict - label))
            err_percent = err / label * 100.0
            result_format = 'predict = %.6f, real = %.6f, err = %.6f, err%% = %.6f'
            #print(result_format % (predict, label, err, err_percent))
            avg_err_percent += err_percent
            predict_list.append(predict)
            err_percent_list.append(err_percent)
        avg_err_percent /= len(test_data)
        print('Test Avg err%% = %.6f' % avg_err_percent)
    return predict_list, err_percent_list, func, [learning_rate, training_epochs]

def result_analyze(data_list):
    sorted_list = sorted(data_list)
    average = np.mean(data_list)
    std_err = np.std(data_list)
    min_data = sorted_list[0]
    max_data = sorted_list[-1]
    idx_top50 = len(data_list) * 50 // 100
    idx_top90 = len(data_list) * 90 // 100
    idx_top95 = len(data_list) * 95 // 100
    idx_top99 = len(data_list) * 99 // 100
    result_string = ''
    result_string += 'avg: %.4f\n' % average
    result_string += 'std_err: %.4f\n' % std_err
    result_string += 'min: %.4f\n' % min_data
    result_string += 'max: %.4f\n' % max_data 
    result_string += 'top50: %.4f\n' % sorted_list[idx_top50]
    result_string += 'top90: %.4f\n' % sorted_list[idx_top90]
    result_string += 'top95: %.4f\n' % sorted_list[idx_top95]
    result_string += 'top99: %.4f\n' % sorted_list[idx_top99]
    return result_string

def main():
    if FLAGS.std_test == 0:
        raw_data, raw_label = data_loading.load_raw_file(FLAGS.raw_data, 
                    input_col_list = list(range(FLAGS.data_col_start,FLAGS.data_col_end + 1)), 
                    output_col = FLAGS.label_col, exp_filename=FLAGS.main_comp_exp)
    else:
        raw_data, raw_label = data_loading.gen_standard_data()
    dataset_list = data_loading.split_dataset(raw_data, raw_label)
    train_data = dataset_list[0]
    train_label = dataset_list[1]
    test_data = dataset_list[2]
    test_label = dataset_list[3]
    train_predict_list, train_err_percent_list, test_predict_list, test_err_percent_list = train(train_data,
    #predict_list, err_percent_list,func,[rate,ep] = linear_regression(train_data,
                    train_label,
                    test_data,
                    test_label)
    result_train = result_analyze(train_err_percent_list)
    result_test = result_analyze(test_err_percent_list)
    return result_train, result_test

if __name__ == '__main__':
    with open('result.txt', 'a') as fdout:

        start = timeit.default_timer()
        result_train, result_test = main()
        stop = timeit.default_timer()
        time = start - stop

        fdout.write('Time taken: ' + str(time))
        fdout.write('Test set:\n')
        fdout.write((result_test))
        fdout.write('Train set:\n')
        fdout.write((result_train))
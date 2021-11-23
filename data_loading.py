import tensorflow as tf
import random
TEST_SET_MOD_NUM = 10 # Pick 1 sample into test from 10 sample

FLAGS = tf.app.flags.FLAGS

current_data_pos = 0
global_pc_log_outputed = False

def load_exp_file(filename):
    with open(filename, 'r') as fd_in:
        line = fd_in.readline()
    raw_exp_list = line.replace('\n','').split(' ')
    exp_list = []
    for item in raw_exp_list:
        exp_list.append(int(item))
    return exp_list

def gen_main_component(raw_record, exp_list=None):
    global global_pc_log_outputed
    main_component = 1.0
    if exp_list == None:
        exp_list = []
        exp = FLAGS.default_pc_exp
        for i in range(len(raw_record)):
            exp_list.append(exp)
    if not global_pc_log_outputed:
        global_pc_log_outputed=True
        print('[DEBUG] PC list is:', exp_list)
    for i in range(len(raw_record)):
    #for record in raw_record:
        record = raw_record[i]
        record = record ** exp_list[i]
        main_component *= record 
    #main_component /= 100 ** (len(raw_record) - 6)
    main_component /= 100000000
    return main_component

'''
Assume data are saved in row. Each row is a data
Does not shuffle the data
'''
def load_raw_file(filename, input_col_list, output_col, split_char=',', exp_filename=''):
    data = []
    label = []
    if exp_filename != '':
        exp_list = load_exp_file(exp_filename)
    else:
        exp_list = None
    with open(filename, 'r') as fd_in:
        lines = fd_in.readlines()
    #random.shuffle(lines)
    for line in lines:
        items = line.split(split_char)
        input_record = []
        
        for i in range(len(input_col_list)):
            input_record.append(float(items[i]))
        if exp_list != None:
            main_component = gen_main_component(input_record, exp_list)
        else:
            main_component = gen_main_component(input_record)
        input_record.append(main_component)
        output = float(items[output_col]) 
        data.append(input_record)
        label.append([output])
    return data, label


def shuffle_two_list(list_a, list_b):
    list_zip = list(zip(list_a, list_b))
    random.shuffle(list_zip)
    list_a1, list_b1 = zip(*list_zip)
    list_a1 = list(list_a1)
    list_b1 = list(list_b1)
    return list_a1, list_b1

'''
Split dataset, then shuffle train set and test set.
'''
def split_dataset(raw_data_list, raw_label_list):
    train_dataset = []
    train_label = []
    test_dataset = []
    test_label = []
    for i in range(len(raw_data_list)):
        if i % TEST_SET_MOD_NUM == 0:
            test_dataset.append(raw_data_list[i])
            test_label.append(raw_label_list[i])
        else:
            train_dataset.append(raw_data_list[i])
            train_label.append(raw_label_list[i])

    train_dataset, train_label = shuffle_two_list(train_dataset, train_label)
    test_dataset, test_label = shuffle_two_list(test_dataset, test_label)
 
    return [train_dataset, train_label, test_dataset, test_label]

def fetch_data(train_data, train_label):
    global current_data_pos
    next_data_pos = current_data_pos + FLAGS.batch_size
    if next_data_pos <= len(train_data):
        step_data =  train_data[current_data_pos:next_data_pos]
        step_label = train_label[current_data_pos:next_data_pos]
        current_data_pos = next_data_pos
    else:
        step_data = train_data[0:FLAGS.batch_size]
        step_label = train_label[0:FLAGS.batch_size]
        current_data_pos = FLAGS.batch_size
    return step_data, step_label

def gen_standard_data():
    data = []
    label = []
    a = 3
    b = 5
    c = 10
    for i in range(10000):
        x1 = random.randint(0, 1000)
        x2 = random.randint(0, 1000)
        y = a * x1 + b * x2 + c
        data.append([x1,x2])
        label.append([y])
    return data, label
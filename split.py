
"""
data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
"""

import pickle
import numpy as np
import jittor as jt
import itertools

PERMUTATIONS = list(itertools.permutations(range(4),4))
PERM_ARRAY = np.array(PERMUTATIONS) #[24,4]
PERM_DICT = {value: index for index, value in enumerate(PERMUTATIONS)  }

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data_array(file):
    # return: data: [batch,channel, row, col ]
    dict = unpickle(file)
    data = dict[b'data']
    # labels = np.array(dict[b'labels']) 
    data = data.reshape(10000, 3, 32, 32)
    return data

def load_data(config):
    # return: train_data, test_data
    train_data = []
    test_data = []
    for i in range(1,6):
        file = config['data_dir'] + '/data_batch_' + str(i)
        data = get_data_array(file)
        train_data.append(data)
    train_data = np.concatenate(train_data)

    file = config['data_dir'] + '/test_batch'
    test_data = get_data_array(file)

    return train_data, test_data

def split_data(data):
    ## input: data:[N,3,32,32]
    ## output: new_data:[N,4,3,16,16], labels:[N,4]

    ## split the image into 4 patches

    new_data = []
    N = data.shape[0]
    for ind in range(4):
        row_ind = ind//2
        col_ind = ind%2
        new_data.append(data[:,None,:,row_ind*16:16+row_ind*16,col_ind*16:16+col_ind*16])
        # print(new_data[ind].shape,f" ({row_ind*16,16+row_ind*16,col_ind*16,16+col_ind*16}) ")
    new_data= np.concatenate(new_data,axis=1)

    return new_data

def shuffle_permutation(data,config):
    N = data.shape[0]
    if config["method"] == "classic":
        labels = np.arange(0,4,1).reshape(1,4).repeat(N,axis=0)
        np.apply_along_axis(np.random.shuffle, axis=1, arr=labels)
        new_labels = np.column_stack((np.repeat(np.arange(N), 4), labels.flatten()))
        new_data = data[new_labels[:,0],new_labels[:,1],:,:,:].reshape(N,4,3,16,16)
    elif config["method"] == "relate":
        labels = np.random.randint(24,size=N) #[N,]
        label_arr = PERM_ARRAY[labels]  #[N,4]
        new_labels = np.column_stack((np.repeat(np.arange(N), 4), label_arr.flatten()))
        new_data = data[new_labels[:,0],new_labels[:,1],:,:,:].reshape(N,4,3,16,16)


    return new_data,labels

def get_permutation_matrix(P):
    # P:[batch,permutation]  (N,4)
    # ret: P:[batch,4,4] DSM
    res = jt.nn.one_hot(P,4)
    return res
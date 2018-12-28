import tensorflow as tf
import numpy as np
from collections import namedtuple
DatasetTuple = namedtuple('DatasetTuple', field_names=['x_train', 'x_test', 'y_train', 'y_test'])

def load_mnist(cache_location='C:/SHUYANG/mnist'):
    '''
    Loads and returns MNIST data

    Args:
        cache_location (str, optional): Defaults to 'C:/SHUYANG/mnist'. Location of cached MNIST training and testing feature/label arrays
    
    Returns:
        DatasetTuple: Contains the various numpy arrays:
            x_train: Array of matrices representing scaled pixel values (0 to 255 -> 0 to 1) for 60,000 training images
            x_test: Array of matrices representing scaled pixel values (0 to 255 -> 0 to 1) for 10,000 testing images
            y_train: Array of 60,000 training labels (1-9, integer)
            y_test: Array of 10,000 testing labels (1-9, integer)
    '''

    npz_loc = cache_location if cache_location.endswith('.npz') else '{}.npz'.format(cache_location)
    try:
        # Load from NPZ cache
        mnist = np.load(npz_loc)

        # Return relevant data
        print('Loaded MNIST data from local cache {}'.format(npz_loc))
        return DatasetTuple(mnist['x_train'], mnist['x_test'], mnist['y_train'], mnist['y_test'])
    except Exception as e:
        print('Unable to load from cache {} - downloading from AWS: {}'.format(npz_loc, e))

        # Retrieve and format MNIST
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Save to NPZ
        np.savez_compressed(cache_location, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        return DatasetTuple(x_train, x_test, y_train, y_test)

import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file, encoding='bytes'):
    with open(file, 'rb') as fptr:
        data = pickle.load(fptr, encoding=encoding)
    return data

def preprocess(data):
    m = data.shape[0]
    data = data / 255
    data = data.reshape(m, 3, 32, 32).transpose([0, 2, 3, 1])
    return data

def one_hot_encode(labels, num_classes):
    return np.identity(num_classes)[labels]

train_data = unpickle('cifar-100-python/train')
test_data = unpickle('cifar-100-python/test')
meta_data = unpickle('cifar-100-python/meta', encoding='ASCII')

X_train = preprocess(train_data[b'data'])
Y_train = one_hot_encode(train_data[b'fine_labels'], 100)
X_test = preprocess(test_data[b'data'])
Y_test = one_hot_encode(test_data[b'fine_labels'], 100)

print("Shape of training images:", X_train.shape)
print("Shape of training labels:", Y_train.shape)
print("Shape of testing images:", X_test.shape)
print("Shape of testing labels:", Y_test.shape)

plt.imshow(X_train[0])
plt.show()
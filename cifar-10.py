import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

def unpickle(file, encoding='bytes'):
    with open(file, 'rb') as fptr:
        data = pickle.load(fptr, encoding=encoding)
    return data

def one_hot_encode(labels, num_classes):
    return np.identity(num_classes)[labels]

def get_data():
    files = ["cifar-10-batches-py/data_batch_1",
        "cifar-10-batches-py/data_batch_2",
        "cifar-10-batches-py/data_batch_3",
        "cifar-10-batches-py/data_batch_4",
        "cifar-10-batches-py/data_batch_5"
    ]

    X_train = np.zeros((50000, 3072))
    Y_train = np.zeros(50000, dtype=np.int)

    test_data = unpickle("cifar-10-batches-py/test_batch")
    X_test = test_data[b'data']
    Y_test = test_data[b'labels']
    
    for i, file in enumerate(files):
        batch = unpickle(file)
        X_train[i * 10000 : (i + 1) * 10000] = batch[b'data']
        Y_train[i * 10000 : (i + 1) * 10000] = batch[b'labels']

    m = X_train.shape[0]
    X_train = X_train / 255
    X_train = X_train.reshape(m, 3, 32, 32).transpose([0, 2, 3, 1])

    m = X_test.shape[0]
    X_test = X_test / 255
    X_test = X_test.reshape(m, 3, 32, 32).transpose([0, 2, 3, 1])

    Y_train = one_hot_encode(Y_train, 10)
    Y_test = one_hot_encode(Y_test, 10)

    class_names = unpickle('cifar-10-batches-py/batches.meta', encoding="ASCII")

    class_names = class_names['label_names']

    return X_train, X_test, Y_train, Y_test, class_names

def get_mini_batches(X, Y, mini_batch_size):
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X = X[permutation]
    Y = Y[permutation]

    num_mini_batches = m // mini_batch_size

    mini_batches = []

    for i in range(num_mini_batches):
        mini_batches.append({
            'X': X[i * mini_batch_size : (i + 1) * mini_batch_size],
            'Y': Y[i * mini_batch_size : (i + 1) * mini_batch_size]
        })

    if m / mini_batch_size > num_mini_batches:
        mini_batches.append({
            'X': X[num_mini_batches * mini_batch_size :],
            'Y': Y[num_mini_batches * mini_batch_size :]
        })

    return mini_batches

def add_conv_layer(X, filter_size, num_op, stride, padding, name):
    filt = tf.Variable(tf.random_normal([filter_size[0], filter_size[1], X.shape[3].value, num_op]))
    b = tf.Variable(tf.zeros([num_op]))
    return tf.nn.relu(tf.nn.conv2d(X, filt, stride, padding) + b, name = name)

def add_dense(X, num_op):
    W = tf.Variable(tf.random_normal([X.shape[1].value,num_op]))
    b = tf.Variable(tf.zeros([num_op]))
    return tf.matmul(X,W) + b


X_train, X_test, Y_train, Y_test, class_names = get_data()

print("Shape of training images:", X_train.shape)
print("Shape of training labels:", Y_train.shape)
print("Shape of testing images:", X_test.shape)
print("Shape of testing labels:", Y_test.shape)

f, axarr = plt.subplots(5,5)
for cnt1 in range(5):
    for cnt2 in range(5):
        num = random.randint(1,X_train.shape[0])
        axarr[cnt1,cnt2].imshow(X_train[num])
        axarr[cnt1,cnt2].set_title(class_names[np.argmax(Y_train[num])])
        axarr[cnt1,cnt2].set_xticks([])
        axarr[cnt1,cnt2].set_yticks([])

f.tight_layout()
plt.show()

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, Y_train, epochs=1, batch_size=128)

X = tf.placeholder(tf.float32, [None,32,32,3], name = 'X')
Y = tf.placeholder(tf.float32, [None,10], name = 'Y')

conv1 = add_conv_layer(X, [3,3], 32, [1,1,1,1], 'VALID', 'conv1')
conv2 = add_conv_layer(conv1, [3,3], 32, [1,1,1,1], 'VALID', 'conv2')
mp = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], 'VALID', name = 'mp')
drop1 = tf.nn.dropout(mp, 0.25, name = 'drop1')
flat = tf.reshape(drop1, [-1, drop1.shape[1].value * drop1.shape[2].value * drop1.shape[3].value], name = 'flat')
dense = tf.nn.relu(add_dense(flat, 128), name = 'dense')
drop2 = tf.nn.dropout(dense, 0.5, name = 'drop2')
logits = add_dense(drop2, 10)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name="xent")
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
epochs = 1
history = []
sess = tf.Session()
sess.run(init)
for epoch in range(epochs):
    mini_batches = get_mini_batches(X_train, Y_train, 128)
    for i, mini_batch in enumerate(mini_batches):
        dummy, c = sess.run([train_step, loss], feed_dict={
                X: mini_batch['X'],
                Y: mini_batch['Y']
            })
        if (i + 1) % 10 == 0:
            print("Minibatch {}: Cost {}".format(i + 1, c))
            train_accuracy = sess.run(accuracy, feed_dict={X: mini_batches[0]['X'], Y: mini_batches[0]['Y']})
            print("Train accuracy::", train_accuracy)
        history.append(c)
    print("Epoch {} completed.".format(epoch + 1))

sess.close()
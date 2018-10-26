import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


def get_data():
    def unpickle(file, encoding='bytes'):
        with open(file, 'rb') as fptr:
            data = pickle.load(fptr, encoding=encoding)
        return data

    def one_hot_encode(labels, num_classes):
        return np.identity(num_classes)[labels]
    
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
        num_mini_batches += 1

    return mini_batches, num_mini_batches


def add_conv_layer(X, filter_size, num_op, stride, padding, name):
    with tf.variable_scope(name):
        filt = tf.get_variable("filter", shape=[filter_size[0], filter_size[1], X.shape[3].value, num_op], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", shape=[num_op], initializer=tf.zeros_initializer())
        act = tf.nn.relu(tf.nn.conv2d(X, filt, stride, padding) + b, name)
        tf.summary.histogram("weights", filt)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act


def add_dense(X, num_op, name, activation=''):
    with tf.variable_scope(name):
        W = tf.get_variable("weights", shape=[X.shape[1].value,num_op], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("bias", shape=[num_op], initializer=tf.zeros_initializer())
        Z = tf.matmul(X,W) + b
        if activation == 'relu':
            act = tf.nn.relu(Z)
        elif activation == 'softmax':
            act = tf.nn.softmax(Z)
        elif activation == '':
            act = Z
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act


def keras_model(X_train, Y_train, epochs):
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=1)
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    logger = keras.callbacks.TensorBoard(
        log_dir='cifar10-logs/',
        histogram_freq=1,
        write_graph=True
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=128, callbacks = [logger])
    return model


# model = keras_model(X_train, Y_train, 3)
# model.save("keras_model_1.h5")


def tensorflow_model(X_train, Y_train, epochs):
    X = tf.placeholder(tf.float32, [None,32,32,3], name = 'X')
    Y = tf.placeholder(tf.float32, [None,10], name = 'Y')

    conv1 = add_conv_layer(X, [3,3], 64, [1,1,1,1], 'VALID', 'conv1')
    conv2 = add_conv_layer(conv1, [3,3], 64, [1,1,1,1], 'VALID', 'conv2')
    mp = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], 'VALID', name = 'max_pool')
    drop1 = tf.nn.dropout(mp, 0.25, name = 'drop1')
    flat = tf.reshape(drop1, [-1, drop1.shape[1].value * drop1.shape[2].value * drop1.shape[3].value], name = 'flat')
    dense = add_dense(flat, 256, name='dense1', activation='relu')
    drop2 = tf.nn.dropout(dense, 0.5, name = 'drop2')
    logits = add_dense(drop2, 10, name='dense2', activation='')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name="xent")
    train_step = tf.train.AdamOptimizer().minimize(loss)
    predict = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predict, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('summary'):
        tf.summary.scalar('current_cost', loss)
        tf.summary.scalar('current_accuracy', accuracy)
        summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        training_writer = tf.summary.FileWriter("./cifar10-logs/training", sess.graph)
        for epoch in range(epochs):
            mini_batches, num_mini_batches = get_mini_batches(X_train, Y_train, 128)
            for i, mini_batch in enumerate(mini_batches):
                sess.run(train_step, feed_dict={X: mini_batch['X'], Y: mini_batch['Y']})
                if (i + 1) % 10 == 0:
                    c, train_accuracy, train_summary = sess.run([loss, accuracy, summary], feed_dict={X: mini_batches[i]['X'], Y: mini_batches[i]['Y']})
                    training_writer.add_summary(train_summary, epoch * num_mini_batches + i)
                    print("Minibatch {}: Cost {}, Train accuracy: {}".format(i + 1, c, train_accuracy))
            print("Epoch {} completed.".format(epoch + 1))

        save_path = saver.save(sess, './models/cifar10-test1.ckpt')
        print("Model saved at: {}".format(save_path))

tensorflow_model(X_train, Y_train, 5)
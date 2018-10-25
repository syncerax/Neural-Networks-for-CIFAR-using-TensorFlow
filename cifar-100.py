import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

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

def add_conv_layer(X, filter_size, num_op, stride, padding, name):
    filt = tf.Variable(tf.random_normal([filter_size[0], filter_size[1], X.shape[3].value, num_op]))
    b = tf.Variable(tf.zeros([num_op]))
    return tf.nn.relu(tf.nn.conv2d(X, filt, stride, padding) + b, name = name)

def add_dense(X, num_op):
    W = tf.Variable(tf.random_normal([X.shape[1].value,num_op]))
    b = tf.Variable(tf.zeros([num_op]))
    return tf.matmul(X,W) + b

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

train_data = unpickle('cifar-100-python/train')
test_data = unpickle('cifar-100-python/test')
meta_data = unpickle('cifar-100-python/meta', encoding='ASCII')

class_names = meta_data['fine_label_names']

X_train = preprocess(train_data[b'data'])
Y_train = one_hot_encode(train_data[b'fine_labels'], 100)
X_test = preprocess(test_data[b'data'])
Y_test = one_hot_encode(test_data[b'fine_labels'], 100)

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

X = tf.placeholder(tf.float32, [None,32,32,3], name = 'X')
Y = tf.placeholder(tf.float32, [None,100], name = 'Y')

conv1 = add_conv_layer(X, [3,3], 32, [1,1,1,1], 'VALID', 'conv1')
conv2 = add_conv_layer(conv1, [3,3], 32, [1,1,1,1], 'VALID', 'conv2')
mp = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], 'VALID', name = 'mp')
drop1 = tf.nn.dropout(mp, 0.25, name = 'drop1')
flat = tf.reshape(drop1, [-1, drop1.shape[1].value * drop1.shape[2].value * drop1.shape[3].value], name = 'flat')
dense = tf.nn.relu(add_dense(flat, 128), name = 'dense')
drop2 = tf.nn.dropout(dense, 0.5, name = 'drop2')
logits = add_dense(drop2, 100)
hypothesis = tf.nn.softmax(logits)
prediction = tf.argmax(logits,axis = 1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name="xent")
train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
epochs = 2
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
        history.append(c)
    print("Epoch {} completed.".format(epoch + 1))
train_pred = sess.run(prediction, feed_dict = {X: X_train[:100], Y: Y_train[:100]})
test_pred = sess.run(hypothesis, feed_dict = {X: X_test, Y: Y_test})

train_acc = np.mean(train_pred == np.argmax(Y_train[:100], axis = 1)) * 100
test_acc = np.mean(test_pred == np.argmax(Y_test, axis = 1)) * 100
print("Training accuracy::", train_acc)
print("Testing accuracy::", test_acc)
sess.close()

# model = Sequential([
#     Conv2D(128, kernel_size=(3, 3), padding = "same", activation='elu', input_shape=(32, 32, 3)),
#     Conv2D(128, kernel_size=(3, 3), activation='elu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.1),
#     Conv2D(256, kernel_size=(3, 3), padding = "same", activation = 'elu'),
#     Conv2D(256, kernel_size=(3, 3), activation='elu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Conv2D(512, kernel_size=(3, 3), padding = "same", activation = 'elu'),
#     Conv2D(512, kernel_size=(3, 3), activation='elu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.5),
#     Flatten(),
#     Dense(1024, activation='elu'),
#     Dropout(0.5),
#     Dense(100, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss=keras.losses.categorical_crossentropy,
#     metrics=['accuracy']
# )

# model.fit(X_train, Y_train, epochs=1, batch_size=128)
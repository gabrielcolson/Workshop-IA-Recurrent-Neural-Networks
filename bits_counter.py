import numpy as np
from random import shuffle
import tensorflow as tf

NUM_EXAMPLES = 10000

train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int, i) for i in train_input]
ti = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    train_output.append(count)

X_test = train_input[NUM_EXAMPLES:]
y_test = train_output[NUM_EXAMPLES:]
X_train = train_input[:NUM_EXAMPLES]
y_train = train_output[:NUM_EXAMPLES]






n_units = 24
n_steps = 20
n_inputs = 1
n_outputs = 21

learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

cell = tf.nn.rnn_cell.LSTMCell(num_units=n_units, state_is_tuple=True)
_, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

logits = tf.layers.dense(state[1], n_outputs)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 5000
batch_size = 1000
n_batches = len(X_train) // batch_size
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        ptr = 0
        for iteration in range(n_batches):
            X_batch, y_batch = X_train[ptr:ptr + batch_size], y_train[ptr:ptr + batch_size]
            ptr += batch_size
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        print(epoch, "Test accuracy:", acc_test)

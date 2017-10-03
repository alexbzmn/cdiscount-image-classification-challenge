import tensorflow as tf

from tf_approach.pre_process import X_train, y_train, X_flattened_shape, Y_flattened_shape

# set up params
seed = 2
tf.set_random_seed(2)
batch_size = 100
LEARNING_RATE = 1e-4

x = tf.placeholder(tf.float32, [None, X_flattened_shape])
y_ = tf.placeholder(tf.float32, [None, Y_flattened_shape])

W = tf.Variable(tf.zeros([X_flattened_shape, Y_flattened_shape]))
b = tf.Variable(tf.zeros([Y_flattened_shape]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.run(train_step, feed_dict={x: X_train, y_: y_train})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: X_train, y_: y_train}))

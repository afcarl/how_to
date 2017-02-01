# How to get started with tensorflow
# 
# Minimum get started guide for using tensorflow.
# By the end of this guide, you will be able to 
# train a convolutional neural network on MNIST 
# dataset
#
# Install tensorflow with pip:
#   `pip install tensorflow`
#
# Adapted from the Tensorflow examples:
#   https://github.com/aymericdamien/TensorFlow-Examples
import tensorflow as tf

# hello world example
hello = tf.constant('Hello world!')
sess = tf.Session()
print sess.run(hello)

# basic operations
a = tf.constant(2)
b = tf.constant(3)
print sess.run(a+b)

# place holder
c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)

add = tf.add(c,d)
mul = tf.mul(a,b)

print sess.run(add, feed_dict={c:2, d:3})
print sess.run(mul, feed_dict={c:2, d:3})

# matrix operations
mat1 = tf.constant([[1.,2.]])
mat2 = tf.constant([[2.],[2.]])

prod = tf.matmul(mat1, mat2)
print sess.run(prod)

mat3 = tf.placeholder(tf.float32, shape=(1, 2))
mat4 = tf.placeholder(tf.float32, shape=(2, 1))
prod2 = tf.matmul(mat3, mat4)
print sess.run(prod2, feed_dict={mat3: [[1.,2.]], mat4: [[2.],[2.]]})




# CNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

lr = 0.001      # learning rate
batch_size = 64 # mini-batch size

n_input = 784   # number of pixels for each input
n_output = 10   # number of classes in MNIST dataset

x = tf.placeholder(tf.float32, [None, n_input])     # Here dimension "None" depends on the batch_size
y = tf.placeholder(tf.float32, [None, n_output])
drop_out_holder = tf.placeholder(tf.float32)

# layer wrappers

def conv_layer(x, W, b, stride=1):
  # https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d
  # 
  # - x: 4d tensor [batch, height, width, channels]
  # - W, b: weights
  # - strides[0] and strides[1] must be 1
  # - padding can be 'VALID'(without padding) or 'SAME'(zero padding)
  #     - http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
  x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
  x = tf.nn.bias_add(x, b) # add bias term
  return tf.nn.relu(x) # rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

def max_pooling(x, k_sz=2):
  # https://www.tensorflow.org/api_docs/python/nn/pooling#max_pool
  # 
  # - x: 4d tensor [batch, height, width, channels]
  # - ksize: The size of the window for each dimension of the input tensor
  return tf.nn.max_pool(x, ksize=[1, k_sz, k_sz, 1], strides=[1, k_sz, k_sz, 1], padding='SAME')

def conv_net(x, weights, biases, drop_out):
  # If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. 
  # In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
  x = tf.reshape(x, shape=[-1,28,28,1])

  conv1 = conv_layer(x, weights['wc1'], biases['bc1'])
  conv1 = max_pooling(conv1, k_sz=2)

  conv2 = conv_layer(conv1, weights['wc2'], biases['bc2'])
  conv2 = max_pooling(conv2, k_sz=2)

  fc1 = tf.reshape(conv2, [-1, weights['wf1'].get_shape().as_list()[0]])
  fc1 = tf.add(tf.matmul(fc1, weights['wf1']), biases['bf1'])
  fc1 = tf.nn.relu(fc1)
  fc1 = tf.nn.dropout(fc1, drop_out)

  out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
  return out

# initialize layers weight & bias
weights = {
  # 5x5 conv, 1 input, 32 outputs
  'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
  # 5x5 conv, 32 inputs, 64 outputs
  'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
  # fully connected, 7*7*64 inputs, 1024 outputs
  'wf1': tf.Variable(tf.random_normal([7*7*64, 1024])),
  # 1024 inputs, 10 outputs (class prediction)
  'out': tf.Variable(tf.random_normal([1024, n_output]))
}


biases = {
  'bc1': tf.Variable(tf.random_normal([32])),
  'bc2': tf.Variable(tf.random_normal([64])),
  'bf1': tf.Variable(tf.random_normal([1024])),
  'out': tf.Variable(tf.random_normal([n_output]))
}

net = conv_net(x, weights, biases, drop_out_holder)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


correct_pred = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

sess.run(init)

iterations = 20000
for i in xrange(iterations):
  batch_x, batch_y = mnist.train.next_batch(batch_size)
  sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       drop_out_holder: 0.75})

  if i % 10 == 0:
    l, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, drop_out_holder: 1.})



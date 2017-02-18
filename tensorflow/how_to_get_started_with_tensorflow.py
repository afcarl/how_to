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
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print sess.run(hello)

# basic operations
a = tf.constant(2)
b = tf.constant(3)
print sess.run(a+b)

# place holder
c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)

add = tf.add(c,d)
mul = tf.multiply(a,b)

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





# train a CNN on MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

lr = 0.001      # learning rate
batch_size = 64 # mini-batch size

n_input = 784   # number of pixels for each input
n_output = 10   # number of classes in MNIST dataset


# layer wrappers
def max_pool(x, k_sz=[2,2]):
  """max pooling layer wrapper
  Args
    x:      4d tensor [batch, height, width, channels]
    k_sz:   The size of the window for each dimension of the input tensor
  Returns
    a max pooling layer
  """
  return tf.nn.max_pool(x, ksize=[1, k_sz[0], k_sz[1], 1], strides=[1, k_sz[0], k_sz[1], 1], padding='SAME')

def conv2d(x, n_kernel, k_sz, stride=1):
  """convolutional layer with relu activation wrapper
  Args:
    x:          4d tensor [batch, height, width, channels]
    n_kernel:   number of kernels (output size)
    k_sz:       2d array, kernel size. e.g. [8,8]
    stride:     stride
  Returns
    a conv2d layer
  """
  W = tf.Variable(tf.random_normal([k_sz[0], k_sz[1], int(x.get_shape()[3]), n_kernel]))
  b = tf.Variable(tf.random_normal([n_kernel]))
  # - strides[0] and strides[1] must be 1
  # - padding can be 'VALID'(without padding) or 'SAME'(zero padding)
  #     - http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
  conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
  conv = tf.nn.bias_add(conv, b) # add bias term
  return tf.nn.relu(conv) # rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)


def fc(x, n_output, activation_fn=None):
  """fully connected layer with relu activation wrapper
  Args
    x:          2d tensor [batch, n_input]
    n_output    output size
  """
  W=tf.Variable(tf.random_normal([int(x.get_shape()[1]), n_output]))
  b=tf.Variable(tf.random_normal([n_output]))
  fc1 = tf.add(tf.matmul(x, W), b)
  if not activation_fn == None:
    fc1 = activation_fn(fc1)
  return fc1


def flatten(x):
  """flatten a 4d tensor into 2d
  Args
    x:          4d tensor [batch, height, width, channels]
  Returns a flattened 2d tensor
  """
  return tf.reshape(x, [-1, int(x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3])])




def conv_net(x, drop_out):
  # If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. 
  # In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
  x = tf.reshape(x, shape=[-1,28,28,1])
  conv1 = conv2d(x, n_kernel=32, k_sz=[5,5], stride=1)
  conv1 = max_pool(conv1, k_sz=[2,2])
  conv2 = conv2d(conv1, n_kernel=64, k_sz=[5,5], stride=1)
  conv2 = max_pool(conv2, k_sz=[2,2])
  # flattening
  # fc1 = tf.reshape(conv2, [-1, 7*7*64])
  fc1 = flatten(conv2)
  fc1 = fc(fc1, activation_fn=tf.nn.relu, n_output=1024)
  fc1 = tf.nn.dropout(fc1, drop_out)
  out = fc(fc1, n_output=n_output)
  return out


x = tf.placeholder(tf.float32, [None, n_input])     # Here dimension "None" depends on the batch_size
y = tf.placeholder(tf.float32, [None, n_output])
drop_out_holder = tf.placeholder(tf.float32)


with tf.device("/cpu:0"):
  net = conv_net(x, drop_out_holder)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
  optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

  correct_pred = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
# newer version of tensorflow:
# init = tf.global_variables_initializer()

sess.run(init)

iterations = 10000
for i in xrange(iterations):
  batch_x, batch_y = mnist.train.next_batch(batch_size)
  sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       drop_out_holder: 0.75})

  if i % 10 == 0:
    # Calculate batch loss and accuracy
    l, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, drop_out_holder: 1.})
    print("Iter " + str(i*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(l) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))


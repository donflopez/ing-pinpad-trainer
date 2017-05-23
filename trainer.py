# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os

dataset_path      = os.getcwd() + '/samples/'
test_labels_file  = "test-labels.csv"
train_labels_file = "train-labels.csv"

test_set_size = 5

IMAGE_HEIGHT  = 60
IMAGE_WIDTH   = 60
NUM_CHANNELS  = 3
BATCH_SIZE    = 10

numbers = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]

def encode_label(label):
  return int(label)

def read_label_file(file):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    filepath, label = line.split(",")
    filepaths.append(filepath)
    labels.append(encode_label(label))
  return filepaths, labels

# reading labels and file path
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

# transform relative path into full path
train_filepaths = [ dataset_path + fp for fp in train_filepaths]
test_filepaths = [ dataset_path + fp for fp in test_filepaths]

# for this example we will create or own test partition
all_filepaths = train_filepaths + test_filepaths
all_labels = train_labels + test_labels

all_filepaths = all_filepaths[:20]
all_labels = all_labels[:20]

# convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# create a partition vector

# create input queues
train_input_queue = tf.train.slice_input_producer(
                                    [all_images, all_labels],
                                    shuffle=False)
test_input_queue = tf.train.slice_input_producer(
                                    [all_images, all_labels],
                                    shuffle=False)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
train_image = tf.cast(train_image, tf.float32)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
test_image = tf.cast(test_image, tf.float32)
test_label = test_input_queue[1]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )

print "input pipeline ready"

x = tf.placeholder(tf.float32, [None, 10800])
W = tf.Variable(tf.zeros([10800, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
  # initialize the variables
sess.run(tf.global_variables_initializer())

# initialize the queue threads to start to shovel data
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

print "from the train set:"
def normalize(data):
    return [ (255-x)*1.0/255.0 for x in data]

train_image_batch = tf.reshape(train_image_batch, [10, -1])

for _ in range(100):
    X = sess.run(train_image_batch)
    Y = sess.run(train_label_batch)
    X = normalize(X)
    for a in X:
        sm = 0
        for v in a:
            sm += v
    Y_ = []
    for i in Y:
        Y_.append(numbers[i])
    sess.run(train_step, feed_dict={x: X, y_: Y_})

test_image_batch = tf.reshape(test_image_batch, [10, -1])
X = sess.run(test_image_batch)
X = normalize(X)
Y = sess.run(test_label_batch)

Y_ = []
for i in Y:
    Y_.append(numbers[i])

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: X,
                                    y_: Y_}))

# stop our queue threads and properly close the session

print "Saving session:"
saver = tf.train.Saver()
saver.save(sess, 'model/mymodel')
print "saved!"

coord.request_stop()
coord.join(threads)
sess.close()
#!/usr/bin/env python3
import os.path
import shutil
import time
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



# defining hyperparameters globally so that they can be accessed anywhere in the code
EPOCHS = 60
BATCH_SIZE = 7
LEARNING_RATE = 0.0001
KEEP_PROBABILITY = 0.75

# more parameters
NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)
DATA_DIR = './data'
RUNS_DIR = './runs'

DEBUG = False
GRAPH_DIR = './graphs'
DEBUG_DIR = './debug'
USE_INITIALIZER = True
# INITIALIZER_CLASS = "tf.contrib.layers.xavier_initializer"  # NOT working with 3 classes :-/
INITIALIZER_CLASS = "tf.truncated_normal_initializer"       # working with 3 classes
INITIALIZER_STDDEV = 0.008                                  # 0.15: working with 3 classes (tested until 60 epochs)
USE_REGULARIZER = True
REGULARIZER_SCALE = 0.001
PROBABILITY_THRESHOLD = 0.85
LEARNING_RATE_DECAY_AFTER_EPOCHS = 10
LEARNING_RATE_DECAY_MULTIPLIER = 1.0
# TEST_FILE_PATTERN = "umm_000009.png"
TEST_FILE_PATTERN = "*.png"



def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # downloading the pretrained vgg model so that the function passes the unit test
    # on my AWS instance when running for the first time
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # straightforward model loading...
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    # ...and tensor extractionn
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def get_initializer():
    """
    Helper function for NN assembling
    """
    initializer = None
    if USE_INITIALIZER:

        # not really a Pythonic solution, but it does the job :-/ 
        if INITIALIZER_CLASS == "tf.contrib.layers.xavier_initializer":
            initializer = tf.contrib.layers.xavier_initializer()

        elif INITIALIZER_CLASS == "tf.truncated_normal_initializer":
            initializer = tf.truncated_normal_initializer(stddev=INITIALIZER_STDDEV)

        elif INITIALIZER_CLASS == "tf.random_normal_initializer":
            initializer = tf.random_normal_initializer(stddev=INITIALIZER_STDDEV)

        else:
            pass

    return initializer


def get_regularizer():
    """
    Helper function for NN assembling
    """
    return tf.contrib.layers.l2_regularizer(scale=REGULARIZER_SCALE) if USE_REGULARIZER else None



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # 1x1 convolution on layer 7
    vgg_layer7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                          kernel_initializer=get_initializer(),
                                          kernel_regularizer=get_regularizer(),
                                          name='vgg_layer7_conv1x1')
    # upsampling: 2x layer 7
    vgg_layer7_x2 = tf.layers.conv2d_transpose(vgg_layer7_conv1x1, num_classes, 4, strides=2, padding='same',
                                               kernel_initializer=get_initializer(),
                                               kernel_regularizer=get_regularizer(),
                                               name='vgg_layer7_x2')
    # scaling + 1x1 convolution on layer 4
    vgg_layer4_scaled = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_scaled')
    vgg_layer4_conv1x1 = tf.layers.conv2d(vgg_layer4_scaled, num_classes, 1, padding='same',
                                          kernel_initializer=get_initializer(),
                                          kernel_regularizer=get_regularizer(),
                                          name='vgg_layer4_conv1x1')
    # adding: layer4 + 2x layer7
    output_layer = tf.add(vgg_layer7_x2, vgg_layer4_conv1x1)

    # upsampling: 2x (layer4 + 2x layer7)
    output_layer = tf.layers.conv2d_transpose(output_layer, num_classes, 4, strides=2, padding='same',
                                              kernel_initializer=get_initializer(),
                                              kernel_regularizer=get_regularizer(),
                                              name='upsampling_2x__layer4_plus_2x_layer7')
    # scaling + 1x1 convolution on layer 3
    vgg_layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    vgg_layer3_conv1x1 = tf.layers.conv2d(vgg_layer3_scaled, num_classes, 1, padding='same',
                                          kernel_initializer=get_initializer(),
                                          kernel_regularizer=get_regularizer(),
                                          name='vgg_layer3_conv1x1')

    # adding: layer3 + 2x (layer4 + 2x layer7)
    output_layer = tf.add(output_layer, vgg_layer3_conv1x1)

    # upsampling: 8x (layer3 + 2x (layer4 + 2x layer7) )
    output_layer = tf.layers.conv2d_transpose(output_layer, num_classes, 16, strides=8, padding='same',
                                              kernel_initializer=get_initializer(),
                                              kernel_regularizer=get_regularizer(),
                                              name='upsampling_8x__layer3_plus_2x__layer4_plus_2x_layer7')
    return output_layer

tests.test_layers(layers)



def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # "the output tensor is 4D so we have to reshape it to 2D"
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # standard cross entropy loss...
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    # ...plus the losses of the regularizers (if using them)
    if USE_REGULARIZER:
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += sum(regularization_losses)

    # using adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return logits, train_op, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # output
    log = []
    log.append("TensorFlow version: {}".format(tf.__version__))
    if tf.test.gpu_device_name():
        log.append("Default GPU device: {}".format(tf.test.gpu_device_name()))

    # learning rate will be decreased over time
    learning_rate_var = LEARNING_RATE
    message = "Learning rate: {:.9f} ({:.3e})".format(learning_rate_var, learning_rate_var)
    log.append(message)
    print(message)

    # measuring training duration
    training_start = time.time()

    # initializing global variables
    sess.run(tf.global_variables_initializer())

    prev_average_loss = 0

    # running through the eopchs
    for epoch in range(epochs):

        # measuring epoch duration
        epoch_start = time.time()
        losses = []

        # learning rate decay
        if float(LEARNING_RATE_DECAY_MULTIPLIER) != 1.0 and epoch > 0 and epoch % LEARNING_RATE_DECAY_AFTER_EPOCHS == 0:

            learning_rate_var *= LEARNING_RATE_DECAY_MULTIPLIER
            message = "New learning rate: {:.9f} ({:.3e})".format(learning_rate_var, learning_rate_var)
            log.append(message)
            print(message)

        # debug message
        print('Epoch {:d}: training...'.format(epoch+1), end='', flush=True)

        for image, label in get_batches_fn(batch_size):

            # assembling the feed dictionary
            train_feed_dict = {input_image: image,
                               correct_label: label,
                               learning_rate: learning_rate_var,
                               keep_prob: KEEP_PROBABILITY}

            # running optimizer and getting the loss
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=train_feed_dict)
            losses.append(loss)

        # bookkeeping
        epoch_duration = time.time() - epoch_start
        average_loss = sum(losses) / float(len(losses))

        prev_loss_message = " (delta: {:+.5f})".format(average_loss - prev_average_loss) if epoch > 0 else ""
        prev_average_loss = average_loss

        # displaying some debug info
        print("\b\b\b took {:.2f} seconds. Average loss: {:.4f}".format(epoch_duration, average_loss) + prev_loss_message)

        log.append("Epoch {:d}: training took {:.2f} seconds. Average loss: {:.4f}".format(epoch+1, epoch_duration, average_loss) + prev_loss_message)

    # displaying some debug info
    training_duration = time.time() - training_start
    training_duration_mins = int(training_duration // 60);
    training_duration_secs = int(training_duration - training_duration_mins * 60);

    min_plural = '' if training_duration_mins == 1 else 's'
    sec_plural = '' if training_duration_secs == 1 else 's'

    training_summary = "Training took {:d} minute{} and {:d} second{}.".format(training_duration_mins, min_plural,
                                                                               training_duration_secs, sec_plural)
    print(training_summary)
    log.append(training_summary)

    return log

tests.test_train_nn(train_nn)




def run():

    # assembling parameters package
    params = {
        'debug': DEBUG,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'keep_probabilty': KEEP_PROBABILITY,
        'use_initializer': USE_INITIALIZER,
        'initializer_class': INITIALIZER_CLASS,
        'initializer_std_dev': INITIALIZER_STDDEV,
        'use_regularizer': USE_REGULARIZER,
        'regularizer_scale': REGULARIZER_SCALE,
        'num_classes': NUM_CLASSES,
        'image_shape': IMAGE_SHAPE,
        'probability_threshold': PROBABILITY_THRESHOLD,
        'learning_rate_decay_after_epochs': LEARNING_RATE_DECAY_AFTER_EPOCHS,
        'learning_rate_decay_multiplier': LEARNING_RATE_DECAY_MULTIPLIER,
        'data_dir': DATA_DIR,
        'runs_dir': RUNS_DIR,
        'graph_dir': GRAPH_DIR,
        'debug_dir': DEBUG_DIR,
        'test_file_pattern': TEST_FILE_PATTERN
    }

    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE, NUM_CLASSES)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        output = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)

        # for TensorBoard (...)
        if DEBUG:
            # cleanup + writing the graph
            if os.path.exists(GRAPH_DIR):
                shutil.rmtree(GRAPH_DIR)
            os.makedirs(GRAPH_DIR)

            writer = tf.summary.FileWriter(GRAPH_DIR, sess.graph)

        # declaring placeholders for the learning rate and for the labels
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(tf.float32, [None, None, None, NUM_CLASSES], name='correct_label')

        logits, train_op, loss = optimize(output, correct_label, learning_rate, NUM_CLASSES)

        # Train NN using the train_nn function
        training_log = train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss, image_input,
                                correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        output_dir = helper.save_inference_samples(sess, logits, keep_prob, image_input, params)

        # saving also the parameters...
        with open(os.path.join(output_dir, "parameters.txt"), "w") as parameters_file:
            for key in params:
                parameters_file.write("{}: {}\n".format((key + " ").ljust(35, "."), params[key]))

        # ...and the training log
        with open(os.path.join(output_dir, "training.log"), "w") as training_log_file:
            for line in training_log:
                training_log_file.write(line + "\n")

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

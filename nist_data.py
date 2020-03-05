import numpy as np
import tensorflow as tf
import glob
import os
import cv2

dir_path = os.getcwd()
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_image_number', 58000, 'Number of images in your tfrecord')
flags.DEFINE_integer('valid_image_number', 7250, 'Number of images in your tfrecord')
flags.DEFINE_integer('test_image_number', 7250, 'Number of images in your tfrecord')
flags.DEFINE_integer('class_number', 29, 'Number of class in your dataset/label.txt')
flags.DEFINE_integer('image_height', 100, 'Height of the output image after crop and resize')
flags.DEFINE_integer('image_width', 100, 'Width of the output image after crop and resize')
flags.DEFINE_string('input_directory',os.path.join(dir_path,'Dataset','Record_file'),'input data directory')

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string)
        self.height = tf.Variable([], dtype = tf.int64)
        self.width = tf.Variable([], dtype = tf.int64)
        self.filename = tf.Variable([], dtype = tf.string)
        self.label = tf.Variable([], dtype = tf.int32)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=1)

    current_image_object = image_object()

    current_image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, FLAGS.image_height, FLAGS.image_width) # cropped image with size 299x299
    current_image_object.height = features["image/height"] # height of the raw image
    current_image_object.width = features["image/width"] # width of the raw image
    current_image_object.filename = features["image/filename"] # filename of the raw image
    current_image_object.label = tf.cast(features["image/class/label"], tf.int32) # label of the raw image
    return current_image_object

def get_file_list_train(data_dir):
    train_list = glob.glob(data_dir + '/' + 'train-*.tfrecord')
    if len(train_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list

def get_file_list_valid(data_dir):
    valid_list = glob.glob(data_dir + '/' + 'validation-*.tfrecord')
    if len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return valid_list

def get_file_list_test(data_dir):
    test_list = glob.glob(data_dir + '/' + 'test-*.tfrecord')
    if len(test_list) == 0:
        raise IOError('No files found at specified path!')
    return test_list

def load_data():
    train_lab = []
    train_img = []
    valid_lab = []
    valid_img = []
    test_lab = []
    test_img = []

    train_filename_queue = tf.train.string_input_producer(
            get_file_list_train(FLAGS.input_directory),
            shuffle = True)
    valid_filename_queue = tf.train.string_input_producer(
            get_file_list_valid(FLAGS.input_directory),
            shuffle = True)
    test_filename_queue = tf.train.string_input_producer(
            get_file_list_test(FLAGS.input_directory),
            shuffle = True)

    train_current_image_object = read_and_decode(train_filename_queue)
    valid_current_image_object = read_and_decode(valid_filename_queue)
    test_current_image_object = read_and_decode(test_filename_queue)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(FLAGS.train_image_number):
            pre_train_image, pre_train_label = sess.run([train_current_image_object.image, train_current_image_object.label])
            pre_train_image = pre_train_image.reshape(FLAGS.image_height*FLAGS.image_width)
            train_img.append(pre_train_image)
            train_lab.append(pre_train_label)

        for i in range(FLAGS.valid_image_number):
            pre_valid_image, pre_valid_label = sess.run([valid_current_image_object.image, valid_current_image_object.label])
            pre_valid_image = pre_valid_image.reshape(FLAGS.image_height*FLAGS.image_width)
            valid_img.append(pre_valid_image)
            valid_lab.append(pre_valid_label)

        for i in range(FLAGS.test_image_number):
            pre_test_image, pre_test_label = sess.run([test_current_image_object.image, test_current_image_object.label])
            pre_test_image = pre_test_image.reshape(FLAGS.image_height*FLAGS.image_width)
            test_img.append(pre_test_image)
            test_lab.append(pre_test_label)

        train_data = np.asarray(train_img,dtype = np.float32)
        train_data = (255.0 - train_data)/255.0
        train_label = np.asarray(train_lab,dtype = np.int32)
        train_label = train_label

        eval_data = np.asarray(valid_img,dtype = np.float32)
        eval_data = (255.0 - eval_data)/255.0
        eval_label = np.asarray(valid_lab,dtype = np.int32)
        eval_label = eval_label

        test_data = np.asarray(test_img,dtype = np.float32)
        test_data = (255.0 - test_data)/255.0
        test_label = np.asarray(test_lab,dtype = np.int32)
        test_label = test_label

        coord.request_stop()
        coord.join(threads)
    return train_data, train_label, eval_data, eval_label, test_data, test_label

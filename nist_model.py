import tensorflow as tf
import numpy as np

def cnn_model(features, labels, mode):
    learning_rate = 0.001
    kernel_size1 = [11,11]
    kernel_size2 = [5,5]
    kernel_size3 = [3,3]

    filter1 = 32
    filter2 = 64
    filter3 = 128
    n_output = 29
    input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])

    conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=filter1,
    kernel_size=kernel_size1,
    padding="same",
    activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2, 2],strides=2,padding='same')

    conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=filter2,
    kernel_size=kernel_size2,
    padding="same",
    activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2, 2],strides=2,padding='same')

    conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=filter3,
    kernel_size=kernel_size3,
    padding="same",
    activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2, 2],strides=2,padding='same')

    pool3_flat = tf.reshape(pool3, [-1, 13 * 13 * filter3])

    dense = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
    inputs=dense, rate=0.5)

    logits = tf.layers.dense(inputs=dropout, units=n_output)

    predictions = {"classes": tf.argmax(input=logits, axis=1),
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

import tensorflow as tf
import numpy as np
import os

from nist_data import *
from nist_model import *

output_path = os.path.join(os.getcwd(),'Model')
train_step = 20000
batch_size = 128

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        train_data,train_labels,eval_data,eval_labels,_,_ = load_data()

        nist_classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir=output_path)

        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,throttle_secs=400)
        
        # train and evaluate model
        tf.estimator.train_and_evaluate(nist_classifier, train_spec, eval_spec)

        print("Complete!!")

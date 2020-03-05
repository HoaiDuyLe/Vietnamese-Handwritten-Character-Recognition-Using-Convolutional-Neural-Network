import tensorflow as tf
import numpy as np
import os

from nist_data import *
from nist_model import *

output_path = os.path.join(os.getcwd(),'model')

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #load data
        _,_,_,_,test_data,test_labels = load_data()

        nist_classifier = tf.estimator.Estimator(model_fn=cnn_model,model_dir=output_path)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=test_labels,
            num_epochs=1,
            shuffle=False)
        test_results = nist_classifier.predict(input_fn=test_input_fn)

        out = []    # list prediction of all image in test data
        for result in test_results:
            out.append(result["classes"])
        out = np.asarray(out,dtype=np.int32)

        confusion_matrix = tf.confusion_matrix(labels = test_labels, predictions = out, num_classes = 29)
        # calculate accuracy
        correction = tf.equal(test_labels, out)
        accuracy = tf.reduce_mean(tf.cast(correction,tf.float32))

        print('accuracy:',sess.run(accuracy))
        print('confusion_matrix:',sess.run(confusion_matrix))

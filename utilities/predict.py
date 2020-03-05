import tensorflow as tf
import numpy as np
import os
import cv2
from operator import itemgetter

from .separate import separate_char

RESULTS = ['a','ă','â','b','c','d','đ','e','ê','g','h','i','k','l','m','n',
            'o','ô','ơ','p','q','r','s','t','u','ư','v','x','y']

def predict_char(image,model_fn,weight_path):
    '''
    Args: image: an array of pixel value of an image
          model_fn: model cnn
          weight_path: path to directory that trained model is located
    Output: a tuple of (RESULTS[kq],top5_prob)
            RESULTS[kq]: prediction result of input image
            top5_prob: probability of top 5 prediction results with input image
    '''
    x = tf.placeholder(tf.float32, [None, 100*100])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nist_classifier = tf.estimator.Estimator(
            model_fn=model_fn, model_dir = weight_path)

        # translate character to the center of the image
        img = separate_char(image)
        # resize to (100,100)
        cv2.imwrite('tmp/output.png',img)
        img = cv2.resize(img,(100,100))
        img = img.astype(np.float32)
        img = img.reshape(1,100*100)
        img = (255.0 - img)/255.0

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
              x = {"x": img},
              shuffle = False)
        # Run prediction
        results = nist_classifier.predict(input_fn = predict_input_fn)
        for result in results:
            kq = result['classes']
            prob = result['probabilities']

    # Sort top 5 probabilities predicted from each input image
    prob = zip(RESULTS,prob)
    prob = list(prob)
    x = sorted(prob,key = itemgetter(1),reverse=True)
    top5_prob = x[:5]

    return (RESULTS[kq],top5_prob)

def translate(img,new_img):
    # translate to the center of the image
    old_h,old_w = img.shape
    new_h,new_w = new_img.shape
    trans_vec = (int((new_h - old_h)/2),int((new_w - old_w)/2))

    for i in range(old_h):
        for j in range(old_w):
            new_img[i+trans_vec[0]][j+trans_vec[1]] = img[i][j]
    return new_img

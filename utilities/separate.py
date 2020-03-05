import numpy as np

def separate_char(img):
    # Arg image_array: an array of pixel value of an image
    h,w = img.shape
    img_inv = img.T
    x = img.shape[0]        #height
    y = img.shape[1]        #width

    mask_x = np.ones(y)*255
    mask_y = np.ones(x)*255
    idx_x = []      #index of height
    idx_y = []     #index of width
    for i in range(x):
        if not (img[i] == mask_x).all():

            idx_x.append(i)
    for j in range(y):
        if not (img_inv[j] == mask_y).all():
            idx_y.append(j)

    up = idx_x[0]
    down = idx_x[-1]
    left = idx_y[0]
    right = idx_y[-1]
    img = img[up:down,left:right]

    # return img
    new_img = np.ones((h,w))*255.0
    new_img = translate(img,new_img)
    return new_img

def translate(img,new_img):
    old_h,old_w = img.shape
    new_h,new_w = new_img.shape
    trans_vec = (int((new_h - old_h)/2),int((new_w - old_w)/2))

    for i in range(old_h):
        for j in range(old_w):
            new_img[i+trans_vec[0]][j+trans_vec[1]] = img[i][j]
    return new_img

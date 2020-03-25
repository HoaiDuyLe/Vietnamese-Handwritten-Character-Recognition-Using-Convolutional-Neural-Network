from __future__ import absolute_import, division, print_function, unicode_literals
from shutil import copyfile

import os
import random
import pathlib
import shutil


def split_data(SOURCE, TRAINING, TESTING, VALIDATION, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        files.append(filename)

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int((len(files) - training_length)/2)
    validation_length = int((len(files) - training_length)/2)

    shuffled_set = random.sample(files, len(files))

    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:training_length+testing_length]
    validation_set = shuffled_set[-validation_length:]

    for filename in training_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TRAINING, filename)
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TESTING, filename)
        copyfile(this_file, destination)

    for filename in validation_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(VALIDATION, filename)
        copyfile(this_file, destination)

if __name__== "__main__":
    source_path = "Path to Source Folder"
    split_size = 0.8
    dataset_path = "Path to Dataset folder"

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    os.mkdir(dataset_path)
    os.mkdir(os.path.join(dataset_path,"Train"))
    os.mkdir(os.path.join(dataset_path,"Test"))
    os.mkdir(os.path.join(dataset_path,"Validation"))

    for dirname in os.listdir(source_path):
        os.mkdir(os.path.join(dataset_path,"Train",dirname))
        os.mkdir(os.path.join(dataset_path,"Test",dirname))
        os.mkdir(os.path.join(dataset_path,"Validation",dirname)) 

    for dirname in os.listdir(source_path):
        SOURCE_DIR = os.path.join(source_path, dirname)
        TRAINING_DIR = os.path.join(dataset_path,"Train",dirname)
        TESTING_DIR = os.path.join(dataset_path,"Test",dirname)
        VALIDATION_DIR = os.path.join(dataset_path,"Validation",dirname)
        split_data(SOURCE_DIR, TRAINING_DIR, TESTING_DIR, VALIDATION_DIR, split_size)
        #print("{} {} {} {}".format(SOURCE_DIR,TRAINING_DIR,TESTING_DIR,VALIDATION_DIR))



    


    


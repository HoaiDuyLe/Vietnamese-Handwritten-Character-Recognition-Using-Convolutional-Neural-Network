# Vietnamese-Handwritten-Character-Recognition-Using-Convolutional-Neural-Network
My Thesis at Ho Chi Minh City University of Technology, Vietnam National University – Ho Chi Minh

## Usage

First you need to download Vietnamese Handwritten Character and save into directory named Source.

To preprocess data:

1. python splitdata.py

To create tfRecord file:

2. python nist_build_tfRecord.py

To train model:

3. python nist_train.py

To test trained model:

4. python nist_test.py

To use UI for prediction:

5. python nist_gui.py



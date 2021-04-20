# pose-classification

This repo uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and Machine Learning for pose classification based on [TV human interactions dataset](https://www.robots.ox.ac.uk/~vgg/data/tv_human_interactions/). 

### OpenPose

OpenPose algorithm implemented in Python is used for feature extraction from videos. 
[Forked OpenPose in TensorFlow library](https://github.com/jegork/tf-pose-estimation)

### Classification

The goal of this project is to find the difference in accuracy of video classification using different Deep Learning techniques.

The proposed options are:
1. Long Short Term Memory based on features generated by OpenPose
2. 2D Convolutional Network based on features generated by OpenPose
3. VGG16 + LSTM trained directly on raw frames
4. 2D ConvNet with LSTM (directly) 
5. 3D ConvNet (directly)

### Project structure

- utils.py contains functions that download and structure (per directories) the videos
- process_video.py contains functions for feature extraction using OpenPose
- main.ipynb utilizes the previous two python files
- openpose_lstm.ipynb implements algorithm #1
- vgg_lstm.ipynb implements algorithm #3
- 2dcnn_lstm.ipynb implements algorithm #4
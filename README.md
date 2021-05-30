# pose-classification

This repo uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and Machine Learning for pose classification based on [TV human interactions dataset](https://www.robots.ox.ac.uk/~vgg/data/tv_human_interactions/). 

### OpenPose

OpenPose algorithm implemented in Python is used for feature extraction from videos. 
[Forked OpenPose in TensorFlow library](https://github.com/jegork/tf-pose-estimation)

### Task

The goal of this project is to find the difference in accuracy of video classification using different Deep Learning techniques.

The proposed options are:
1. Long Short Term Memory based on features generated by OpenPose
2. 2D Convolutional Network based on features generated by OpenPose
3. VGG16 + LSTM trained directly on raw frames
4. 2D ConvNet with LSTM (directly) 
5. 3D ConvNet (directly) (isn't implemented due to computational limitations)

### Project structure

- utils.py contains functions that download and structure (per directories) the videos
- process_video.py contains functions for feature extraction using OpenPose
- main.ipynb utilizes the previous two python files
- openpose_lstm.ipynb implements algorithm #1
- vgg_lstm.ipynb implements algorithm #3
- 2dcnn_lstm.ipynb implements algorithm #4

### Installation

To use the notebooks, first install needed python packages
```
pip install -r requirements.txt
pip install git+https://github.com/okankop/vidaug
pip install git+https://github.com/jegork/tf-pose-estimation
```

to compile OpenPose (optional, not recommended)
```
git clone https://www.github.com/jegork/tf-pose-estimation
cd tf-pose-estimation
pip install -r requirements.txt

cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

cd ..
cd ..
cd ..

pip install tf-pose-estimation/
```

(optional, if using server) Install codecs needed for opencv (Ubuntu example)
```
apt-get install ffmpeg libsm6 libxext6
```

then run video_utils.py and process_video.py (Linux example below)
```
python video_utils.py
python process_video.py
```
to use VGG16 and CNN-based models, first run video_to_matrix.py
```
python video_to_matrix.py
```
### Results

| Model               | Training (%) | Test (%) |
| ------------------- | ------------ | ------- |
| OpenPose-based (#1) | 33           | 25      |
| VGG16-based (#3)    | 100          | 100     |
| CNN-based (#4)      | 80           | 75      |



<figure>
    <figcaption>Figure 1. Confusion matrix of CNN-based model</figcaption>
    <img src="./graphs/cnn/cnn_confusion_matrix.png" width="500">
</figure>

<figure class="image">
    <figcaption>Figure 2. Training of OpenPose-based model</figcaption>
    <img src="./graphs/openpose/openpose_loss_accuracy.png" width="800">
</figure>

<figure class="image">
    <figcaption>Figure 3. Training of VGG16-based model</figcaption>
    <img src="./graphs/vgg/vgg_loss_accuracy.png" width="800">
</figure>

<figure class="image">
    <figcaption>Figure 4. Training of VGG16-based model</figcaption>
    <img src="./graphs/vgg/vgg_confusion_matrix.png" width="500">
</figure>

### Discussion

By using data augmentation, which generated 600 additional samples, CNN-based model's accuracy went from 75%/40% to 80%/75%!

No matter the extended hyperparameter tuning and changes to the architecture of the OpenPose-based model (adding layers, changing amount of nodes per layer, experimental addition of Convolutional Layers) the model performs very poorly, achieving very low accuracy that is merely better than random guessing. 

Mainly, the poor performance can be attributed to the size of data set: it only included 200 samples which is usually not enough to train a deep neural network (while using more shallow neural nets also did not improve the metrics). 

It can also be attributed to unsuitability of OpenPose to this task. First of all, a lot of frames did not contain some of the body parts (e.g. many frames contained only top body, which produced NA values for lower body parts). As was detected later, each row of our data that was generated by OpenPose contained at least one NA and about 70\% of columns contained at least a single NA value. 

Secondly, as OpenPose was not made to be used for such cases, there was no connection between humans between frames. If there were two persons on a video, there is no way to assign a unique identifier to each human, so the model did not have any idea whether the human on the left side of the first frame was indeed the same human as on the left side of the screen of the second frame. 

Thirdly, the additional noise data caused by the people in the background was hard to get rid of because again, there was no identifier for humans, so if there are four persons, it is hard to tell from the data which ones should be deleted. 

### TODO
- Structure notebooks
- (done) Use config file
- (done) Change folder names
- Fix OpenPose
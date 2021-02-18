import numpy as np
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

def single(video_path, model = 'mobilenet_thin'):
    if model not in ['cmu', 'mobilenet_thin', 'mobilenet_v2_large', 'mobilenet_v2_small']:
        raise Exception('Incompatible model chosen! Available models: cmu, mobilenet_thin, mobilenet_v2_large, mobilenet_v2_small')
    
    cap = cv2.VideoCapture(video_path)
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    e = TfPoseEstimator(get_graph_path(model), target_size=(int(width), int(height)))
    
    points_per_frame = []
    
    if cap.isOpened() is False:
        raise Exception('Error opening file!')
    while cap.isOpened():
        ret, image = cap.read()
        
        if image is None:
            break
        humans = e.inference(image, upsample_size=4.0)
        points = TfPoseEstimator.get_points(image, humans)
        
        points_per_frame.append(points)
        
    return points_per_frame
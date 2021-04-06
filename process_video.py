import numpy as np
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

result_width = 256
result_height = 256

def resize_frame(frame):
    center_x = frame.shape[0] / 2
    center_y = frame.shape[1] / 2
    
    size = frame.shape[0]
    x = center_x - size/2
    y = center_y - size/2
    
    frame = frame[:, int(y):int(y+size)]
    
    frame = cv2.resize(frame,(result_width,result_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
    return frame

def single(video_path, model = 'mobilenet_thin', verbose=10, model_instance=None):
    if model not in ['cmu', 'mobilenet_thin', 'mobilenet_v2_large', 'mobilenet_v2_small', None]:
        raise Exception('Incompatible model chosen! Available models: cmu, mobilenet_thin, mobilenet_v2_large, mobilenet_v2_small')
    
    cap = cv2.VideoCapture(video_path)
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # future support for batch processing of videos using same instance of TfPoseEstimator
    if model_instance == None:
        model_instance = TfPoseEstimator(get_graph_path(model), target_size=(result_width, result_height))
    
    points_per_frame = np.array([])
    frameN = 1
    
    if cap.isOpened() is False:
        raise Exception('Error opening file!')
    while cap.isOpened():
        ret, image = cap.read()
                
        if image is None:
            break
            
        image = resize_frame(image)
        
        humans = model_instance.inference(image, upsample_size=4.0)
        points = TfPoseEstimator.get_points(image, humans)
        
        _zeros = np.full((points.shape[0], 1), frameN)
        _points = np.c_[_zeros, points]

        
        if _points.size != 0:
            if points_per_frame.size == 0:
                    points_per_frame = _points
            else:
                points_per_frame = np.vstack((points_per_frame, _points))

        #            else:
        #                _zeros = np.array([[0, 0, 0, 0, 'NA', 0]])
        #                _zeros[0, 0] = int(_zeros[0, 0]) + frameN
        #                print(_zeros)
        #                points_per_frame = np.vstack((points_per_frame, _zeros))

        # change level of verbosity
        if verbose==0:
            print(f"Frame number {frameN} processed")
        else:
            if frameN % verbose == 0:
                 print(f"Frame number {frameN} processed")
        frameN += 1
    return points_per_frame
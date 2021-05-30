import numpy as np
import cv2
import os
import pandas as pd
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

def transform(video):
    _df = pd.DataFrame(video, columns=['frame', 'human', 'x', 'y', 'bodypart', 'score'])
    
    #convert long to wide format
    _points = _df.pivot(index=['frame', 'human'], columns='bodypart', values=['x', 'y', 'score']).reset_index()


    # change multi-index to single index with underscores
    _points.columns = list(map(lambda x: str(x[1] + "_" + x[0]), _points.columns))
    # remove underscore in first two columns
    _points.columns = _points.columns[2:].insert(0, 'human').insert(0, 'frame')
    
    _points.frame = _points.frame.astype('int64')
    _points.human = _points.human.astype('int64')

    # get column types
    id_columns = _points.columns[:2]
    coord_columns = list(filter(lambda x: x.endswith('x') or x.endswith('y'), list(_points.columns)))
    score_columns = list(filter(lambda x: x.endswith('score'), list(_points.columns)))
    
    #replaces NAs with zeros and cast types
    filled_points = _points.copy(True)
    filled_points[coord_columns] = _points[coord_columns].fillna(0).astype('float64')    
    filled_points[score_columns] = _points[score_columns].fillna(0).astype('float64')
    
    filled_points[score_columns] = filled_points[score_columns].round(2)
    
    filled_points.sort_values(['frame','human'], inplace=True)

    return filled_points


def batch(dir_path, model = 'mobilenet_thin', video_ids=None):
    if model not in ['cmu', 'mobilenet_thin', 'mobilenet_v2_large', 'mobilenet_v2_small']:
        raise Exception('Incompatible model chosen! Available models: cmu, mobilenet_thin, mobilenet_v2_large, mobilenet_v2_small')
        
    e = TfPoseEstimator(get_graph_path(model), target_size=(result_width, result_height))

    videos_list = os.listdir(dir_path)
    videos_list.sort(key=lambda x: int(x.split('.')[0]))
    
    finished_videos = []
    
    if len(videos_list) == 0:
        print("Folder is empty")
        return False
    
    if not os.path.isdir('out'):
        os.mkdir('out')
        
    if video_ids is not None:
        videos_list = [videos_list[i-1] for i in video_ids]
    
    for video in videos_list:
        single_video = single(dir_path+"/"+video, model=None, model_instance=e)
        transformed = transform(single_video)
        
        video_path_no_ext = video.split('.')[0]
        class_name = dir_path.split('/')[1]
        if not os.path.exists(f'out/{class_name}'):
            os.mkdir(f'out/{class_name}')
            
        transformed.to_csv(f'out/{class_name}/{video_path_no_ext}.csv', index=False)
        print(f'Finished {class_name}/{video}')
        finished_videos.append(f'{class_name}/{video}')

    return finished_videos

if __name__ == "__main__":
    batch('videos/highfive')
    batch('videos/handshake')
    batch('videos/hug')
    batch('videos/kiss')
    #batch('videos/negative')
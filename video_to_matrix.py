import tensorflow as tf
import cv2
import shutil
import os
import numpy as np
import video_utils

base = 'videos/'
result_width = 128
result_height = 128

def resize_frame(frame):
    center_x = frame.shape[0] / 2
    center_y = frame.shape[1] / 2
    
    size = frame.shape[0]
    x = center_x - size/2
    y = center_y - size/2
    
    frame = frame[:, int(y):int(y+size)]
    
    frame = cv2.resize(frame,(result_width,result_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
    return frame

def process_class(class_name):
    folder_path = base+class_name
    videos = os.listdir(folder_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    videos.sort(key=lambda x: int(x.split('.')[0]))
    
    len_videos = len(videos)
    videoN = 1
    
    for video in videos:
        if os.path.isfile(folder_path+'/'+video):
            
            input_file = folder_path+'/'+video

            video_name = video.split('.')[0]
            output_folder = 'output/'+class_name

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
                
            cap = cv2.VideoCapture(input_file)
            #width = cap.get(3)
            #height = cap.get(4)
            
            tensors_list = []

            frameN = 1

            if cap.isOpened() is False:
                raise Exception('Error opening file!')
            while cap.isOpened():
                _, image = cap.read()

                if image is None:
                    break

                image = resize_frame(image)
                tensors_list.append(image)

                frameN += 1
            
            tensor = np.array(tensors_list)
            np.save(f'{output_folder+"/"+video_name}.npy', tensor)

            #print(f'Class {class_name}: video {videoN}/{len_videos}')
            videoN += 1
    print(f'Class {class_name} finished')

if __name__ == '__main__':
    process_class('kiss')
    process_class('highfive')
    process_class('handshake')
    process_class('hug')
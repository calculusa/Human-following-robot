import tensorrt as trt
from tensorrt_model import TRTModel
from ssd_tensorrt import load_plugins, parse_boxes,TRT_INPUT_NAME, TRT_OUTPUT_NAME
import ctypes
import numpy as np
import cv2
import os
import ctypes

# Define mean and standard deviation values for image preprocessing 
mean = 255.0 * np.array([0.5, 0.5, 0.5])
stdev = 255.0 * np.array([0.5, 0.5, 0.5])

# Takes a BGR image from the camera
def bgr8_to_ssd_input(camera_value):
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1)).astype(np.float32) # The image dimensions are transposed from (height, width, channels) to (channels, height, width). 
    x -= mean[:, None, None]
    x /= stdev[:, None, None]
    return x[None, ...]

# This class encapsulates an object detection model using TensorRT
class ObjectDetector(object):
    
    def __init__(self, engine_path, preprocess_fn=bgr8_to_ssd_input):
        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, '')
        load_plugins()
        self.trt_model = TRTModel(engine_path, input_names=[TRT_INPUT_NAME],output_names=[TRT_OUTPUT_NAME, TRT_OUTPUT_NAME + '_1'])
        self.preprocess_fn = preprocess_fn
        
    def execute(self, *inputs):
        trt_outputs = self.trt_model(self.preprocess_fn(*inputs))
        return parse_boxes(trt_outputs)
    
    def __call__(self, *inputs):
        return self.execute(*inputs)

model = ObjectDetector('ssd_mobilenet_v2_coco.engine')


#use opencv to covert the depth image to RGB image for displaying purpose
import cv2
import numpy as np

#using realsense to capture the color and depth image
import pyrealsense2 as rs

#multi-threading is used to capture the image in real time performance
import threading

class Camera(SingletonConfigurable):
    
    #this changing of this value will be captured by traitlets
    color_value = traitlets.Any()
    
    def __init__(self):
        super(Camera, self).__init__()
        
        #configure the color and depth sensor
        self.pipeline = rs.pipeline()
        self.configuration = rs.config()  
        
        #set resolution for the color camera
        self.color_width = 640
        self.color_height = 480
        self.color_fps = 30
        self.configuration.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)

        #set resolution for the depth camera
        self.depth_width = 640
        self.depth_height = 480
        self.depth_fps = 30
        self.configuration.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)

        #flag to control the thread
        self.thread_runnning_flag = False
        
        #start the RGBD sensor
        self.pipeline.start(self.configuration)
        self.pipeline_started = True
        frames = self.pipeline.wait_for_frames()

        #start capture the first color image
        color_frame = frames.get_color_frame()   
        image = np.asanyarray(color_frame.get_data())
        self.color_value = image

        #start capture the first depth image
        depth_frame = frames.get_depth_frame()           
        self.depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.depth_value = depth_colormap   

    def _capture_frames(self):
        while(self.thread_runnning_flag==True): #continue until the thread_runnning_flag is set to be False
            frames = self.pipeline.wait_for_frames() #receive data from RGBD sensor
            
            color_frame = frames.get_color_frame() #get the color image
            image = np.asanyarray(color_frame.get_data()) #convert color image to numpy array
            self.color_value = image #assign the numpy array image to the color_value variable 

            depth_frame = frames.get_depth_frame() #get the depth image           
            self.depth_image = np.asanyarray(depth_frame.get_data()) #convert depth data to numpy array
            #conver depth data to BGR image for displaying purpose
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
            self.depth_value = depth_colormap #assign the color BGR image to the depth value
    
    def start(self): #start the data capture thread
        if self.thread_runnning_flag == False: #only process if no thread is running yet
            self.thread_runnning_flag=True #flag to control the operation of the _capture_frames function
            self.thread = threading.Thread(target=self._capture_frames) #link thread with the function
            self.thread.start() #start the thread

    def stop(self): #stop the data capture thread
        if self.thread_runnning_flag == True:
            self.thread_runnning_flag = False #exit the while loop in the _capture_frames
            self.thread.join() #wait the exiting of the thread       

def bgr8_to_jpeg(value):#convert numpy array to jpeg coded data for displaying 
    return bytes(cv2.imencode('.jpg',value)[1])


#create a camera object
camera = Camera.instance()
camera.start() # start capturing the data

import ipywidgets.widgets as widgets
from IPython.display import display, HTML

width = 640
height = 480



import time
from RobotClass import Robot

#initialize the Robot class
robot = Robot()

def processing(change):
    image = change['new']
    tempi = image
    depthi_display = image
    depthi_display2 = image
    imgsized= cv2.resize(image,(300,300))
    # compute all detected objects
    detections = model(imgsized)
    
    matching_detections = [d for d in detections[0] if d['label'] == int(1)]
    
    target_personx = 320
    target_distance = 10000
    
    for det in matching_detections:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
        
        x_min = int(width * bbox[0])
        y_min = int(height * bbox[1])
        x_max = int(width * bbox[2])
        y_max = int(height * bbox[3])
        tempi =image[y_min:y_max,x_min:x_max]
        depthi = camera.depth_image[y_min:y_max,x_min:x_max]
        
        depthi[depthi<100]=0
        depthi[depthi>3000]=0
        depthi_display2 = cv2.applyColorMap(cv2.convertScaleAbs(depthi, alpha=0.03), cv2.COLORMAP_JET)

        # Thresholding the depth image to focus on a specific range
        depthi[0,0]=5000 
        depthi = depthi[depthi!=0]

        # Calculate the minimum depth value (distance)
        distance = depthi.min()
        cv2.putText(image, str(distance), (int((x_min+x_max)/2),int((y_min+y_max)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    
        # Update the target person position and distance if a closer person is detected
        if distance < target_distance:
            target_distance = distance
            target_personx = (x_min + x_max)/2  

    # If no matching detections, stop the robot
    if len(matching_detections)==0:
        robot.stop()
        time.sleep(0.01)
    else:
        # Adjust robot movement based on the target person's position and distance
        if target_personx < 200:
            robot.left(0.7)
            time.sleep(0.01)
        elif target_personx > 440:
            robot.right(0.7)
            time.sleep(0.01)
        else:
            if distance > 700:
                robot.forward(0.5)
            elif distance < 400:
                robot.backward(0.5)
            else:
                robot.stop()
            time.sleep(0.01)
        

    
#the camera.observe function will monitor the color_value variable. If this value changes, the excecute function will be excuted.
camera.observe(processing, names='color_value')


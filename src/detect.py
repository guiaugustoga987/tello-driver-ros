#!/usr/bin/env python3

import rospy
import cv2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge,CvBridgeError
from rostopic import get_topic_type
from sensor_msgs.msg import Image,CompressedImage
import numpy as np
import cv_bridge
import matplotlib.pyplot as plt
from tello_driver.msg import TelloStatus
from ultralytics.yolo.utils.ops import scale_image
import random
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import subprocess
import time
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import Pipeline

class Yolov8Detector:

    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description='Display WLAN signal strength.')
        self.parser.add_argument(dest='interface', nargs='?', default='wlp3s0',
                                help='wlan interface (default: wlan0)')
        self.args = self.parser.parse_args()

        self.bridge = cv_bridge.CvBridge()
        #self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/tello/camera/image_raw',Image, self.callback)
        #self.image_sub = rospy.Subscriber('/tello/image_raw/h264',CompressedImage, self.callback,queue_size=1)
        self.model = YOLO("/home/ga/yolov8/weights/bestv8n.pt")
        self.status_sub = rospy.Subscriber('/tello/status', TelloStatus,self.status,queue_size=1)
        self.sub_odom = rospy.Subscriber('tello/odom', Odometry,self.odom_callback, queue_size=1)
        print('ae')
        self.test = 0
        #self.img = cv2.imread('/home/ga/Documents/5.jpg')
        #cv2.imshow('mask', img)
        #cv2.waitKey(0)
        #print(self.model)
    
    def odom_callback(self,data):
        self.msg2 = data.pose.pose.position.z


    def status(self,data):
        self.msg = data.battery_percentage
        self.msg1 = data.wind_state
        #self.msg2 = data.height_m/10.


    def callback(self, data):
        
        
        #np_arr = np.fromstring(data.data, np.uint8)
        #im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #print(im)
        #global im
        try:
            self.im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            #im = self.bridge.compressed_imgmsg_to_cv2(data)
            #img = cv2.imread('/home/ga/Documents/5.jpg')
        except CvBridgeError as e:
            print(e)
        
        rate = rospy.Rate(100)
        #print(im)
        #print('ta aki')
        
        #img = cv2.imread('/home/ga/Documents/5.jpg')
        #cv2.startWindowThread()
        #cv2.namedWindow("preview")
        #cv2.imshow("preview", im)
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()  # 1 millisecond
        #cv2.destroyAllWindows()
        #print('asd')
        #cv2.destroyAllWindows()
        #cv2.waitKey(1)
        
        #result = self.model(im, verbose=False,imgsz = 320)
        #print(result)
        '''
        if result[0].cpu().masks is None :
            output = np.zeros((480, 640, 3), dtype="uint8")
        else :
            output = (result[0].cpu().masks.data[0].numpy() * 255).astype("uint8")
        '''
        #height1, width1, channel1 = output.shape
        #print(result[0].cpu().masks)
        #print('output', height1, width1, channel1)
        #cv2.imshow('a',output)
        #self.a = output
        #cv2.waitKey(1)  # 1 millisecond
        
        #print('ta aki 2')
        #rate.sleep()

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def trajectory(imagebw,image_original):
    height,width = imagebw.shape

    image_bw = cv2.rotate(imagebw, cv2.ROTATE_90_CLOCKWISE)
    image_bw = cv2.flip(image_bw,1)

    #linha_bw = cv2.cvtColor(image_bw, cv2.COLOR_BGR2GRAY)
    linha_bw = image_bw

    #print('c', linha_bw.shape)
    thresh = cv2.threshold(linha_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    per = []

    coords = np.column_stack(np.where(thresh > 0))

    gy,gx = np.array_split(coords,[-1],1)


    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression(fit_intercept=False))])

    ransac = RANSACRegressor(model, 
                            stop_probability=0.99,
                            max_trials=100,
                            min_samples=50,  
                            residual_threshold=250, 
                            random_state=42
    )
    ransac.fit(gx, gy)

    line_X = np.arange(0, height, 1)

    line_y_ransac = ransac.predict(line_X[:, np.newaxis])

    line_X1 = line_X.reshape(-1, 1)
    model.fit(line_X1,line_y_ransac)
    coeficientes = model.named_steps['linear'].coef_

    a = coeficientes[0][2]
    b = coeficientes[0][1] 
    c = coeficientes[0][0]

    eixo_x = []
    eixo_y = []
    eixo_y = np.arange(0, height, 1)

    for i in range(height) :
        eixo_x.append((a*(i**2) +b*i + c)) 


    y_pto_1 = (10)
    x_pto_1 = eixo_x[10]

    y_pto_2 = (100)
    x_pto_2 = eixo_x[100]

    y_pto_3 = (150)
    x_pto_3 = eixo_x[150]

    y_pto_4 = (220)
    x_pto_4 = eixo_x[220]

    y_pto_5 = (240)
    x_pto_5 = eixo_x[240]

    y_pto_6 = (260)
    x_pto_6 = eixo_x[260] 

    y_pto_7 = (330)
    x_pto_7 = eixo_x[330]

    #print(output)

    cv2.circle(image_original, (int(x_pto_6),int(y_pto_6)), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(image_original, (int(x_pto_5),int(y_pto_5)), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(image_original, (int(x_pto_7),int(y_pto_7)), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(image_original, (int(x_pto_1),int(y_pto_1)), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(image_original, (int(x_pto_2),int(y_pto_2)), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(image_original, (int(x_pto_3),int(y_pto_3)), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(image_original, (int(x_pto_4),int(y_pto_4)), radius=3, color=(0, 0, 255), thickness=-1)
    return image_original,eixo_x

def get_signal(args):
    cmd = subprocess.Popen('iwconfig %s' % args.interface, shell=True,
                        stdout=subprocess.PIPE)
    #print(cmd.stdout)
    for line in cmd.stdout:
        if b'Link Quality' in line:
            a = line.decode('utf-8')
            #print(a)
            b = a.split ()

            c = b[1]
            d = int(c[8:10])
            e = int(c[11:13])

            signal = (d/e)*100
            signal = round(signal,2)
    
    return signal


def calc_erro(eixo_x,width,height,image_combined):

    x_pto_240 = eixo_x[240]
    f = 692.8183639889837

    ang = 21.26 # Camera tilting related to y axis.
    ang = (ang*np.pi)/180

    alt = 4.5
    x = alt/np.cos(ang)
    x1 = (width*x)/f
    conv1 = (1*x1)/width

    erro = x_pto_240 - width/2
    erro_ze = erro*(conv1)

    f = 'Erro Lateral: {:.2f}'.format(erro_ze)
    cv2.putText(image_combined,f, (20, height-100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return erro_ze,image_combined

def process():
    rospy.init_node("yolov8", anonymous=True)
    ic = Yolov8Detector()
    while not rospy.is_shutdown():
        frame = ic.im
        size = 448
        

        frame = cv2.resize(frame, (size, size))
        h, w, _ = frame.shape
        #print(frame.shape)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.flip(frame,1)

        
        #result = ic.model(ic.im, verbose=False,imgsz = 448)[0]
        results = ic.model.predict(frame, verbose=True,imgsz = size,conf = 0.7)
        class_names = ic.model.names
        #print('Class Names: ', class_names)
        #colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
        colors = [[0,0,255]]
        #print(colors)
        #print(ic.msg)

        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs

        if masks is not None :
            
            output_bw = (results[0].cpu().masks.data[0].numpy() * 255).astype("uint8")
            masks = masks.data.cpu()
            #masks = masks[0].data.cpu() # Só a máscara de maior probabilidade
            frame,eixo_x = trajectory(output_bw,frame)
            
            
            # rescale masks to original image
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                image_combined = overlay(frame, seg, colors[int(box.cls)], 0.4)
            
            erro_ze,image_combined = calc_erro(eixo_x,w,h,image_combined)

        else :
            image_combined = frame
                
            output_bw = np.zeros((size, size, 3), dtype="uint8")

        
        signal = get_signal(ic.args)

        height,width,channel = frame.shape
        c = 'Battery: {}%'.format(ic.msg)
        d = 'Wind: {}'.format(ic.msg1)
        e = 'Signal: {:.2f}%'.format(signal)
        cv2.putText(image_combined, c, (20, height-40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image_combined, d, (20, height-60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image_combined, e, (20, height-80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("preview", image_combined)
        cv2.waitKey(1)

        '''
        if result[0].cpu().masks is None :
            output = np.zeros((480, 640, 3), dtype="uint8")
        else :
            output = (result[0].cpu().masks.data[0].numpy() * 255).astype("uint8")
        '''
        

if __name__ == "__main__":

    
    rospy.init_node("yolov8", anonymous=True)
    detector = Yolov8Detector()
    process()
    #rospy.spin()
    
    while not rospy.is_shutdown():
        rospy.spin()
    


    
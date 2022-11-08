#!/usr/bin/env python
#coding:utf-8

import rospy
import roslib
import numpy as np
import sys, time
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge,CvBridgeError

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

def publish(image_np,ros_data):
    global image_pub
    #### Create CompressedIamge ####

    msg = CompressedImage()

    msg.header.stamp = ros_data.header.stamp 
    # header.frame_id = ros_data.frame_id
    # msg.height=image_size[1]
    # msg.width=image_size[0]
    # msg.encoding='bgr8'
    # msg.header=ros_data.header
    # msg.step=ros_data.step
    
    msg.format = "jpeg"

    msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tobytes()

    # Publish new image
    global image_pub
    image_pub.publish(msg)



def callback(ros_data):
    print(1)
    global model, palette, opacity,image_size,model_size
    np_arr = np.fromstring(ros_data.data, np.uint8)
    
    image_np = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
    
    # image_np = cv2.resize(image_np, model_size, interpolation = cv2.INTER_AREA)
    # result = inference_segmentor(model, image_np)
    # draw_img = model.show_result(
    #             image_np,
    #             result,
    #             palette=get_palette(palette),
    #             show=False,
    #             opacity=opacity)
    
    # cv2.imshow("raw",draw_img)
   
    # draw_img = cv2.resize(draw_img, image_size, interpolation = cv2.INTER_LINEAR)
    cv2.imshow("raw",image_np)
    cv2.waitKey(1)
    # publish(draw_img,ros_data)

def showImage():
    global model, palette, opacity,image_pub,image_size,model_size
    model_size = (1024, 1024)  # width * height
    image_size = (1280,1024)
    open_mmlab = '/home/qian/open-mmlab/mmsegmentation'
    # cgnet 轻量级但不准
    # config = open_mmlab + '/configs/cgnet/cgnet_680x680_60k_cityscapes.py'
    # checkpoint = open_mmlab + '/checkpoints/cgnet_680x680_60k_cityscapes_20201101_110253-4c0b2f2d.pth'
    # segformer 慢但准
    config = open_mmlab + '/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'
    checkpoint = open_mmlab + '/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
    
    
    device = 'cuda:0'
    palette = 'cityscapes'
    opacity = 0.5
    rospy.init_node('showImage',anonymous = True)
    # bridge = CvBridge()
    rospy.Subscriber('/camera/image_color', Image, callback)    
    image_pub = rospy.Publisher("/output/image_raw/compressed",CompressedImage, queue_size = 10)
    
    # model = init_segmentor(config, checkpoint, device=device)
    rospy.spin()

if __name__ == '__main__':
    showImage()

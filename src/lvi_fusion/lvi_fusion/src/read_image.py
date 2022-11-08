#!/usr/bin/env python
#coding:utf-8

import rospy
import roslib
import numpy as np
import sys, time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

def publish(image,ros_data):
    global bridge,image_pub
    #### Create Iamge ####
    msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    # msg = Image()
    # header.frame_id = ros_data.frame_id
    msg.header.stamp = ros_data.header.stamp 
    # image_temp.height=image_size[1]
    # image_temp.width=image_size[0]
    # image_temp.encoding='bgr8'
    # msg.header=ros_data.header
    # msg.step=ros_data.step
    # msg.format = "jpeg"

    # msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tobytes()

    # Publish new image
    image_pub.publish(msg)

def callback(data):
    global bridge,model, palette, opacity,image_size,model_size
    cv_image = bridge.imgmsg_to_cv2(data,desired_encoding="passthrough")
    # cv_image = cv_image[0:1000,:]  # 只对urbannav数据集使用
    cv_image = cv2.resize(cv_image, model_size, interpolation = cv2.INTER_AREA)
    result = inference_segmentor(model, cv_image)
    draw_img = model.show_result(
                cv_image,
                result,
                palette=get_palette(palette),
                show=False,
                opacity=opacity)
    
  
   
    draw_img = cv2.resize(draw_img, image_size, interpolation = cv2.INTER_LINEAR)
    # cv2.imshow("raw",draw_img)
    # cv2.waitKey(1)
    publish(draw_img,data)
    
def showImage():
    global bridge,model, palette, opacity,image_pub,image_size,model_size
    model_size = (1024, 1024)  # width * height
    image_size = (1920,1200)
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
    bridge = CvBridge() 
    rospy.Subscriber('/wideangle/image_color', Image, callback)  #/wideangle/image_color  #/camera/image_color
    image_pub = rospy.Publisher("/output/image_raw",Image, queue_size = 10)
    
    model = init_segmentor(config, checkpoint, device=device)
    rospy.spin()

if __name__ == '__main__':
    showImage()

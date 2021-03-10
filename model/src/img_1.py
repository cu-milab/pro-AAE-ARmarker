#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import sys
import time
import numpy as np
import random
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *
import tf.transformations as tft
import math

#from annotation import anno

class image_converter:
    def __init__(self):

        rospy.init_node('image_converter', anonymous=True)
        self.bridge = CvBridge()
        sub_rgb = message_filters.Subscriber("/camera2/color/image_raw", Image)
        sub_depth = message_filters.Subscriber("/camera/depth/image_raw", Image)
        self.mf = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], 100, 10.0)
        self.mf.registerCallback(self.ImageCallback)

        self.max = 162.635
        self.min = 128


    def ImageCallback(self, rgb_data, depth_data):
        try:
            color_image = self.bridge.imgmsg_to_cv2(rgb_data, 'passthrough')
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, 'passthrough')
        except CvBridgeError ,e:
        
        
        
                   rospy.logerr(e)

        color_image.flags.writeable = True
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, c = color_image.shape

        self.color_image = color_image
        self.depth_image = depth_image*255
        #print("convert_image")


    def SaveImage(self, num):
        #cv2.normalize(depth_image, depth_image, 0, 1, cv2.NORM_MINMAX)
        depth_image = self.ImgNormalize(self.depth_image)
        cv2.imwrite("../img_0131/hyouka_heimen/hi/"+ str(num).zfill(6) +".png", self.color_image)
        #cv2.imwrite("../img/AAE_img/"+ str(num).zfill(6) +".png", depth_image)
        #cv2.imwrite("../img/depth/"+ str(num).zfill(6) +".png", self.depth_image)
        print("save_img"+ str(num).zfill(6))

    def ImgNormalize(self, img):
        img = (img - self.min)/(self.max-self.min)
        return img*255


if __name__ == '__main__':
    try:
        ic = image_converter()
        #rospy.spin()
    except rospy.ROSInterruptException:
        pass

    #rospy.init_node("spawn_products_in_bins")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    model_list = ["ARt0_20","AR0_30","AR0_40"]#"ARt_20",]
    model_name = model_list[0]

   
    
    xyz = [0.5, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]

    # 乱数範囲
    min_xyz = [0.8,  0,  0.5]
    max_xyz = [0.8,  0,  0.5]
    min_rpy = [0, -0.3, -0.30]
    max_rpy = [6, 0.3, 0.3]

 

    while not rospy.is_shutdown():


        #random.seed(seed_number)
        for i in range(0,1):
            with open("../models/"+ model_name +"/model.sdf", "r") as f:
                product_xml = f.read()


            for j in range(3):
                xyz[j] = random.uniform(min_xyz[j], max_xyz[j])
                rpy[j] = random.uniform(min_rpy[j], max_rpy[j])

            item_pose = Pose()
            item_pose.position.x = 0.8
            item_pose.position.y = xyz[1]
            #item_pose.position.z = xyz[0.5]
            item_pose.position.z = 0.5

            tmpq = tft.quaternion_from_euler(rpy[0],rpy[1],rpy[2])
            rad_r= rpy[0]*(180/math.pi)
            rad_p= rpy[1]*(180/math.pi)
            rad_y= rpy[2]*(180/math.pi)
            q = Quaternion(tmpq[0],tmpq[1],tmpq[2],tmpq[3])
            item_pose.orientation = q;

            #x_min = 256 - (xyz[1]*656) - 40
            #x_max = 256 - (xyz[1]*656) + 40
            #y_min = 212 - (xyz[0] - 0.5)* 613.3  - 40
            #y_max = 212 - (xyz[0] - 0.5)* 613.3  + 40

            print("spawn_model")
            spawn_model(model_name, product_xml, "", item_pose, "world")

            #print("database")
            time.sleep(0.3)
            #save_name = m+ str(i) 
            save_name = "r"+str(round(rad_r))+"p"  +str(round(rad_p)) +"y"+  str(round(rad_y))
            ic.SaveImage(save_name)
            #anno(str(i).zfill(6), model_name, x_min, x_max, y_min, y_max)

            delete_model(model_name)
            print("delete_model")
            #time.sleep(1.0)

        break

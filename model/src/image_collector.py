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
        cv2.imwrite("../img_0115/AR4/"+ str(num).zfill(6) +".png", self.color_image)
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

    model_list = ["AR4_40","ARt4_30","ARt4_40","ARt4_20","ARt4_30","ARt4_40"]#"ARt_20",]
    #model_name = model_list[0]

    for model_name in model_list:
        
        xyz = [0.5, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]

        # 乱数範囲
        min_xyz = [0.8,  0,  0.5]
        max_xyz = [0.8,  0,  0.5]
        min_rpy = [0, -0.80, -0.30]
        max_rpy = [5.5, 0.80, 0.3]

        while not rospy.is_shutdown():

            if model_name == "AR4_20":
                seed_number = 10
                model_name_1 =  "AR4_20_"
                cx = 0.8
            elif model_name == "AR4_30":
                seed_number = 11
                model_name_1 = "AR4_30_"
                cx = 0.81
            elif model_name == "AR4_50":
                seed_number = 12
                model_name_1 = "AR4_50_"
                cx = 0.88
            elif model_name == "ARt4_20":
                seed_number = 10
                model_name_1 =  "ARt4_20_"
                cx = 0.82
            elif model_name == "ARt4_30":
                seed_number = 11
                model_name_1 = "ARt4_30_"
                cx = 0.83
            else:
                seed_number = 12
                model_name_1 = "ARt4_50_"
                cx = 0.9

            random.seed(seed_number)
            for i in range(0,2000):
                with open("../models/"+ model_name +"/model.sdf", "r") as f:
                    product_xml = f.read()


                for j in range(3):
                    xyz[j] = random.uniform(min_xyz[j], max_xyz[j])
                    rpy[j] = random.uniform(min_rpy[j], max_rpy[j])

                item_pose = Pose()
                item_pose.position.x = cx
                item_pose.position.y = xyz[1]
                #item_pose.position.z = xyz[0.5]
                item_pose.position.z = 0.5

                tmpq = tft.quaternion_from_euler(rpy[0],rpy[1],rpy[2])
                q = Quaternion(tmpq[0],tmpq[1],tmpq[2],tmpq[3])
                item_pose.orientation = q;

                #x_min = 256 - (xyz[1]*656) - 40
                #x_max = 256 - (xyz[1]*656) + 40
                #y_min = 212 - (xyz[0] - 0.5)* 613.3  - 40
                #y_max = 212 - (xyz[0] - 0.5)* 613.3  + 40

                print("spawn_model")
                spawn_model(model_name, product_xml, "", item_pose, "world")

                #print("database")
                time.sleep(0.2)
                save_name = model_name_1 + str(i) 
                ic.SaveImage(save_name)
                #anno(str(i).zfill(6), model_name, x_min, x_max, y_min, y_max)

                delete_model(model_name)
                print("delete_model")
                #time.sleep(1.0)

            break

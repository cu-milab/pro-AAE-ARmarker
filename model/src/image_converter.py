#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import sys
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class image_converter:
    def __init__(self):

        rospy.init_node('image_converter', anonymous=True)
        self.bridge = CvBridge()
        sub_rgb = message_filters.Subscriber("/camera2/color/image_raw", Image)
        sub_depth = message_filters.Subscriber(
            "/camera/depth/image_raw", Image)
        self.mf = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth], 100, 10.0)
        self.mf.registerCallback(self.ImageCallback)

    def ImageCallback(self, rgb_data, depth_data):
        try:
            color_image = self.bridge.imgmsg_to_cv2(rgb_data, 'passthrough')
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, 'passthrough')
        except CvBridgeError, e:
            rospy.logerr(e)

        color_image.flags.writeable = True
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, c = color_image.shape



        '''
        cv2.namedWindow("color_image")
        cv2.namedWindow("depth_image")
        cv2.imshow("color_image", color_image)
        cv2.imshow("depth_image", depth_image)
        cv2.waitKey(10)
        '''

        #cv2.normalize(depth_image, depth_image, 0, 1, cv2.NORM_MINMAX)
        cv2.imwrite("../img/color_test/color_image.png", color_image)
        cv2.imwrite("../img/depth_test/depth_image.png", depth_image*255)
        print("save image")

if __name__ == '__main__':
    try:
        ic = image_converter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

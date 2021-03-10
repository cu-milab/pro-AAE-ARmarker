#!/usr/bin/env python
# $ python spawn_model.py

import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *
import tf.transformations as tft

if __name__ == '__main__':
    print("Waiting for gazebo services...")
    rospy.init_node("spawn_products_in_bins")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    print("Got it.")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    model_list = ["gum_tape", "box", "cup"]
    model_name = model_list[2]

    with open("../models/"+ model_name +"/model.sdf", "r") as f:
        product_xml = f.read()

    xyz = [0.3855, -0.175, 0.0]
    rpy = [0.0, 0.0, 0.0]

    item_pose = Pose()
    item_pose.position.x = xyz[0]
    item_pose.position.y = xyz[1]
    item_pose.position.z = xyz[2]

    tmpq = tft.quaternion_from_euler(rpy[0],rpy[1],rpy[2])
    q = Quaternion(tmpq[0],tmpq[1],tmpq[2],tmpq[3])
    item_pose.orientation = q;

    spawn_model(model_name, product_xml, "", item_pose, "world")


    #delete_model(model_name)

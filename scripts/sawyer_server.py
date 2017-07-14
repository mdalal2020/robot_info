#!/usr/bin/env python

from robot_info.srv import *
import rospy
import numpy as np
from baxter_pykdl import baxter_kinematics

def handle_get_jacobian(req):
    right_kin = baxter_kinematics('right')
    jacobian = kin.jacobian().getA()
    return GetJacobianResponse(jacobian[0], jacobian[1], jacobian[2], jacobian[3], jacobian[4], jacobian[5])
def get_jacobian_server():
    rospy.init_node('get_jacobian_server')
    s = rospy.Service('get_jacobian', GetJacobian, handle_get_jacobian)
    rospy.spin()

if __name__ == "__main__":
    get_jacobian_server()

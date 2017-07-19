#!/usr/bin/env python

from robot_info.srv import *
import rospy
import numpy as np
from urdf_parser_py.urdf import URDF
import intera_interface as ii
from pykdl_utils.kdl_kinematics import KDLKinematics

def handle_get_robot_pose_jacobian(req):
    arm = ii.Limb(req.name)
    q = arm.joint_angles()
    q = [q[req.name + '_j0'], q[req.name + '_j1'], q[req.name + '_j2'], q[req.name + '_j3'], q[req.name + '_j4'],
         q[req.name + '_j5'], q[req.name + '_j6']]
    pose = kin.forward(q, end_link=req.name + req.tip)
    pose = np.squeeze(np.asarray(pose))
    pose = [pose[0][3], pose[1][3], pose[2][3]]
    jacobian = kin.jacobian(q).getA()
    return getRobotPoseAndJacobianResponse(pose, jacobian[0], jacobian[1], jacobian[2], jacobian[3], jacobian[4], jacobian[5])

def get_robot_pose_jacobian_server():
    rospy.init_node('get_robot_pose_jacobian_server')
    global kin
    robot = URDF.from_parameter_server(key='robot_description')
    base_link = 'base'
    end_link = 'right_hand'
    kin = KDLKinematics(robot, base_link, end_link)
    s = rospy.Service('get_robot_pose_jacobian', getRobotPoseAndJacobian, handle_get_robot_pose_jacobian)
    rospy.spin()

if __name__ == "__main__":
    get_robot_pose_jacobian_server()

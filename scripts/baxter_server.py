#!/usr/bin/env python

from robot_info.srv import *
import rospy
import numpy as np
from urdf_parser_py.urdf import URDF
import baxter_interface as bi
from pykdl_utils.kdl_kinematics import KDLKinematics

def handle_get_robot_pose_jacobian(req):
    if req.name == 'right':
        kin = right_kin
    else:
        kin = left_kin
    arm = bi.Limb(req.name)
    q = arm.joint_angles()
    q = [q[req.name + '_s0'], q[req.name + '_s1'], q[req.name + '_e0'], q[req.name + '_e1'], q[req.name + '_w0'],
         q[req.name + '_w1'], q[req.name + '_w2']]
    pose = kin.forward(q, end_link=req.name + req.tip)
    pose = np.squeeze(np.asarray(pose))
    pose = [pose[0][3], pose[1][3], pose[2][3]]
    jacobian = kin.jacobian(q).getA()
    return getRobotPoseAndJacobianResponse(pose, jacobian[0], jacobian[1], jacobian[2], jacobian[3], jacobian[4], jacobian[5])

def get_robot_pose_jacobian_server():
    rospy.init_node('get_robot_pose_jacobian_server')
    global right_kin
    robot = URDF.from_parameter_server(key='robot_description')
    base_link = 'base'
    right_end_link = 'right_gripper'
    right_kin = KDLKinematics(robot, base_link, right_end_link)
    global left_kin
    left_end_link = 'left_gripper'
    left_kin = KDLKinematics(robot, base_link, left_end_link)
    s = rospy.Service('get_robot_pose_jacobian', getRobotPoseAndJacobian, handle_get_robot_pose_jacobian)
    rospy.spin()

if __name__ == "__main__":
    get_robot_pose_jacobian_server()
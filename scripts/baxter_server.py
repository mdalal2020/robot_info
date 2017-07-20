#!/usr/bin/env python

from robot_info.srv import *
import rospy
import numpy as np
from urdf_parser_py.urdf import URDF
import baxter_interface as bi
from pykdl_utils.kdl_kinematics import KDLKinematics

joints = [
    'right_upper_shoulder',
    'right_lower_shoulder',
    'right_upper_elbow',
    'right_lower_elbow',
    'right_upper_forearm',
    'right_lower_forearm',
    'right_wrist',
    'right_gripper',
    'left_upper_shoulder',
    'left_lower_shoulder',
    'left_upper_elbow',
    'left_lower_elbow',
    'left_upper_forearm',
    'left_lower_forearm',
    'left_wrist',
    'left_gripper'
]
kin_dict = {}
arms_dict = {}

def handle_get_robot_pose_jacobian(req):
    joint = req.name + req.tip
    kin = kin_dict[joint]
    arm = arms_dict[req.name]
    q = arm.joint_angles()
    q = [q[req.name + '_s0'], q[req.name + '_s1'], q[req.name + '_e0'], q[req.name + '_e1'], q[req.name + '_w0'],
         q[req.name + '_w1'], q[req.name + '_w2']]
    pose = kin.forward(q)
    pose = np.squeeze(np.asarray(pose))
    pose = [pose[0][3], pose[1][3], pose[2][3]]
    jacobian = kin.jacobian(q).getA()
    return getRobotPoseAndJacobianResponse(pose, jacobian[0], jacobian[1], jacobian[2], jacobian[3], jacobian[4], jacobian[5])

def get_robot_pose_jacobian_server():
    rospy.init_node('get_robot_pose_jacobian_server')
    robot = URDF.from_parameter_server(key='robot_description')

    for joint in joints:
        kin_dict[joint] = KDLKinematics(robot, 'base', joint)

    arms_dict['right'] = bi.Limb('right')
    arms_dict['left'] = bi.Limb('left')

    s = rospy.Service('get_robot_pose_jacobian', getRobotPoseAndJacobian, handle_get_robot_pose_jacobian)
    rospy.spin()

if __name__ == "__main__":
    get_robot_pose_jacobian_server()
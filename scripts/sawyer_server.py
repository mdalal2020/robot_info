#!/usr/bin/env python

from robot_info.srv import *
import rospy
import numpy as np
from urdf_parser_py.urdf import URDF
import baxter_interface as bi
from pykdl_utils.kdl_kinematics import KDLKinematics

kin_dict = {}
arms_dict = {}

def handle_get_robot_pose_jacobian(req):
    joint = req.name + req.tip
    kin = kin_dict[req.name]
    arm = arms_dict[req.name]
    q = arm.joint_angles()
    q = [q[req.name + '_j0'], q[req.name + '_j1'], q[req.name + '_j2'], q[req.name + '_j3'], q[req.name + '_j4'],
         q[req.name + '_j5'], q[req.name + '_j6']]
    pose = kin.forward(q, joint)
    pose = np.squeeze(np.asarray(pose))
    pose = [pose[0][3], pose[1][3], pose[2][3]]
    jacobian = kin.jacobian(q, pose).getA()
    return getRobotPoseAndJacobianResponse(pose, jacobian[0], jacobian[1], jacobian[2], jacobian[3], jacobian[4], jacobian[5])

def get_robot_pose_jacobian_server():
    rospy.init_node('get_robot_pose_jacobian_server')
    robot = URDF.from_parameter_server(key='robot_description')

    kin_dict['right'] = KDLKinematics(robot, 'base', 'right_gripper')
    kin_dict['left'] = KDLKinematics(robot, 'base', 'left_gripper')

    arms_dict['right'] = bi.Limb('right')
    arms_dict['left'] = bi.Limb('left')

    s = rospy.Service('get_robot_pose_jacobian', getRobotPoseAndJacobian, handle_get_robot_pose_jacobian)
    rospy.spin()

if __name__ == "__main__":
    get_robot_pose_jacobian_server()